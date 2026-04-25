from dataclasses import asdict
from datasets import Dataset
import torch
from transformers import TrainerCallback
import wandb
import os
import json
import pickle
import random
import numpy as np

from rewarduq.methods.mlp_head_ensemble import (
    MLPHeadEnsembleModel as ENNRewardModel,
    MLPHeadEnsembleModelConfig as ENNRewardModelConfig,
    MLPHeadEnsembleTrainer as ENNRewardModelTrainer,
    MLPHeadEnsembleTrainerConfig as ENNRewardModelTrainerConfig,
    MLPHeadEnsemblePipeline as ENNRewardModelPipeline,
)

from activeuf.loop.arguments import ENNConfig


def main_process_only(f, accelerator):
    """Decorator to ensure the wrapped logging function runs only on the main process."""

    def wrapper(*args, **kwargs):
        if accelerator.is_main_process:
            return f(*args, **kwargs)
        return None

    return wrapper


def custom_collate(batch):
    out = {}
    for key in [
        "prompt_id",
        "prompt",
        "source",
        "completions",
        "features",
    ]:
        if key in batch[0]:
            out[key] = [x[key] for x in batch]
    return out


def custom_decollate(collated_batch):
    out = []
    for i in range(len(collated_batch["prompt_id"])):
        item = {
            "prompt_id": collated_batch["prompt_id"][i],
            "prompt": collated_batch["prompt"][i],
            "source": collated_batch["source"][i],
            "completions": collated_batch["completions"][i],
        }
        if "features" in collated_batch:
            item["features"] = collated_batch["features"][i]
        out.append(item)
    return out


def compute_acquisition_function_KPIs(rewards, chosen_idxs, rejected_idxs):
    mean_rewards_per_sample = rewards.mean(dim=1)  # (n_samples, 2)

    chosen_rewards = rewards.gather(
        1, chosen_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))
    ).squeeze(1)
    rejected_rewards = rewards.gather(
        1, rejected_idxs.unsqueeze(-1).expand(-1, -1, rewards.size(-1))
    ).squeeze(1)

    # Add to KPIs
    kpis = {
        "mean_rewards_per_sample": mean_rewards_per_sample[:, 0].tolist(),
        "mean_uncertainty_per_sample": mean_rewards_per_sample[:, 1].tolist(),
        "chosen_rewards_per_sample": chosen_rewards[:, 0].tolist(),
        "chosen_uncertainty_per_sample": chosen_rewards[:, 1].tolist(),
        "rejected_rewards_per_sample": rejected_rewards[:, 0].tolist(),
        "rejected_uncertainty_per_sample": rejected_rewards[:, 1].tolist(),
    }
    return kpis


WANDB_LOGS_CACHE = []
TRAINER_LOGS_CACHE = []
MAX_TRAINER_LOGS_CACHE_SIZE = None


class WandbStepLoggerCallback(TrainerCallback):
    def __init__(self, accelerator):
        self.accelerator = accelerator

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Custom callback for wandb logging with caches. This callback
        waits until the trainer's log cache hits the specified maximum size.
        Then, it aggregates the cached trainer logs (by taking mean),
        and lets the aggregated logs "piggyback" with the final log in
        the wandb cache. It ignores logs made at the end of each epoch,
        by checking for the key 'train_runtime'.
        """
        global WANDB_LOGS_CACHE, TRAINER_LOGS_CACHE

        if self.accelerator.is_main_process and logs and "train_runtime" not in logs:
            TRAINER_LOGS_CACHE.append(logs)

            if len(TRAINER_LOGS_CACHE) == MAX_TRAINER_LOGS_CACHE_SIZE:
                # aggregate trainer logs by taking mean
                mean_trainer_logs = {}
                keys = {k for x in TRAINER_LOGS_CACHE for k in x}
                for k in keys:
                    values = [x.get(k) for x in TRAINER_LOGS_CACHE]
                    values = [_ for _ in values if _ is not None]
                    mean_trainer_logs[k] = sum(values) / len(values)

                for key in ["regularization_towards_initial_weights"]:
                    mean_trainer_logs[key] = getattr(args, key)

                # let current logs piggyback on the last entry in the cache
                if WANDB_LOGS_CACHE:
                    WANDB_LOGS_CACHE[-1].update(mean_trainer_logs)
                else:
                    WANDB_LOGS_CACHE.append(mean_trainer_logs)

                for _logs in WANDB_LOGS_CACHE:
                    wandb.log(_logs)

                # clear caches
                WANDB_LOGS_CACHE = []
                TRAINER_LOGS_CACHE = []


def init_model_trainer(
    reward_model_type: str, reward_args: ENNConfig | None, n_processes: int
):
    if reward_model_type == "none":
        model, trainer = None, None

    elif reward_model_type == "enn":
        trainer_config = ENNRewardModelTrainerConfig(
            per_device_train_batch_size=-(
                reward_args.effective_batch_size // -n_processes
            ),
            **asdict(reward_args.trainer),
        )

        if reward_args.previous_checkpoint_path:
            model = ENNRewardModel.from_pretrained(
                reward_args.previous_checkpoint_path,
            )
            tokenizer = model.tokenizer
        else:
            pipeline = ENNRewardModelPipeline(
                ENNRewardModelConfig(**asdict(reward_args.model)),
                trainer_config,
            )
            model = pipeline.model
            tokenizer = model.tokenizer

        # initialize trainer with a dummy Dataset. So we have access to the uq_pipeline.trainer before entering the loop.
        # The dataset must have raw columns (prompt, chosen, rejected) because the new rewarduq
        # version runs prepare_preference_dataset during __init__.
        trainer = ENNRewardModelTrainer(
            args=trainer_config,
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(
                [
                    {
                        "prompt": "dummy",
                        "chosen": "dummy",
                        "rejected": "dummy",
                    }
                ]
            ),
        )
    else:
        raise NotImplementedError(f"{reward_model_type=} not implemented.")

    return model, trainer


def compute_rewards_with_uncertainty_bounds(
    samples, model, tokenizer, inference_batch_size
) -> torch.tensor:
    n_samples = len(samples)
    n_completions_per_sample = len(samples[0]["completions"])

    if model is None:
        return torch.zeros(
            (n_samples, n_completions_per_sample, 3),
            dtype=torch.float32,
        )

    has_precomputed_features = "features" in samples[0]

    if has_precomputed_features:
        # Use precomputed features
        def get_features_yielder():
            for sample in samples:
                for i in range(n_completions_per_sample):
                    yield torch.tensor(sample["features"][i])

        features_yielder = get_features_yielder()
        rewards_batch = []
        while True:
            features_mbatch = []
            for _ in range(inference_batch_size):
                try:
                    features_mbatch.append(next(features_yielder))
                except StopIteration:
                    break
            if not features_mbatch:
                break
            features_mbatch = torch.stack(features_mbatch).to(model.device)

            with torch.no_grad():
                output = model(features=features_mbatch)

            rewards_batch.extend(output["rewards"].cpu())
    else:
        # Tokenize and compute features on-the-fly
        all_input_ids = []
        for sample in samples:
            for completion in sample["completions"]:
                if isinstance(sample["prompt"], list):
                    prompt_messages = sample["prompt"]
                else:
                    prompt_messages = [{"role": "user", "content": sample["prompt"]}]
                conversation = prompt_messages + [{"role": "assistant", "content": completion["response_text"]}]

                tokenized = tokenizer.apply_chat_template(
                    conversation, tokenize=True, return_tensors="pt"
                ).squeeze(0)
                all_input_ids.append(tokenized)

        rewards_batch = []
        features_batch = []
        for start in range(0, len(all_input_ids), inference_batch_size):
            batch_ids = all_input_ids[start : start + inference_batch_size]
            padded = tokenizer.pad(
                {"input_ids": batch_ids},
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                output = model(
                    input_ids=padded["input_ids"],
                    attention_mask=padded["attention_mask"],
                    output_features=True,
                )

            rewards_batch.extend(output["rewards"].cpu())
            features_batch.extend(output["features"].cpu())

        # Store computed features back in samples for downstream use (replay buffer)
        idx = 0
        for sample in samples:
            sample["features"] = []
            for _ in sample["completions"]:
                sample["features"].append(features_batch[idx])
                idx += 1

    torch.cuda.empty_cache()
    rewards_batch = torch.stack(rewards_batch).view(n_samples, -1, 3)

    return rewards_batch


def get_acquired(samples, acquired_idxs):
    acquired = []
    for sample, (a, b) in zip(samples, acquired_idxs):
        assert a != b
        completions = sample["completions"]

        acquired.append(
            {
                "prompt_id": sample["prompt_id"],
                "prompt": sample["prompt"],
                "source": sample["source"],
                "response_text_1": completions[a]["response_text"],
                "features_1": sample["features"][a],
                "model_1": completions[a]["model"],
                "score_1": completions[a]["overall_score"],
                "response_text_2": completions[b]["response_text"],
                "features_2": sample["features"][b],
                "model_2": completions[b]["model"],
                "score_2": completions[b]["overall_score"],
            }
        )
    return acquired


def compute_kpis(rewards, acquired_idxs) -> list[dict]:
    _rewards, _lower_bounds, _upper_bounds = rewards.unbind(-1)
    _uncertainty = (_upper_bounds - _lower_bounds) / 2  # half-width of the confidence interval
    _chosen_idxs, _rejected_idxs = acquired_idxs.unbind(-1)

    index = torch.arange(_rewards.size(0))
    mean_rewards_per_sample = _rewards.mean(dim=1)
    mean_uncertainty_per_sample = _uncertainty.mean(dim=1)

    kpis = []
    for i in range(_rewards.size(0)):
        kpi = {
            "mean_rewards_per_sample": mean_rewards_per_sample[i].item(),
            "mean_uncertainty_per_sample": mean_uncertainty_per_sample[i].item(),
            "chosen_rewards_per_sample": _rewards[index, _chosen_idxs][i].item(),
            "rejected_rewards_per_sample": _rewards[index, _rejected_idxs][i].item(),
            "chosen_uncertainty_per_sample": _uncertainty[index, _chosen_idxs][
                i
            ].item(),
            "rejected_uncertainty_per_sample": _uncertainty[index, _rejected_idxs][
                i
            ].item(),
        }
        kpi["reward_differences_per_sample"] = (
            kpi["chosen_rewards_per_sample"] - kpi["rejected_rewards_per_sample"]
        )
        kpis.append(kpi)
    return kpis


def restructure_sample(x: dict) -> dict:
    if isinstance(x["prompt"], list):
        prompt_messages = x["prompt"]
    else:
        prompt_messages = [{"role": "user", "content": x["prompt"]}]
    for key in ["chosen", "rejected"]:
        x[key] = prompt_messages + [{"role": "assistant", "content": x[key]}]
    return x


def get_new_regularization(
    n_done: int,
    n_total: int,
    decay_type: str,
    initial_value: float,
    exponential_decay_base: float = None,
    exponential_decay_scaler: float = None,
) -> float:
    if decay_type == "linear":
        return initial_value * (1.0 - n_done / n_total)
    elif decay_type == "exponential":
        frac_done = n_done / n_total
        exponent = exponential_decay_scaler * frac_done
        return initial_value * (exponential_decay_base**exponent)
    else:
        raise ValueError(f"{decay_type=} not supported")


def save_loop_checkpoint(
    save_dir: str, 
    args, 
    loop_state: dict, 
    replay_buffer, 
    output_data: list, 
    trainer=None,
    model=None
):
    """Saves loop state, buffer, data, AND RNG states."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Standard Data
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)

    with open(os.path.join(save_dir, "loop_state.json"), "w") as f:
        json.dump(loop_state, f, indent=2)

    with open(os.path.join(save_dir, "replay_buffer.pkl"), "wb") as f:
        pickle.dump(replay_buffer, f)
    
    with open(os.path.join(save_dir, "output_list.pkl"), "wb") as f:
        pickle.dump(output_data, f)

    # 2. Save Model/Trainer
    if trainer is not None:
        trainer.save_model(save_dir) 
        trainer.save_state() 
        if trainer.optimizer is not None:
            torch.save(trainer.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        if trainer.lr_scheduler is not None:
            torch.save(trainer.lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
    elif model is not None:
        model.save_pretrained(save_dir)

    # --- NEW: SAVE RNG STATES ---
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    with open(os.path.join(save_dir, "rng_states.pkl"), "wb") as f:
        pickle.dump(rng_states, f)


def load_loop_checkpoint(checkpoint_dir: str):
    """Loads loop state and returns data + RNG states dict."""
    
    # 1. Load Data
    with open(os.path.join(checkpoint_dir, "loop_state.json"), "r") as f:
        loop_state = json.load(f)

    with open(os.path.join(checkpoint_dir, "replay_buffer.pkl"), "rb") as f:
        replay_buffer = pickle.load(f)

    with open(os.path.join(checkpoint_dir, "output_list.pkl"), "rb") as f:
        output_data = pickle.load(f)

    # 2. Load RNG States (but do NOT apply them yet)
    rng_states = None
    rng_path = os.path.join(checkpoint_dir, "rng_states.pkl")
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng_states = pickle.load(f)
            
    return loop_state, replay_buffer, output_data, rng_states