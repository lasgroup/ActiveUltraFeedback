#!/bin/bash

#SBATCH --job-name=loop
#SBATCH -D .
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH --output=./logs/O-%x.%j
#SBATCH --error=./logs/E-%x.%j
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=04:00:00
#SBATCH --environment=activeuf_dev

export GPUS_PER_NODE=4
export HF_HOME=${HF_HOME:-/path/to/hf_cache}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Defaults (override via environment variables)
ACQ_FUNC="${ACQ_FUNC:?ERROR: ACQ_FUNC not set}"
REPLAY_BUFFER="${REPLAY_BUFFER:-100}"
EXP_DECAY_BASE="${EXP_DECAY_BASE:-0.9}"
ACQ_BETA="${ACQ_BETA:-1.0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:--}"

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

RESUME_ARG=""
if [[ "$RESUME_CHECKPOINT" != "-" ]]; then
    RESUME_ARG="--resume_from_checkpoint=$RESUME_CHECKPOINT"
fi

export CMD="accelerate launch \
    --config_file=configs/accelerate/multi_node.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    -m activeuf.loop.run \
    --config_path configs/loop.yaml \
    --acquisition_function_type=$ACQ_FUNC \
    --replay_buffer_factor=$REPLAY_BUFFER \
    --enn.regularization.exponential_decay_base=$EXP_DECAY_BASE \
    --acquisition_function.beta=$ACQ_BETA \
    $RESUME_ARG"

srun $CMD
