import torch

from activeuf.acquisition_function.base import BaseAcquisitionFunction


class MaxMinLCB(BaseAcquisitionFunction):
    """
    MaxMin Lower Confidence Bound (LCB) acquisition function for active
    dueling/preference selection.

    For each prompt, constructs pairwise reward difference and joint uncertainty
    matrices, then selects the completion pair (chosen, rejected) that maximizes
    the minimum LCB across opponents. Optionally prunes clearly suboptimal arms
    via a UCB-based candidate set before solving the max-min problem.

    Assumes upper and lower bounds are symmetric around the reward estimate.
    """

    def __init__(
        self,
        beta: float = 1.0,
        argmax_tol: float = 1e-4,
        decision_buffer: float = 0.0,
        use_candidate_set: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.beta = beta
        self.argmax_tol = argmax_tol
        self.decision_buffer = decision_buffer
        self.use_candidate_set = use_candidate_set

        if seed is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(0)
        else:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

    def __call__(
        self,
        rewards: torch.Tensor,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> list[list[int, int]]:
        """
        Args:
            rewards: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the reward scores for each completion
            lower_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the lower bounds for each completion
            upper_bounds: tensor of shape (n_prompts, n_completions_per_prompt)
                containing the upper bounds for each completion
        Returns:
            list[list[int, int]]: The selected indices per prompt.
                The first index is the one that should be chosen, the second
                one is the one that should be rejected.
        """
        std_deviation = (upper_bounds - lower_bounds) / 2

        selected_ids_batch = []
        for i in range(len(rewards)):
            # Shape: (n_completions_per_prompt,)
            rewards_i = rewards[i].cpu().numpy()
            # Shape: (n_completions_per_prompt,)
            std_i = std_deviation[i].cpu().numpy()

            # Create pairwise difference matrix: rewards[i] - rewards[j]
            # Shape: (n_completions, n_completions)
            reward_diff_matrix = rewards_i[:, None] - rewards_i[None, :]

            # Create pairwise uncertainty matrix: std[i] + std[j]
            # Shape: (n_completions, n_completions)
            uncertainty_matrix = std_i[:, None] + std_i[None, :]

            arm_i, arm_j = self._max_min_lcb(
                torch.tensor(reward_diff_matrix, dtype=torch.float32),
                torch.tensor(uncertainty_matrix, dtype=torch.float32),
            )
            selected_ids_batch.append((int(arm_i), int(arm_j)))

        return selected_ids_batch

    def _max_min_lcb(
        self, posterior_mean: torch.Tensor, posterior_var: torch.Tensor
    ) -> tuple[int, int]:
        """
        Computes the max-min LCB acquisition function.

        Args:
            posterior_mean: tensor of shape (n_completions_per_prompt, n_completions_per_prompt)
                containing the posterior means for each completion pair
            posterior_var: tensor of shape (n_completions_per_prompt, n_completions_per_prompt)
                containing the posterior variances for each completion pair

        Returns:
            tuple: Indices of the arms to select.
        """
        lcb = posterior_mean - self.beta * posterior_var  # Shape: (n_arms, n_arms)
        n = lcb.shape[0]

        # Set values to nan for arms that are clearly suboptimal
        if self.use_candidate_set:
            ucb = posterior_mean + self.beta * posterior_var  # Shape: (n_arms, n_arms)
            candidate_arms_mask = torch.all(
                torch.logical_or(
                    ucb > -self.decision_buffer,
                    torch.diag(torch.full((n,), True, dtype=torch.bool)),
                ),
                dim=1,
            )  # Shape: (n_arms,)
            # Make sure you do not consider the same arms at once, Shape: (n_arms, )
            lcb = torch.where(
                candidate_arms_mask[:, None] * candidate_arms_mask[None, :],
                lcb,
                torch.tensor(float("nan")),
            )
            lcb = torch.where(
                torch.eye(lcb.shape[0], dtype=torch.bool),
                torch.tensor(float("nan")),
                lcb,
            )
        else:
            candidate_arms_mask = torch.ones(n, dtype=torch.bool)
            lcb = torch.where(torch.eye(n, dtype=torch.bool), float("nan"), lcb)

        # Manual implementation of nanmin for older PyTorch versions
        def nanmin_1d(tensor):
            """Compute min ignoring NaN values along the last dimension."""
            mask = ~torch.isnan(tensor)
            if not mask.any(dim=1).all():
                # If any row has all NaNs, handle it
                result = torch.full((tensor.shape[0],), float("inf"))
                for i in range(tensor.shape[0]):
                    if mask[i].any():
                        result[i] = tensor[i][mask[i]].min()
                return result
            else:
                return torch.stack(
                    [tensor[i][mask[i]].min() for i in range(tensor.shape[0])]
                )

        def nanmax_0d(tensor):
            """Compute max ignoring NaN values for a 1D tensor."""
            mask = ~torch.isnan(tensor)
            if mask.any():
                return tensor[mask].max()
            else:
                return torch.tensor(float("-inf"))

        def nanargmax_0d(tensor):
            """Compute argmax ignoring NaN values for a 1D tensor."""
            mask = ~torch.isnan(tensor)
            if mask.any():
                valid_tensor = tensor.clone()
                valid_tensor[~mask] = float("-inf")
                return torch.argmax(valid_tensor)
            else:
                return torch.tensor(0)  # Default to 0 if all NaN

        min_j = nanmin_1d(lcb)  # Shape: (n_arms, )

        # Create argmin_j with random tie-breaking
        lcb_diff = torch.abs(lcb - min_j[:, None])
        tie_mask = lcb_diff < self.argmax_tol

        # Create random values for tie-breaking
        random_values = torch.randperm(
            n**2, generator=self.generator, dtype=torch.float32
        ).view(n, n)
        argmin_j_set = torch.where(tie_mask, random_values, torch.tensor(-float("inf")))
        argmin_j = torch.argmax(argmin_j_set, dim=1)

        maxmin_lcb = nanmax_0d(min_j)  # Shape: ()

        def choose_next_arms():
            # Create random values for tie-breaking in argmax
            argmax_ties = torch.abs(min_j - maxmin_lcb) < self.argmax_tol
            random_argmax = torch.randperm(
                n, generator=self.generator, dtype=torch.float32
            )
            argmax_set = torch.where(
                argmax_ties, random_argmax, torch.tensor(float("nan"))
            )
            next_arm_i = nanargmax_0d(argmax_set)
            next_arm_j = argmin_j[next_arm_i]
            return next_arm_i, next_arm_j

        # Replace jax.lax.cond with regular conditional
        if torch.sum(candidate_arms_mask) == 1:
            single_candidate = torch.argmax(candidate_arms_mask.float())
            return single_candidate, single_candidate
        else:
            return choose_next_arms()
