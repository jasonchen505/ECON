# src/components/action_selectors.py
# -*- coding: utf-8 -*-
import torch
from typing import Any

class MultinomialActionSelector:
    def __init__(self, args: Any):
        self.args = args

    def select_action(self, agent_inputs: torch.Tensor, avail_actions: torch.Tensor, t_env: int, test_mode: bool = False):
        """
        agent_inputs: (B, N, A)  —— logits
        avail_actions: (B, N, A) 或 (B, N, 1)
        """
        if avail_actions.size(-1) == 1 and agent_inputs.size(-1) > 1:
            avail_actions = avail_actions.expand_as(agent_inputs)


        masked_q = agent_inputs.clone()
        masked_q[avail_actions == 0] = -1e9
        masked_q = torch.nan_to_num(masked_q, nan=0.0, posinf=1e4, neginf=-1e4)

        probs = torch.softmax(masked_q, dim=-1)
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        safe_uniform = 1.0 / probs.size(-1)
        probs = torch.where(probs_sum <= 0, torch.full_like(probs, safe_uniform), probs / probs_sum)

        if test_mode:
            actions = probs.argmax(dim=-1)  # (B,N)
        else:
            m = torch.distributions.Categorical(probs=probs)
            actions = m.sample()            # (B,N)
        return actions

REGISTRY = {
    "multinomial": MultinomialActionSelector
}