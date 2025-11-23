# src/modules/mixer/mix_llm.py
# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _nan_to_num(x: torch.Tensor, nan: float = 0.0, posinf: float = 1e6, neginf: float = -1e6) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a, dim=dim, eps=eps)
    b = F.normalize(b, dim=dim, eps=eps)
    return (a * b).sum(dim=dim).clamp(-1.0, 1.0)


class LLMQMixer(nn.Module):
    """
    Centralized mixing network with agent-level attention and commitment alignment.
    """

    def __init__(self, args: Any):
        super().__init__()
        self.n_agents = int(getattr(args, "n_agents", 3))
        d_belief = int(getattr(args, "belief_dim", 128))
        h = int(getattr(args.arch, "mlp_hidden_size", 128))
        cfg_heads = int(getattr(getattr(args, "mixer", object()), "attention_heads", 4))
        n_heads = cfg_heads if h % max(1, cfg_heads) == 0 else 1

        self.use_reward_context = bool(getattr(getattr(args, "mixer", object()), "use_reward_context", True))

        # Projections
        self.proj_belief = nn.Linear(d_belief, h)
        self.proj_prompt = nn.Linear(2, h)
        self.proj_group = nn.Linear(d_belief, h)

        # Agent-to-agent attention over fused features
        self.attn_in = nn.Linear(2 * h, h)
        self.attn = nn.MultiheadAttention(embed_dim=h, num_heads=max(1, n_heads), batch_first=True)
        self.attn_norm = nn.LayerNorm(h)

        # Agent weights + aggregation
        self.agent_weight = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Linear(h, 1)
        )
        self.mix_mlp = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 1)
        )

        # Optional commitment projection created lazily to match dim
        self.proj_commit: Optional[nn.Linear] = None

        if self.use_reward_context:
            self.proj_reward = nn.Sequential(
                nn.Linear(1, h),
                nn.Sigmoid()
            )

    def forward(
        self,
        q_local: torch.Tensor,  # (B,T,N)
        belief_states: torch.Tensor,  # (B,T,N,d_b)
        prompt_embeddings: torch.Tensor,  # (B,T,N,2)
        group_repr: torch.Tensor,  # (B,T,d_b)
        commitment_embedding: Optional[torch.Tensor] = None,  # (B,T,d_c) or (B,1,d_c)
        reward_ctx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B, T, N = q_local.shape

        b = _nan_to_num(self.proj_belief(belief_states))          # (B,T,N,h)
        e = _nan_to_num(self.proj_prompt(prompt_embeddings))      # (B,T,N,h)
        g = _nan_to_num(self.proj_group(group_repr))              # (B,T,h)
        g_exp = g.unsqueeze(2).expand(B, T, N, g.size(-1))        # (B,T,N,h)

        # Agent interaction via attention
        fused = torch.cat([b, e], dim=-1)                         # (B,T,N,2h)
        fused = _nan_to_num(self.attn_in(fused))                  # (B,T,N,h)
        attn_out, _ = self.attn(
            fused.view(B * T, N, -1),
            fused.view(B * T, N, -1),
            fused.view(B * T, N, -1)
        )
        attn_out = attn_out.view(B, T, N, -1)
        agent_ctx = self.attn_norm(attn_out + fused)              # (B,T,N,h)

        # Optional reward context gating
        if self.use_reward_context and reward_ctx is not None:
            r_ctx = reward_ctx
            if r_ctx.dim() == 2:
                r_ctx = r_ctx.unsqueeze(1)
            if r_ctx.dim() == 3 and r_ctx.size(1) != T:
                r_ctx = r_ctx.expand(B, T, -1)
            r_gate = self.proj_reward(r_ctx)                     # (B,T,h)
            agent_ctx = agent_ctx * (0.7 + 0.3 * r_gate.unsqueeze(2))

        # Agent mixing weights
        mix_input = torch.cat([agent_ctx, g_exp], dim=-1)         # (B,T,N,2h)
        weights = self.agent_weight(mix_input).squeeze(-1)        # (B,T,N)
        weights = F.softmax(weights, dim=-1)

        # Aggregate Q and context
        q_bar = (weights * q_local).sum(dim=-1, keepdim=True)     # (B,T,1)
        mix_ctx = (weights.unsqueeze(-1) * agent_ctx).sum(dim=2)  # (B,T,h)
        mix_ctx = mix_ctx + g                                     # (B,T,h)
        q_tot = self.mix_mlp(mix_ctx) + q_bar                     # (B,T,1)

        # Loss terms
        q_local_mean = q_local.mean(dim=-1, keepdim=True)
        l_cons = (q_tot - q_local_mean).pow(2).mean()

        if commitment_embedding is not None:
            d_c = commitment_embedding.size(-1)
            if (self.proj_commit is None) or (self.proj_commit.in_features != d_c):
                self.proj_commit = nn.Linear(d_c, mix_ctx.size(-1)).to(commitment_embedding.device)
            c = commitment_embedding
            if c.dim() == 2:
                c = c.unsqueeze(1)
            if c.size(1) != T:
                c = c.expand(B, T, -1)
            c_proj = _nan_to_num(self.proj_commit(c))             # (B,T,h)
            sim = _cosine_sim(mix_ctx.view(B * T, -1), c_proj.view(B * T, -1), dim=-1, eps=1e-6)
            l_align = (1.0 - sim).mean()
        else:
            l_align = q_tot.new_zeros(())

        return q_tot, {"consistency_loss": l_cons, "align_loss": l_align}
