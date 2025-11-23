
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

class BeliefEncoder(nn.Module):

    def __init__(self, belief_dim: int, n_agents: int, n_heads: int = 4, 
                 key_dim: int = 64, device: torch.device = None):
        super(BeliefEncoder, self).__init__()
        self.belief_dim = belief_dim
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.multihead_attn = nn.MultiheadAttention(embed_dim=belief_dim, num_heads=n_heads, batch_first=True)
        self.out_proj = nn.Linear(belief_dim, belief_dim)
        self.layer_norm = nn.LayerNorm(belief_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(belief_dim, 4 * belief_dim),
            nn.ReLU(),
            nn.Linear(4 * belief_dim, belief_dim)
        )
        self.final_layer_norm = nn.LayerNorm(belief_dim)

    def forward(self, belief_states: torch.Tensor) -> torch.Tensor:
        # belief_states: [batch_size, n_agents, belief_dim]
        attn_output, _ = self.multihead_attn(query=belief_states, key=belief_states, value=belief_states)
        attn_output = self.layer_norm(belief_states + attn_output)
        ff_output = self.feedforward(attn_output)
        ff_output = self.final_layer_norm(attn_output + ff_output)
        group_repr = ff_output.mean(dim=1)  # [batch_size, belief_dim]
        group_repr = self.out_proj(group_repr)
        return group_repr

    def compute_loss(self, td_loss_tot: torch.Tensor, td_losses_i: List[torch.Tensor], lambda_e: float) -> torch.Tensor:
        return td_loss_tot + lambda_e * sum(td_losses_i)