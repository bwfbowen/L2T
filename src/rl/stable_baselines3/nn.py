import torch 
import torch.nn as nn 
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PSExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, 
                 features_dim: int = 16, 
                 sol_input_dim: int = 12,
                 hidden_dim: int = 64,
                 num_heads: int = 4):
        super().__init__(observation_space, features_dim=features_dim)

        self.sol_embed_head = nn.Sequential(*[
            nn.Conv1d(sol_input_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim)
        ])
        self.sol_self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.sol_residual = nn.Sequential(*[
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1)
        ])
        self.sol_norm_after = nn.BatchNorm1d(hidden_dim)
        self.ln = nn.Linear(hidden_dim, sol_input_dim)
    
    def forward(self, observations):
        solution_features, problem_features = observations['solution'], observations['problem']
        solution_features = torch.transpose(solution_features, 1, 2)
        sol_embed = self.sol_embed_head(solution_features)
        sol_embed = torch.transpose(sol_embed, 1, 2)
        sol_embed, _ = self.sol_self_attn(sol_embed, sol_embed, sol_embed)
        sol_embed = torch.transpose(sol_embed, 1, 2)
        identity = sol_embed
        sol_out = self.sol_residual(sol_embed)
        sol_out += identity
        sol_out = self.sol_norm_after(sol_out)
        sol_out = torch.sum(sol_out, dim=2)
        sol_out = self.ln(sol_out)
        out = torch.concat([sol_out, problem_features], dim=1)
        return out 