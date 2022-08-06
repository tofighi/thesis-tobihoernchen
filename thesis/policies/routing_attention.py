from torch import nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from typing import Tuple, Callable
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class AttentionACPolicy(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        latent_pi_reshaped = latent_pi.view(
            latent_pi.shape[:-1]
            + (self.features_extractor.max_fleetsize, self.features_extractor.embed_dim)
        )
        return super()._get_action_dist_from_latent(latent_pi_reshaped)

    def _build(self, lr_schedule: Schedule) -> None:
        ret = super()._build(lr_schedule)
        self.action_net = nn.Sequential(
            nn.Linear(self.features_extractor.embed_dim, 5, bias=False),
            nn.Softmax(-1),
            nn.Flatten(1),
        )
        return ret


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, n_heads=8, fn_mask=None):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True, bias=False
        )
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.fn_mask = fn_mask

    def forward(self, x):
        if self.fn_mask is not None:
            mask = self.fn_mask()
            mask = mask.repeat_interleave(self.n_heads, dim=0)
        else:
            mask = None
        attended = self.attention(x, x, x, need_weights=False, attn_mask=mask == 0,)[
            0
        ].nan_to_num(0)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        return x


class RoutingFE(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        max_fleetsize,
        embed_dim=64,
        n_heads=8,
        depth=6,
    ):
        super().__init__(observation_space, max_fleetsize * embed_dim)
        self.embed_dim = embed_dim
        self.n_obs_per_agv = observation_space.shape[0] // max_fleetsize
        self.max_fleetsize = max_fleetsize
        self.mask = None

        def get_mask():
            m = self.mask
            # return torch.bmm(m, m.transpose(1, 2)).detach()
            return m.repeat(1, 1, m.shape[1]).detach()

        self.embedding = nn.Sequential(
            nn.Linear(self.n_obs_per_agv, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 4, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim, bias=False),
            nn.ReLU(),
        )

        ablocks = []
        for i in range(depth):
            ablocks.append(
                AttentionBlock(embed_dim=embed_dim, n_heads=n_heads, fn_mask=get_mask)
            )
        self.ablocks = nn.Sequential(*ablocks)

    def forward(self, x: torch.Tensor):
        reshaped = x.view(x.shape[:-1] + (self.max_fleetsize, self.n_obs_per_agv))
        self.mask = reshaped[:, :, 0, None]
        reshaped = reshaped * self.mask.detach()
        x = self.embedding(reshaped)
        x = self.ablocks(x)
        return torch.flatten(x, start_dim=1)
