from torch import nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, n_heads=8, fn_mask=None):
        super().__init__()
        self.fn_mask = fn_mask
        self.n_heads = n_heads

        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True, bias=False
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fn_mask = fn_mask

    def forward(self, x):
        if self.fn_mask is not None:
            mask = self.fn_mask()
            mask = mask.repeat_interleave(self.n_heads, dim=0)
        else:
            mask = None
        attended = self.attention(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=mask,
        )[0]

        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        # x = x * self.fn_mask().detach()
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
            return (m.repeat(1, 1, m.shape[1]) + torch.eye(m.shape[1])) == 0

        self.embedd = nn.Sequential(
            nn.Linear(self.n_obs_per_agv, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
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
        self.actionmask = reshaped[:, :, 1, None]
        reshaped.masked_fill(self.mask.repeat(1, 1, self.n_obs_per_agv) == 0, 0)
        x = self.embedd(reshaped)
        x = self.ablocks(x)
        x.masked_fill(self.actionmask.repeat(1, 1, self.embed_dim) == 0, 0)
        return x
