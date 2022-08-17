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

        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
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


class Nodewise_Embedder(nn.Module):
    def __init__(self, not_node_obs=3, node_len=2, embed_len=10, n_nodes=30) -> None:
        super().__init__()
        self.not_node_obs = not_node_obs
        self.embedder = nn.Embedding(n_nodes, embed_len)
        self.nodes = -torch.ones(n_nodes, node_len).detach()
        self.node_hwm = 0
        self.out_dim = lambda n: not_node_obs + n * embed_len
        self.embed_len = embed_len

    def register(self, nodes):
        start = self.node_hwm
        for node in nodes:
            self.nodes[self.node_hwm] = node
            self.node_hwm += 1
        return torch.arange(start, self.node_hwm).to(dtype=torch.int32)

    def get_indices(self, nodes: torch.Tensor):
        nodes_cmp = nodes[:, None, :].repeat(1, self.nodes.shape[0], 1)
        found = torch.all(
            torch.isclose(nodes_cmp, self.nodes.repeat(len(nodes), 1, 1), 0.1), axis=2
        )
        indices_picker = torch.arange(0, len(self.nodes)).repeat(len(nodes), 1)
        indices = torch.max(
            torch.where(found, indices_picker, -torch.ones_like(indices_picker)), axis=1
        )[0]
        return indices.to(dtype=torch.int32).detach()

    def forward(self, x: torch.Tensor):
        not_node = x[:, :, : self.not_node_obs]
        nodes = x[:, :, self.not_node_obs :]
        shape_original = nodes.shape[:1]
        n_nodes = int(nodes.shape.numel() / 2)
        nodes = nodes.reshape((n_nodes, 2))
        indices = self.get_indices(nodes)
        indices[indices == -1] = self.register(nodes[indices == -1])
        return self.embedder(indices).reshape(shape_original + (self.embed_len,))


class RoutingFE(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        max_fleetsize,
        n_features=None,
        embed_dim=64,
        n_heads=8,
        depth=6,
    ):
        n_features = embed_dim * max_fleetsize if n_features is None else n_features
        super().__init__(observation_space, n_features)
        self.embed_dim = embed_dim
        if isinstance(observation_space, spaces.Box):
            self.n_obs_per_agv = observation_space.shape[0] // max_fleetsize
        elif isinstance(observation_space, spaces.Dict):
            self.n_obs_per_agv = (
                observation_space.spaces["observation"].shape[0] // max_fleetsize
            )
        self.max_fleetsize = max_fleetsize
        self.mask = None
        self.aye = None

        def get_mask():
            m = self.mask
            # return torch.bmm(m, m.transpose(1, 2)).detach()
            return m.repeat(1, 1, m.shape[1]) + self.aye == 0

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
                AttentionBlock(
                    embed_dim=embed_dim, n_heads=n_heads
                )  # , fn_mask=get_mask)
            )
        self.ablocks = nn.Sequential(*ablocks)

    def forward(self, x: torch.Tensor):
        if isinstance(x, dict):
            x = x["observation"]
        reshaped = x.view(x.shape[:-1] + (self.max_fleetsize, self.n_obs_per_agv))
        self.mask = reshaped[:, :, 0, None]
        if self.aye is None:
            self.aye = torch.eye(self.mask.shape[1], device=self.mask.device)
        self.actionmask = reshaped[:, :, 1, None]
        # reshaped = reshaped.masked_fill(self.mask.repeat(1, 1, self.n_obs_per_agv) == 0, 0)
        x = self.embedd(reshaped)
        x = self.ablocks(x)
        # x = x.masked_fill(self.actionmask.repeat(1, 1, self.embed_dim) == 0, 0)
        return x


class RoutingFE_offPolicy(RoutingFE):
    def __init__(
        self,
        observation_space: spaces.Box,
        max_fleetsize,
        n_features=None,
        embed_dim=64,
        n_heads=8,
        depth=6,
    ):
        super().__init__(
            observation_space, max_fleetsize, embed_dim, embed_dim, n_heads, depth
        )

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        x = x.max(dim=1)[0]
        return x
