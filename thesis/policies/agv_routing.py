import torch.nn as nn
import torch as th
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AgvRoutingFE(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: spaces.Box, max_fleetsize, with_transformer=False
    ):
        self.n_agv = max_fleetsize
        self.n_obs_per_agv = observation_space.shape[0] // max_fleetsize
        self.with_transformer = with_transformer
        n_embedded = 32
        self.n_embedded = n_embedded
        features_dim = (n_embedded + (1 if with_transformer else 0)) * self.n_agv
        super().__init__(observation_space, features_dim)

        self.lin_each_agv = nn.Sequential(
            nn.Linear(self.n_obs_per_agv, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_embedded),
            nn.ReLU(),
        )
        if with_transformer:
            self.transformer = nn.Transformer(
                n_embedded + 1,
                1,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=64,
            )
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        reshaped = th.reshape(
            observations, observations.shape[:-1] + (self.n_agv, self.n_obs_per_agv)
        )
        self.mask = reshaped[:, :, 0, None].repeat(1, 1, self.n_embedded)
        embedded = self.lin_each_agv(reshaped)
        if self.with_transformer:
            pos_encodings = th.zeros(embedded.shape[:-1] + (1,))
            pos_encodings[:] = th.arange(0, embedded.shape[-2]).unsqueeze(1)
            pos_encoded = th.concat([pos_encodings, embedded], dim=-1)
            transformed = self.transformer(pos_encoded, pos_encoded)
        return self.flatten(transformed if self.with_transformer else embedded)

    def _save(self, path: str):
        th.save(self.lin_each_agv.state_dict(), path)

    def _load(self, path: str):
        self.lin_each_agv.load_state_dict(th.load(path))
        self.lin_each_agv.eval()

    def _lock(self):
        for param in self.parameters():
            param.requires_grad = False

    def unlock(self):
        for param in self.parameters():
            param.requires_grad = True
