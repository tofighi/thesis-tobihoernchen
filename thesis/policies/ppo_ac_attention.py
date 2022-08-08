from torch import nn
import torch
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

class CustomValueNet(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Lin = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            # nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = x.max(dim=1)[0]
        return self.Lin(x)


class CustomActionNet(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Lin = nn.Sequential(
            nn.Linear(embed_dim, 5, bias=False),
            nn.Softmax(-1),
            nn.Flatten(1),
        )

    def forward(self, x: torch.Tensor):
        return self.Lin(x)


class AttentionACPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule: Schedule) -> None:
        ret = super()._build(lr_schedule)
        self.action_net = CustomActionNet(self.features_extractor.embed_dim)
        self.value_net = CustomValueNet(self.features_extractor.embed_dim)
        return ret