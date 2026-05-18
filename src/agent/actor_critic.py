import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        action_dim: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def get_dist(self, s_x: torch.Tensor) -> Categorical:
        logits = self.actor(s_x)
        return Categorical(logits=logits)

    def get_value(self, s_x: torch.Tensor) -> torch.Tensor:
        return self.critic(s_x).squeeze(-1)

    def get_action_and_value(
        self,
        s_x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(s_x)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(s_x)

        return action, log_prob, entropy, value


__all__ = ["ActorCritic"]
