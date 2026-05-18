import torch
from torch import nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(
        self,
        action_dim: int = 2,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.predictor_layer1 = nn.Linear(latent_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim, bias=False)
        self.predictor_layer2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, 
                s_x, # (b, latent_dim)
                action, # (b,) or (b, action_dim)
                **kwargs):
        if action.dim() == 2 and action.shape[-1] == 1:
            action = action.squeeze(-1)

        if action.dim() == 1:
            action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        elif action.dim() == 2 and action.shape[-1] == self.action_dim:
            action_one_hot = action.float()
        else:
            raise ValueError(
                "action must have shape (B,) discrete ids or (B, action_dim) one-hot vectors"
            )

        act_emb = self.action_proj(action_one_hot)
        hidden = self.predictor_layer1(s_x) + act_emb
        hidden = F.relu(hidden)
        s_y_pred = self.predictor_layer2(hidden) 
        return s_y_pred


__all__ = ["Predictor"]