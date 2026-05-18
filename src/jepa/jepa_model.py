import copy

import torch
from torch import nn

from .encoder import SpatioTemporalViTEncoder
from .predictor import Predictor


class JEPAModel(nn.Module):
    def __init__(
        self,
        img_height: int = 400,
        img_width: int = 300,
        num_frames: int = 3,
        in_channels: int = 3,
        embed_dim: int = 64,
        action_dim: int = 2,
        ema_momentum: float = 0.99,
        patch_size: int = 25,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        predictor_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.ema_momentum = ema_momentum

        self.x_encoder = SpatioTemporalViTEncoder(
            img_height=img_height,
            img_width=img_width,
            num_frames=num_frames,
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.y_encoder = copy.deepcopy(self.x_encoder)
        self.y_encoder.eval()
        for param in self.y_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(
            action_dim=action_dim,
            latent_dim=embed_dim,
            hidden_dim=predictor_hidden_dim,
        )

    @torch.no_grad()
    def update_y_encoder(self) -> None:
        momentum = self.ema_momentum
        for x_param, y_param in zip(self.x_encoder.parameters(), self.y_encoder.parameters()):
            y_param.data.mul_(momentum).add_(x_param.data, alpha=1.0 - momentum)

        for x_buffer, y_buffer in zip(self.x_encoder.buffers(), self.y_encoder.buffers()):
            y_buffer.copy_(x_buffer)

    def forward(
        self,
        frames_x: torch.Tensor,
        frames_y: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_x = self.x_encoder(frames_x)
        with torch.no_grad():
            s_y = self.y_encoder(frames_y)

        s_y_pred = self.predictor(s_x, action)
        return s_x, s_y, s_y_pred


__all__ = ["JEPAModel"]
