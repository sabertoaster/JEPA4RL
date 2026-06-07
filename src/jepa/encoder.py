import torch
from torch import nn
import torch.nn.functional as F


class SpatioTemporalViTEncoder(nn.Module):
    def __init__(
        self,
        img_height: int = 400,
        img_width: int = 300,
        num_frames: int = 3,
        in_channels: int = 3,
        embed_dim: int = 64,
        patch_size: int = 25,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if img_height % patch_size != 0 or img_width % patch_size != 0:
            raise ValueError("patch_size must divide img_height and img_width")

        self.img_height = img_height
        self.img_width = img_width
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.grid_h = img_height // patch_size
        self.grid_w = img_width // patch_size
        self.patches_per_frame = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.row_pos = nn.Parameter(torch.zeros(1, self.grid_h, embed_dim))
        self.col_pos = nn.Parameter(torch.zeros(1, self.grid_w, embed_dim))
        self.time_pos = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cls_pos, std=0.02)
        nn.init.normal_(self.row_pos, std=0.02)
        nn.init.normal_(self.col_pos, std=0.02)
        nn.init.normal_(self.time_pos, std=0.02)

    def _spatio_temporal_positional_encoding(self) -> torch.Tensor:
        # NOTE that H and W in this scenario is grid_h and grid_w (kinda like the number of patches along height and width)
        row = self.row_pos[:, :, None, :]  # Shape: (1, H, 1, D)
        col = self.col_pos[:, None, :, :]  # Shape: (1, 1, W, D)
        spatial = (row + col).reshape(
            1, self.patches_per_frame, self.embed_dim
        )  # Shape: (1, HxW, D)

        temporal = self.time_pos[:, :, None, :]  # Shape: (1, num_frames=3, 1, D)
        spatial = spatial[:, None, :, :]  # Shape: (1, 1, H*W, D)

        pos = temporal + spatial  # Shape: (1, 3, H*W, D)
        return pos.reshape(
            1, self.num_frames * self.patches_per_frame, self.embed_dim
        )  # Shape: (1, 3*H*W, D)

    def _to_b_t_c_h_w(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)

        if frames.dim() != 5:
            raise ValueError("Expected input shape (B, T, C, H, W) or (B, T, H, W, C)")

        if frames.shape[2] == self.in_channels:
            b_t_c_h_w = frames
        elif frames.shape[-1] == self.in_channels:
            b_t_c_h_w = frames.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError("Could not infer channel dimension in input frames")

        if b_t_c_h_w.shape[1] != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} stacked frames, got {b_t_c_h_w.shape[1]}"
            )

        return b_t_c_h_w.float()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        frames = self._to_b_t_c_h_w(frames)
        batch_size, num_frames, channels, height, width = frames.shape

        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")

        flat_frames = frames.reshape(batch_size * num_frames, channels, height, width)
        if height != self.img_height or width != self.img_width:
            flat_frames = F.interpolate(
                flat_frames,
                size=(self.img_height, self.img_width),
                mode="bilinear",
                align_corners=False,
            )

        patch_tokens = self.patch_embed(flat_frames)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        patch_tokens = patch_tokens.reshape(
            batch_size, num_frames * self.patches_per_frame, self.embed_dim
        )

        pos = self._spatio_temporal_positional_encoding().to(
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        patch_tokens = patch_tokens + pos

        cls = (self.cls_token + self.cls_pos).expand(batch_size, -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)

        encoded = self.transformer(tokens)
        return self.norm(encoded[:, 0, :])


Encoder = SpatioTemporalViTEncoder


__all__ = ["SpatioTemporalViTEncoder", "Encoder"]
