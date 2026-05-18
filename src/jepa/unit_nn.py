import torch 
from torch import nn


class MLP(nn.Module):
	def __init__(
		self,
		in_dim: int,
		hidden_dim: int,
		out_dim: int,
		num_layers: int = 2,
		activation: nn.Module = nn.ReLU,
		dropout: float = 0.0,
	) -> None:
		super().__init__()

		if num_layers < 2:
			raise ValueError("num_layers must be >= 2")

		layers = []
		last_dim = in_dim
		for _ in range(num_layers - 1):
			layers.append(nn.Linear(last_dim, hidden_dim))
			layers.append(activation())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			last_dim = hidden_dim
		layers.append(nn.Linear(last_dim, out_dim))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


## (baseline) MLP with configurable activation + layer depth


__all__ = ["MLP"]

