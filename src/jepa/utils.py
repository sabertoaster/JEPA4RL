
import torch


# Loss 

## Loss regularization

## Loss JEPA

## Loss actor


def jepa_loss(s_y_pred: torch.Tensor, s_y: torch.Tensor) -> torch.Tensor:
	"""Squared Euclidean distance between predicted and target embeddings."""
	return torch.sum((s_y_pred - s_y) ** 2, dim=-1).mean()


def variance_regularization(s_x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Negative clipped mean variance to discourage representation collapse."""
	var_per_dim = torch.var(s_x, dim=0, unbiased=False) + eps
	mean_var = var_per_dim.mean()
	one = torch.ones((), device=s_x.device, dtype=s_x.dtype)
	return -torch.minimum(one, mean_var)


__all__ = ["jepa_loss", "variance_regularization"]