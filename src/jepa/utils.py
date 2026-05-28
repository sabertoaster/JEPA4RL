import torch
import torch.nn.functional as F

# Loss

## Loss regularization

## Loss JEPA

## Loss actor


def jepa_loss(s_y_pred: torch.Tensor, s_y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between predicted and target embeddings."""
    # return torch.sum((s_y_pred - s_y) ** 2, dim=-1).mean()
    return F.mse_loss(s_y_pred, s_y)


# def variance_regularization(s_x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
# 	"""Negative clipped mean variance to discourage representation collapse."""
# 	var_per_dim = torch.var(s_x, dim=0, unbiased=False) + eps
# 	mean_var = var_per_dim.mean()
# 	one = torch.ones((), device=s_x.device, dtype=s_x.dtype)
# 	return -torch.minimum(one, mean_var)


def variance_regularization(s_x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Hinge loss on standard deviation (VICReg style).
    This guarantees a constant-magnitude gradient that prevents
    the vanishing gradient trap of raw variance.
    """
    # Calculate standard deviation instead of raw variance
    std_per_dim = torch.sqrt(torch.var(s_x, dim=0, unbiased=False) + eps)

    # Penalize the standard deviation if it drops below 1.0.
    # This returns a positive loss that we want to minimize to 0.0.
    return torch.mean(torch.relu(1.0 - std_per_dim))


__all__ = ["jepa_loss", "variance_regularization"]
