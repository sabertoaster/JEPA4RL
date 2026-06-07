from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    img_height: int = 60
    img_width: int = 60
    num_frames: int = 3
    channels: int = 3

    embed_dim: int = 64
    action_dim: int = 2
    ema_momentum: float = 0.99

    batch_size: int = 64
    ppo_rollout_steps: int = 2000
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 64

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    use_jepa_loss: bool = True
    use_actor_critic_grad: bool = False
    use_reg_loss: bool = True

    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    total_updates: int = 10

    patch_size: int = 20
    transformer_depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    predictor_hidden_dim: int = 128

    device: str = "cuda"

    checkpoint_dir: str = "artifacts/checkpoints"
    run_name: str | None = None
    run_group: str | None = None
    save_config: bool = True
    save_every_n_updates: int = 1
    save_last_checkpoint: bool = True
    export_onnx: bool = True

    save_episodic_returns: bool = True
    episodic_returns_path: str = "episodic_returns.csv"


def get_default_config() -> TrainConfig:
    return TrainConfig()
