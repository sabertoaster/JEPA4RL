"""Train JEPA + PPO on CartPole.

Run:
    python scripts/training.py

Customize (example):
    python -c "from scripts.training import train; from src.config import TrainConfig; train(TrainConfig(total_updates=5))"

Key config flags:
    - use_jepa_loss: enable JEPA prediction loss.
    - use_actor_critic_grad: allow actor/critic gradients to update the encoder.
    - use_reg_loss: enable variance regularization on encoder embeddings.
    - save_episodic_returns: append completed episode returns to CSV.
    - episodic_returns_path: relative paths are resolved under the run directory.

Per-run outputs:
    Outputs are written under artifacts/checkpoints/{run_group}/{run_name}. If run_name
    is not provided, it is auto-generated using the variant tag and timestamp. A config.yaml
    file is saved in the run directory at the start of training.

Recommended variants:
    1) J^, grad, R^: use_jepa_loss=False, use_actor_critic_grad=True, use_reg_loss=False
    2) J, grad, R^: use_jepa_loss=True, use_actor_critic_grad=True, use_reg_loss=False
    3) J, grad^, R^: use_jepa_loss=True, use_actor_critic_grad=False, use_reg_loss=False
    4) J, grad^, R: use_jepa_loss=True, use_actor_critic_grad=False, use_reg_loss=True

CLI examples (one per variant):
    python scripts/training.py --use_ac_grad --no-use-jepa-loss --no-use-reg-loss
    python scripts/training.py --use_jepa_loss_grad --use_ac_grad --no-use-reg-loss
    python scripts/training.py --use_jepa_loss_grad --no-use-ac-grad --no-use-reg-loss
    python scripts/training.py --use_jepa_loss_grad --no-use-ac-grad --use_reg_loss_grad
"""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.agent.actor_critic import ActorCritic
from src.agent.ppo_buffer import PPOBuffer
from src.config import TrainConfig, get_default_config
from src.env.cartpole import make_jepa_cartpole_env
from src.jepa.jepa_model import JEPAModel
from src.jepa.utils import jepa_loss, variance_regularization


def preprocess_observation(observation: np.ndarray) -> np.ndarray:
    frames = np.asarray(observation, dtype=np.uint8)
    if frames.ndim != 4:
        raise ValueError(
            f"Expected stacked frame observation with 4 dims, got shape={frames.shape}"
        )

    # Convert from (T, H, W, C) to (T, C, H, W) if needed.
    if frames.shape[-1] == 3:
        frames = np.transpose(frames, (0, 3, 1, 2))
    elif frames.shape[1] != 3:
        raise ValueError(
            f"Could not infer channels from observation shape={frames.shape}"
        )

    return np.ascontiguousarray(frames)


def to_model_input(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames).unsqueeze(0).to(device).float() / 255.0


def build_models(
    cfg: TrainConfig, device: torch.device
) -> tuple[JEPAModel, ActorCritic]:
    jepa_model = JEPAModel(
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        num_frames=cfg.num_frames,
        in_channels=cfg.channels,
        embed_dim=cfg.embed_dim,
        action_dim=cfg.action_dim,
        ema_momentum=cfg.ema_momentum,
        patch_size=cfg.patch_size,
        depth=cfg.transformer_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        predictor_hidden_dim=cfg.predictor_hidden_dim,
    ).to(device)

    actor_critic = ActorCritic(
        embed_dim=cfg.embed_dim,
        action_dim=cfg.action_dim,
    ).to(device)

    return jepa_model, actor_critic


class PolicyInferenceWrapper(nn.Module):
    def __init__(self, x_encoder: nn.Module, actor_critic: ActorCritic) -> None:
        super().__init__()
        self.x_encoder = x_encoder
        self.actor_critic = actor_critic

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_x = self.x_encoder(frames)
        logits = self.actor_critic.actor(s_x)
        value = self.actor_critic.get_value(s_x)
        return logits, value


def variant_tag(cfg: TrainConfig) -> str:
    return (
        f"J{int(cfg.use_jepa_loss)}"
        f"G{int(cfg.use_actor_critic_grad)}"
        f"R{int(cfg.use_reg_loss)}"
    )


def resolve_run_config(cfg: TrainConfig) -> TrainConfig:
    run_name = cfg.run_name
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{variant_tag(cfg)}_{timestamp}"

    run_group = cfg.run_group or None
    return replace(cfg, run_name=run_name, run_group=run_group)


def checkpoint_dir_path(cfg: TrainConfig) -> Path:
    checkpoint_dir = Path(cfg.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT_DIR / checkpoint_dir
    if cfg.run_group:
        checkpoint_dir = checkpoint_dir / cfg.run_group
    if cfg.run_name:
        checkpoint_dir = checkpoint_dir / cfg.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_config_yaml(cfg: TrainConfig) -> Path:
    config_path = checkpoint_dir_path(cfg) / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(cfg), handle, sort_keys=False)
    return config_path


def episodic_returns_path(cfg: TrainConfig) -> Path:
    returns_path = Path(cfg.episodic_returns_path)
    if not returns_path.is_absolute():
        returns_path = checkpoint_dir_path(cfg) / returns_path
    returns_path.parent.mkdir(parents=True, exist_ok=True)
    return returns_path


def checkpoint_path_for_update(cfg: TrainConfig, update: int) -> Path:
    return checkpoint_dir_path(cfg) / f"jepa_ppo_update_{update:04d}.pt"


def latest_checkpoint_path(cfg: TrainConfig) -> Path:
    return checkpoint_dir_path(cfg) / "jepa_ppo_latest.pt"


def save_checkpoint(
    cfg: TrainConfig,
    update: int,
    jepa_model: JEPAModel,
    actor_critic: ActorCritic,
) -> Path:
    payload = {
        "update": update,
        "config": asdict(cfg),
        "x_encoder_state": jepa_model.x_encoder.state_dict(),
        "predictor_state": jepa_model.predictor.state_dict(),
        "actor_critic_state": actor_critic.state_dict(),
    }

    checkpoint_path = checkpoint_path_for_update(cfg, update)
    torch.save(payload, checkpoint_path)
    torch.save(payload, latest_checkpoint_path(cfg))
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
) -> tuple[JEPAModel, ActorCritic, TrainConfig, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = TrainConfig(**checkpoint["config"])
    jepa_model, actor_critic = build_models(config, device)

    jepa_model.x_encoder.load_state_dict(checkpoint["x_encoder_state"])
    jepa_model.predictor.load_state_dict(checkpoint["predictor_state"])
    actor_critic.load_state_dict(checkpoint["actor_critic_state"])
    jepa_model.update_y_encoder()

    jepa_model.eval()
    actor_critic.eval()

    update = int(checkpoint.get("update", 0))
    return jepa_model, actor_critic, config, update


def export_policy_onnx(
    cfg: TrainConfig,
    jepa_model: JEPAModel,
    actor_critic: ActorCritic,
    device: torch.device,
    update: int,
) -> Path:
    export_dir = checkpoint_dir_path(cfg)
    onnx_path = export_dir / f"policy_inference_update_{update:04d}.onnx"
    wrapper = PolicyInferenceWrapper(jepa_model.x_encoder, actor_critic).to(device)
    wrapper.eval()

    dummy_frames = torch.zeros(
        1,
        cfg.num_frames,
        cfg.channels,
        cfg.img_height,
        cfg.img_width,
        device=device,
        dtype=torch.float32,
    )

    try:
        torch.onnx.export(
            wrapper,
            args=(dummy_frames,),
            f=str(onnx_path),
            input_names=["frames"],
            output_names=["logits", "value"],
            dynamic_axes={
                "frames": {0: "batch"},
                "logits": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=17,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX export failed because required package is missing. "
            "Install 'onnx' and retry."
        ) from exc

    return onnx_path


def train(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or get_default_config()
    cfg = resolve_run_config(cfg)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if cfg.save_config:
        save_config_yaml(cfg)

    env = make_jepa_cartpole_env(
        height=cfg.img_height,
        width=cfg.img_width,
        stack_size=cfg.num_frames,
    )

    jepa_model, actor_critic = build_models(cfg, device)

    buffer = PPOBuffer(
        rollout_steps=cfg.ppo_rollout_steps,
        num_frames=cfg.num_frames,
        channels=cfg.channels,
        img_height=cfg.img_height,
        img_width=cfg.img_width,
    )

    trainable_params = (
        list(jepa_model.x_encoder.parameters())
        + list(jepa_model.predictor.parameters())
        + list(actor_critic.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.learning_rate)

    observation, _ = env.reset()
    frames_x = preprocess_observation(observation)

    returns_file = None
    returns_writer = None
    episode_index = 0
    episode_return = 0.0
    episode_length = 0
    total_env_steps = 0

    if cfg.save_episodic_returns:
        returns_path = episodic_returns_path(cfg)
        write_header = not returns_path.exists() or returns_path.stat().st_size == 0
        returns_file = returns_path.open("a", newline="")
        returns_writer = csv.DictWriter(
            returns_file,
            fieldnames=["episode", "update", "env_step", "return", "length"],
        )
        if write_header:
            returns_writer.writeheader()
            returns_file.flush()

    try:
        for update in range(1, cfg.total_updates + 1):
            buffer.reset()
            last_done = False
            update_episode_returns = []

            for _ in range(cfg.ppo_rollout_steps):
                with torch.no_grad():
                    model_frames_x = to_model_input(frames_x, device)
                    s_x = jepa_model.x_encoder(model_frames_x)
                    action, log_prob, _, value = actor_critic.get_action_and_value(s_x)

                action_item = int(action.item())
                next_obs, reward, terminated, truncated, _ = env.step(action_item)
                done = terminated or truncated
                frames_y = preprocess_observation(next_obs)

                buffer.add(
                    frames_x=frames_x,
                    frames_y=frames_y,
                    action=action_item,
                    reward=float(reward),
                    done=done,
                    value=float(value.item()),
                    log_prob=float(log_prob.item()),
                )

                frames_x = frames_y
                last_done = done

                total_env_steps += 1
                episode_return += float(reward)
                episode_length += 1

                if done:
                    episode_index += 1
                    update_episode_returns.append(episode_return)
                    if returns_writer is not None:
                        returns_writer.writerow(
                            {
                                "episode": episode_index,
                                "update": update,
                                "env_step": total_env_steps,
                                "return": episode_return,
                                "length": episode_length,
                            }
                        )
                        returns_file.flush()

                    episode_return = 0.0
                    episode_length = 0

                    observation, _ = env.reset()
                    frames_x = preprocess_observation(observation)

            with torch.no_grad():
                bootstrap_frames = to_model_input(frames_x, device)
                bootstrap_s_x = jepa_model.x_encoder(bootstrap_frames)
                last_value = float(actor_critic.get_value(bootstrap_s_x).item())

            buffer.compute_returns_and_advantages(
                last_value=last_value,
                last_done=last_done,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            metrics = {
                "total": 0.0,
                "jepa": 0.0,
                "reg": 0.0,
                "actor": 0.0,
                "critic": 0.0,
            }
            metric_steps = 0

            for _ in range(cfg.ppo_epochs):
                for batch in buffer.iter_minibatches(
                    batch_size=cfg.batch_size,
                    device=device,
                    shuffle=True,
                ):
                    advantages = batch["advantages"]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std(unbiased=False) + 1e-8
                    )

                    if cfg.use_jepa_loss:
                        s_x, s_y, s_y_pred = jepa_model(
                            batch["frames_x"],
                            batch["frames_y"],
                            batch["actions"],
                        )
                    else:
                        s_x = jepa_model.x_encoder(batch["frames_x"])
                        s_y = None
                        s_y_pred = None

                    s_x_actor = s_x if cfg.use_actor_critic_grad else s_x.detach()
                    _, new_log_probs, entropy, new_values = (
                        actor_critic.get_action_and_value(
                            s_x_actor,
                            batch["actions"],
                        )
                    )

                    ratio = (new_log_probs - batch["old_log_probs"]).exp()
                    unclipped_policy_loss = -advantages * ratio
                    clipped_policy_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - cfg.clip_coef,
                        1.0 + cfg.clip_coef,
                    )
                    actor_loss = torch.max(
                        unclipped_policy_loss, clipped_policy_loss
                    ).mean()

                    critic_loss = F.mse_loss(new_values, batch["returns"])
                    loss_jepa = (
                        jepa_loss(s_y_pred, s_y)
                        if cfg.use_jepa_loss
                        else s_x.new_zeros(())
                    )
                    loss_reg = (
                        variance_regularization(s_x)
                        if cfg.use_reg_loss
                        else s_x.new_zeros(())
                    )

                    total_loss = (
                        loss_jepa
                        + loss_reg
                        + actor_loss
                        + cfg.value_coef * critic_loss
                        - cfg.entropy_coef * entropy.mean()
                    )

                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    optimizer.step()

                    if cfg.use_jepa_loss:
                        # EMA update for y-encoder after each optimizer step.
                        jepa_model.update_y_encoder()

                    metrics["total"] += float(total_loss.item())
                    metrics["jepa"] += float(loss_jepa.item())
                    metrics["reg"] += float(loss_reg.item())
                    metrics["actor"] += float(actor_loss.item())
                    metrics["critic"] += float(critic_loss.item())
                    metric_steps += 1

            mean_total = metrics["total"] / max(metric_steps, 1)
            mean_jepa = metrics["jepa"] / max(metric_steps, 1)
            mean_reg = metrics["reg"] / max(metric_steps, 1)
            mean_actor = metrics["actor"] / max(metric_steps, 1)
            mean_critic = metrics["critic"] / max(metric_steps, 1)

            episode_count = len(update_episode_returns)
            if episode_count:
                ep_return_mean = sum(update_episode_returns) / episode_count
                ep_return_min = min(update_episode_returns)
                ep_return_max = max(update_episode_returns)
                episode_summary = (
                    f"episodes={episode_count} "
                    f"ep_return_mean={ep_return_mean:.2f} "
                    f"ep_return_min={ep_return_min:.2f} "
                    f"ep_return_max={ep_return_max:.2f}"
                )
            else:
                episode_summary = "episodes=0"

            print(
                f"update={update:04d}/{cfg.total_updates:04d} "
                f"total={mean_total:.4f} jepa={mean_jepa:.4f} reg={mean_reg:.4f} "
                f"actor={mean_actor:.4f} critic={mean_critic:.4f} {episode_summary}",
                flush=True,
            )

            should_save = cfg.save_every_n_updates > 0 and (
                update % cfg.save_every_n_updates == 0
            )
            if should_save:
                checkpoint_path = save_checkpoint(cfg, update, jepa_model, actor_critic)
                print(f"saved checkpoint: {checkpoint_path}", flush=True)

                if cfg.export_onnx:
                    onnx_path = export_policy_onnx(
                        cfg,
                        jepa_model,
                        actor_critic,
                        device,
                        update,
                    )
                    print(f"exported onnx: {onnx_path}", flush=True)

        if cfg.save_last_checkpoint:
            checkpoint_path = save_checkpoint(
                cfg,
                cfg.total_updates,
                jepa_model,
                actor_critic,
            )
            print(f"saved final checkpoint: {checkpoint_path}", flush=True)

            if cfg.export_onnx:
                onnx_path = export_policy_onnx(
                    cfg,
                    jepa_model,
                    actor_critic,
                    device,
                    cfg.total_updates,
                )
                print(f"exported final onnx: {onnx_path}", flush=True)
    finally:
        if returns_file is not None:
            returns_file.close()

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JEPA + PPO on CartPole.")
    parser.add_argument("--run-name", "--run_name", dest="run_name", default=None)
    parser.add_argument("--run-group", "--run_group", dest="run_group", default=None)
    parser.add_argument(
        "--total-updates", "--total_updates", dest="total_updates", type=int, default=50
    )
    parser.add_argument(
        "--use-jepa-loss",
        "--use_jepa_loss",
        "--use_jepa_loss_grad",
        "--use-jepa-loss-grad",
        dest="use_jepa_loss",
        action="store_true",
    )
    parser.add_argument(
        "--no-use-jepa-loss",
        "--no_use_jepa_loss",
        dest="use_jepa_loss",
        action="store_false",
    )
    parser.set_defaults(use_jepa_loss=None)

    parser.add_argument(
        "--use-ac-grad",
        "--use_ac_grad",
        "--use-actor-critic-grad",
        "--use_actor_critic_grad",
        dest="use_actor_critic_grad",
        action="store_true",
    )
    parser.add_argument(
        "--no-use-ac-grad",
        "--no_use_ac_grad",
        "--no-use-actor-critic-grad",
        dest="use_actor_critic_grad",
        action="store_false",
    )
    parser.set_defaults(use_actor_critic_grad=None)

    parser.add_argument(
        "--use-reg-loss",
        "--use_reg_loss",
        "--use_reg_loss_grad",
        "--use-reg-loss-grad",
        dest="use_reg_loss",
        action="store_true",
    )
    parser.add_argument(
        "--no-use-reg-loss",
        "--no_use_reg_loss",
        dest="use_reg_loss",
        action="store_false",
    )
    parser.set_defaults(use_reg_loss=None)

    parser.add_argument("--save-config", dest="save_config", action="store_true")
    parser.add_argument("--no-save-config", dest="save_config", action="store_false")
    parser.set_defaults(save_config=None)

    return parser.parse_args()


def apply_cli_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    updates: dict[str, object] = {}
    if args.use_jepa_loss is not None:
        updates["use_jepa_loss"] = args.use_jepa_loss
    if args.use_actor_critic_grad is not None:
        updates["use_actor_critic_grad"] = args.use_actor_critic_grad
    if args.use_reg_loss is not None:
        updates["use_reg_loss"] = args.use_reg_loss
    if args.run_name is not None:
        updates["run_name"] = args.run_name
    if args.run_group is not None:
        updates["run_group"] = args.run_group
    if args.save_config is not None:
        updates["save_config"] = args.save_config
    if args.total_updates is not None:
        updates["total_updates"] = args.total_updates
    return replace(cfg, **updates) if updates else cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = apply_cli_overrides(get_default_config(), args)
    train(cfg)
