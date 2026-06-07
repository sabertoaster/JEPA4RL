from argparse import ArgumentParser
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.training import load_checkpoint, preprocess_observation, to_model_input
from src.env.cartpole import make_jepa_cartpole_env


def run_demo(
    checkpoint: Path,
    device_name: str | None = None,
    sample_actions: bool = False,
    delay_ms: int = 40,
) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    jepa_model, actor_critic, cfg, update = load_checkpoint(checkpoint, device)
    env = make_jepa_cartpole_env(
        height=400,
        width=400,
        stack_size=cfg.num_frames,
    )

    window_name = "JEPA PPO Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for _ in range(0, 15):
        try:
            observation, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            steps = 0

            while not (done or truncated):
                frames = preprocess_observation(observation)
                model_input = to_model_input(frames, device)

                with torch.no_grad():
                    s_x = jepa_model.x_encoder(model_input)
                    dist = actor_critic.get_dist(s_x)
                    value = actor_critic.get_value(s_x)
                    if sample_actions:
                        action = dist.sample()
                    else:
                        action = torch.argmax(dist.logits, dim=-1)

                action_item = int(action.item())
                observation, reward, done, truncated, _ = env.step(action_item)
                episode_reward += float(reward)
                steps += 1

                latest_frame = observation[-1]
                bgr = cv2.cvtColor(np.asarray(latest_frame), cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"step={steps} action={action_item} value={float(value.item()):.3f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, bgr)

                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

            print(
                f"demo checkpoint={checkpoint} update={update} steps={steps} reward={episode_reward:.2f}",
                flush=True,
            )
        finally:
            env.close()
            cv2.destroyAllWindows()


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run JEPA+PPO demo from a saved checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/jepa_ppo_latest.pt"),
        help="Path to training checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cpu or cuda). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--sample-actions",
        action="store_true",
        help="Sample actions from policy distribution instead of argmax.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=40,
        help="Delay between frames in milliseconds.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    run_demo(
        checkpoint=args.checkpoint,
        device_name=args.device,
        sample_actions=args.sample_actions,
        delay_ms=args.delay_ms,
    )
