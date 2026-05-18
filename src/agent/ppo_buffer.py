from collections.abc import Iterator

import numpy as np
import torch


class PPOBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_frames: int,
        channels: int,
        img_height: int,
        img_width: int,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.ptr = 0

        frame_shape = (rollout_steps, num_frames, channels, img_height, img_width)
        self.frames_x = np.zeros(frame_shape, dtype=np.uint8)
        self.frames_y = np.zeros(frame_shape, dtype=np.uint8)

        self.actions = np.zeros((rollout_steps,), dtype=np.int64)
        self.rewards = np.zeros((rollout_steps,), dtype=np.float32)
        self.dones = np.zeros((rollout_steps,), dtype=np.float32)

        self.values = np.zeros((rollout_steps,), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps,), dtype=np.float32)

        self.advantages = np.zeros((rollout_steps,), dtype=np.float32)
        self.returns = np.zeros((rollout_steps,), dtype=np.float32)

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        frames_x: np.ndarray,
        frames_y: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        if self.ptr >= self.rollout_steps:
            raise RuntimeError("PPOBuffer is full. Call reset() before adding new rollouts.")

        self.frames_x[self.ptr] = frames_x
        self.frames_y[self.ptr] = frames_y
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.ptr == 0:
            return

        advantage = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            self.advantages[t] = advantage

        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def iter_minibatches(
        self,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
    ) -> Iterator[dict[str, torch.Tensor]]:
        indices = np.arange(self.ptr)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.ptr, batch_size):
            mb_idx = indices[start : start + batch_size]
            yield {
                "frames_x": torch.from_numpy(self.frames_x[mb_idx]).to(device).float() / 255.0,
                "frames_y": torch.from_numpy(self.frames_y[mb_idx]).to(device).float() / 255.0,
                "actions": torch.from_numpy(self.actions[mb_idx]).to(device),
                "old_log_probs": torch.from_numpy(self.log_probs[mb_idx]).to(device),
                "returns": torch.from_numpy(self.returns[mb_idx]).to(device),
                "advantages": torch.from_numpy(self.advantages[mb_idx]).to(device),
            }


__all__ = ["PPOBuffer"]
