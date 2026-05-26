from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def _load_returns(
    csv_path: str | Path, max_env_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    returns: list[float] = []
    with Path(csv_path).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(float(row["env_step"]))
            if step > max_env_steps:
                break
            steps.append(step)
            returns.append(float(row["return"]))
    if not steps:
        return np.array([]), np.array([])
    return np.asarray(steps, dtype=np.float32), np.asarray(returns, dtype=np.float32)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_average_episodic_returns(
    runs: Dict[str, List[str]],
    output_path: str | Path,
    max_env_steps: int = 100_000,
    grid_points: int = 1000,
    smooth_window: int = 25,
    legend_labels: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Plot episodic returns for multiple configs.

    Args:
        runs: Dict of config name -> list of episodic_returns.csv paths.
        output_path: Where to save the plot (png, pdf, etc).
        max_env_steps: Only include data up to this step count.
        grid_points: Common x-grid for averaging across runs.
        smooth_window: Moving average window for the mean curve.
        legend_labels: Optional mapping from config name to legend label.
        colors: Optional mapping from config name to color hex.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    grid = np.linspace(1, max_env_steps, grid_points, dtype=np.float32)

    for config_name, csv_paths in runs.items():
        color = colors.get(config_name) if colors else None
        label = legend_labels.get(config_name) if legend_labels else config_name

        interp_runs = []
        for csv_path in csv_paths:
            steps, returns = _load_returns(csv_path, max_env_steps)
            if steps.size == 0:
                continue

            # Interpolate to common grid for averaging.
            interp = np.interp(grid, steps, returns, left=np.nan, right=np.nan)
            interp_runs.append(interp)

        if not interp_runs:
            continue

        stacked = np.vstack(interp_runs)
        mean_returns = np.nanmean(stacked, axis=0)
        mean_steps = grid
        # Dim background line (per run).
        ax.plot(mean_steps, mean_returns, color=color, alpha=0.15, linewidth=1.0)

        if smooth_window > 1:
            mean_returns = _moving_average(mean_returns, smooth_window)
            mean_steps = mean_steps[smooth_window - 1 :]

        ax.plot(mean_steps, mean_returns, color=color, linewidth=1.5, label=label)

    ax.set_title("Average Episodic Return")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Return")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_env_steps)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # runs = {
    #     "J0G1R0": [
    #         "artifacts/checkpoints/J0G1R0_20260518_212050/episodic_returns.csv",
    #         # add 4 more runs...
    #     ],
    #     "J1G1R0": [...],
    #     "J1G0R0": [...],
    #     "J1G0R1": [...],
    # }
    runs = {
        "J1G1R0": [
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp1\\J1G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp2\\J1G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp3\\J1G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp4\\J1G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp5\\J1G1R0\\episodic_returns.csv",
        ],
        "J0G1R0": [
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp1\\J0G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp2\\J0G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp3\\J0G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp4\\J0G1R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp5\\J0G1R0\\episodic_returns.csv",
        ],
        "J1G0R1": [
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp1\\J1G0R1\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp2\\J1G0R1\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp3\\J1G0R1\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp4\\J1G0R1\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp5\\J1G0R1\\episodic_returns.csv",
        ],
        "J1G0R0": [
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp1\\J1G0R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp2\\J1G0R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp3\\J1G0R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp4\\J1G0R0\\episodic_returns.csv",
            "D:\\Projects\\Jepa4RL\\artifacts\\checkpoints\\exp5\\J1G0R0\\episodic_returns.csv",
        ],
    }

    legend_labels = {
        "J1G1R0": r"$J, \nabla, \hat{R}$",
        "J0G1R0": r"$\hat{J}, \nabla, \hat{R}$",
        "J1G0R1": r"$J, \hat{\nabla}, R$",
        "J1G0R0": r"$J, \hat{\nabla}, \hat{R}$",
    }

    colors = {
        "J1G1R0": "#f28e2b",  # orange
        "J0G1R0": "#4e79a7",  # blue
        "J1G0R1": "#9e9e9e",  # gray
        "J1G0R0": "#e15759",  # red
    }

    plot_average_episodic_returns(
        runs,
        output_path="artifacts/fig3_average_return.png",
        max_env_steps=100_000,
        legend_labels=legend_labels,
        colors=colors,
    )
