# Jepa4RL

JEPA + PPO on pixel observations for CartPole. This repo trains a spatio‑temporal ViT encoder with a JEPA predictor while a PPO actor‑critic learns from the encoder embeddings.

## What this repo does
- Uses pixel observations from CartPole with a 3‑frame stack.
- Trains a JEPA model (ViT x‑encoder + EMA y‑encoder + predictor).
- Trains a PPO actor‑critic on top of the encoder embeddings.
- Supports four training variants controlled by three flags (J, G, R).

## Training variants (J / G / R)
- **J**: JEPA loss (`use_jepa_loss`)
- **G**: actor‑critic gradients flow into the encoder (`use_actor_critic_grad`)
- **R**: variance regularization on embeddings (`use_reg_loss`)

Recommended configs:
1) `J^, G, R^`: `use_jepa_loss=False, use_actor_critic_grad=True, use_reg_loss=False`
2) `J, G, R^`: `use_jepa_loss=True, use_actor_critic_grad=True, use_reg_loss=False`
3) `J, G^, R^`: `use_jepa_loss=True, use_actor_critic_grad=False, use_reg_loss=False`
4) `J, G^, R`: `use_jepa_loss=True, use_actor_critic_grad=False, use_reg_loss=True`

## Project layout
- [scripts/training.py](scripts/training.py): main training loop, checkpointing, ONNX export, episodic return logging.
- [src/config.py](src/config.py): training config defaults.
- [src/env/cartpole.py](src/env/cartpole.py): pixel CartPole wrapper with frame stacking.
- [src/jepa/jepa_model.py](src/jepa/jepa_model.py): JEPA model with EMA target encoder.
- [src/jepa/encoder.py](src/jepa/encoder.py): spatio‑temporal ViT encoder.
- [src/agent/actor_critic.py](src/agent/actor_critic.py): PPO actor‑critic.
- [src/agent/ppo_buffer.py](src/agent/ppo_buffer.py): rollout buffer + GAE.
- [scripts/demo_in_action.py](scripts/demo_in_action.py): play a trained checkpoint.
- [src/visualize/plots.py](src/visualize/plots.py): Fig. 3 plotting helper.

## Setup
```bash
pip install -r requirements.txt
```

Plotting requires `matplotlib` (not listed in requirements):
```bash
pip install matplotlib
```

## Run training
Default run:
```bash
python scripts/training.py
```

Four variant commands:
```bash
python scripts/training.py --use_ac_grad --no-use-jepa-loss --no-use-reg-loss --run-group exp1
python scripts/training.py --use_jepa_loss_grad --use_ac_grad --no-use-reg-loss --run-group exp1
python scripts/training.py --use_jepa_loss_grad --no-use-ac-grad --no-use-reg-loss --run-group exp1
python scripts/training.py --use_jepa_loss_grad --no-use-ac-grad --use_reg_loss_grad --run-group exp1
```

### 100k environment steps
Total env steps is approximately:
$ \text{total\_updates} \times \text{ppo\_rollout\_steps} $

The default `ppo_rollout_steps` is 2000 in config.py, so 100k steps is `total_updates=50`.

Run:
```bash
python scripts/training.py --flags --total_updates 50
```

## Outputs
Each run writes to a unique run directory under checkpoints with:
- Checkpoints (`jepa_ppo_update_*.pt` and `jepa_ppo_latest.pt`)
- ONNX policy snapshots
- A YAML config snapshot (example: config.yaml)
- Episodic returns CSV (example: episodic_returns.csv)

## Fig. 3 plotting (average of 5 runs)
Use plots.py to plot mean curves across multiple runs:

```python
from src.visualize.plots import plot_average_episodic_returns

runs = {
    "J0G1R0": [
        "artifacts/checkpoints/J0G1R0_run1/episodic_returns.csv",
        "artifacts/checkpoints/J0G1R0_run2/episodic_returns.csv",
        "artifacts/checkpoints/J0G1R0_run3/episodic_returns.csv",
        "artifacts/checkpoints/J0G1R0_run4/episodic_returns.csv",
        "artifacts/checkpoints/J0G1R0_run5/episodic_returns.csv",
    ],
    "J1G1R0": [...],
    "J1G0R0": [...],
    "J1G0R1": [...],
}

plot_average_episodic_returns(
    runs,
    output_path="artifacts/fig3_average_return.png",
    max_env_steps=100_000,
)
```

## Demo (optional - WIP)
Play a trained policy:
```bash
python scripts/demo_in_action.py --checkpoint artifacts/checkpoints/<run_name>/jepa_ppo_latest.pt
```