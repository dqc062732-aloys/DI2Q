# DI2Q

Motion Forecasting, Trajectory Prediction.

## Code Release

The full source code will be released upon acceptance of the paper.

The current partial release contains general evaluation and learning-rate
utilities used by the project. It can be used independently of the forecasting
model. The model architecture, HCM and BMF modules, training pipeline, and
pretrained checkpoints are not included in this release.

## Available Utilities

| Path | Contents |
| --- | --- |
| `src/metrics/` | Displacement errors, miss rate, probability-adjusted endpoint error, and joint multi-agent metrics |
| `src/utils/optim.py` | Linear warmup followed by cosine learning-rate decay |
| `examples/metrics_example.py` | A small CPU example with synthetic trajectories |

The single-agent metrics accept a dictionary with `y_hat` of shape
`[batch, modes, future_steps, 2]` and `pi` of shape `[batch, modes]`, together
with a target tensor of shape `[batch, future_steps, 2]`.

- `minADE(k)` and `minFDE(k)` select the top `k` modes by score, then minimize
  average displacement error and endpoint error separately.
- `MR` uses all supplied modes and a default endpoint threshold of 2 meters.
- `brier_minFDE(k)` adds `(1 - probability)^2` to each selected mode's endpoint
  error, then takes the minimum. Its default input scores are logits.
- `AvgMinADE`, `AvgMinFDE`, and `ActorMR` accept joint predictions of shape
  `[batch, modes, agents, future_steps, 2]` and a scored-agent mask. Each scene
  must contain at least one scored agent.

These implementations expose the project's utility interfaces. Benchmark
comparisons should also specify mode selection, probability normalization,
actor selection, and aggregation conventions.

## Quick Start

The utilities have been checked with Python 3.10 and the versions in
`requirements.txt`. A GPU and dataset files are not required for the example.

```bash
python -m pip install -r requirements.txt
python -m examples.metrics_example
```

Minimal usage:

```python
from src.metrics import minFDE

metric = minFDE(k=6)
metric.update({"y_hat": trajectories, "pi": scores}, target)
value = metric.compute()
```

The model training and inference commands will be documented with the full
code release.
