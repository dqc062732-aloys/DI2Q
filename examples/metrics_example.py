import torch

from src.metrics import MR, brier_minFDE, minADE, minFDE


def main():
    target = torch.zeros(2, 3, 2)
    trajectories = torch.zeros(2, 6, 3, 2)
    trajectories[..., 0] = 4.0
    trajectories[0, 0, :, 0] = 1.0
    trajectories[0, 1, :, 0] = 0.0
    trajectories[1, 0, :, 0] = 3.0
    trajectories[1, 1, :, 0] = 2.5
    scores = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]]).repeat(2, 1)
    outputs = {"y_hat": trajectories, "pi": scores}
    metrics = {
        "minADE1": minADE(k=1),
        "minFDE1": minFDE(k=1),
        "minADE6": minADE(k=6),
        "minFDE6": minFDE(k=6),
        "MR": MR(),
        "b-minFDE6": brier_minFDE(k=6),
    }
    for name, metric in metrics.items():
        metric.update(outputs, target)
        print(f"{name}: {metric.compute().item():.6f}")


if __name__ == "__main__":
    main()
