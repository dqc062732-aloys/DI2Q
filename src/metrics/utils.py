import torch


def sort_predictions(predictions, probability, k=6):
    indices = torch.argsort(probability, dim=-1, descending=True)
    sorted_prob = probability[torch.arange(probability.size(0))[:, None], indices]
    sorted_predictions = predictions[
        torch.arange(predictions.size(0))[:, None], indices
    ]
    return (sorted_predictions[:, :k], sorted_prob[:, :k])
