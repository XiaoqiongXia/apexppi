"""Binary link-prediction metrics."""

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_binary_metrics(
    labels: torch.Tensor, logits: torch.Tensor
) -> dict[str, float]:
    labels_np = labels.detach().cpu().numpy()
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return {
        "average_precision": float(average_precision_score(labels_np, probabilities)),
        "roc_auc": float(roc_auc_score(labels_np, probabilities)),
    }
