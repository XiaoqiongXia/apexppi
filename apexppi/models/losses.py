"""Bidirectional contrastive objective used to train ApexPPI."""

import torch
import torch.nn.functional as F


def sample_negative_hosts(
    pathogen_idx: torch.Tensor,
    positives_by_pathogen: dict[int, set[int]],
    num_hosts: int,
    num_negatives: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    rows = []
    for pathogen in pathogen_idx.detach().cpu().tolist():
        known = positives_by_pathogen.get(int(pathogen), set())
        if len(known) >= num_hosts:
            raise ValueError(f"Pathogen index {pathogen} has no available negative hosts")
        row = []
        while len(row) < num_negatives:
            candidate = int(torch.randint(num_hosts, (1,), generator=generator).item())
            if candidate not in known:
                row.append(candidate)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=pathogen_idx.device)


def sample_negative_pathogens(
    host_idx: torch.Tensor,
    positives_by_host: dict[int, set[int]],
    pathogen_candidates: torch.Tensor,
    num_negatives: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    candidates = pathogen_candidates.detach().cpu().tolist()
    if not candidates:
        raise ValueError("pathogen_candidates must not be empty")
    rows = []
    for host in host_idx.detach().cpu().tolist():
        known = positives_by_host.get(int(host), set())
        if len(known) >= len(candidates):
            raise ValueError(f"Host index {host} has no available negative pathogens")
        row = []
        while len(row) < num_negatives:
            position = int(torch.randint(len(candidates), (1,), generator=generator).item())
            candidate = int(candidates[position])
            if candidate not in known:
                row.append(candidate)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=host_idx.device)


def infonce_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits / temperature, labels)
