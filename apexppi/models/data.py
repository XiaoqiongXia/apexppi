"""Graph and supervision data helpers for ApexPPI."""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData


def load_graph(path: Path) -> tuple[HeteroData, dict[str, dict[str, int]]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["data"], obj["node_maps"]


def build_supervision_tensors(
    edges: pd.DataFrame, node_maps: dict[str, dict[str, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    host_map = node_maps["host_protein"]
    pathogen_map = node_maps["pathogen_protein"]
    host_idx = torch.tensor(
        [host_map[f"host_protein:{protein_id}"] for protein_id in edges["host_uniprot"]],
        dtype=torch.long,
    )
    pathogen_idx = torch.tensor(
        [
            pathogen_map[f"pathogen_protein:{protein_id}"]
            for protein_id in edges["pathogen_uniprot"]
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(edges["label"].astype(float).tolist(), dtype=torch.float32)
    return host_idx, pathogen_idx, labels


def build_positives_by_pathogen(
    edges: pd.DataFrame, node_maps: dict[str, dict[str, int]]
) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = {}
    for row in edges[edges["label"].astype(float) > 0].itertuples(index=False):
        host_idx = node_maps["host_protein"][f"host_protein:{row.host_uniprot}"]
        pathogen_idx = node_maps["pathogen_protein"][
            f"pathogen_protein:{row.pathogen_uniprot}"
        ]
        positives.setdefault(pathogen_idx, set()).add(host_idx)
    return positives


def build_positives_by_host(
    edges: pd.DataFrame, node_maps: dict[str, dict[str, int]]
) -> dict[int, set[int]]:
    positives: dict[int, set[int]] = {}
    for row in edges[edges["label"].astype(float) > 0].itertuples(index=False):
        host_idx = node_maps["host_protein"][f"host_protein:{row.host_uniprot}"]
        pathogen_idx = node_maps["pathogen_protein"][
            f"pathogen_protein:{row.pathogen_uniprot}"
        ]
        positives.setdefault(host_idx, set()).add(pathogen_idx)
    return positives
