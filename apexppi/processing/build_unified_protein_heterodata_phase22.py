#!/usr/bin/env python3
"""Build a unified protein-node PyG HeteroData graph for HPIDB Phase22."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData


def load_embedding_map(embedding_path: Path) -> tuple[dict[str, torch.Tensor], int]:
    obj = torch.load(embedding_path, map_location="cpu")
    embeddings = obj["embeddings"].float()
    protein_ids = obj["protein_ids"]
    return {protein_id: embeddings[i] for i, protein_id in enumerate(protein_ids)}, embeddings.shape[1]


def make_bidirectional_edges(edges: list[tuple[int, int]]) -> torch.Tensor:
    directed = set()
    for source, target in edges:
        if source == target:
            continue
        directed.add((source, target))
        directed.add((target, source))
    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    rows = sorted(directed)
    return torch.tensor(rows, dtype=torch.long).t().contiguous()


def _strip_node_id(value: str) -> str:
    return str(value).split(":", 1)[1] if ":" in str(value) else str(value)


def _load_internal_edges(path: Path, protein_to_idx: dict[str, int]) -> list[tuple[int, int]]:
    frame = pd.read_csv(path, sep="\t")
    edges = []
    for row in frame.itertuples(index=False):
        source = _strip_node_id(getattr(row, "source_id"))
        target = _strip_node_id(getattr(row, "target_id"))
        if source in protein_to_idx and target in protein_to_idx:
            edges.append((protein_to_idx[source], protein_to_idx[target]))
    return edges


def build_unified_protein_heterodata(
    data_dir: Path,
    embedding_path: Path,
    include_hp_ppi: bool = True,
    host_ppi_edge_path: Path | None = None,
    host_similarity_edge_path: Path | None = None,
    host_lincs_drugbank_edge_path: Path | None = None,
    host_lincs_knockdown_edge_path: Path | None = None,
    host_lincs_overexpression_edge_path: Path | None = None,
    pathogen_similarity_edge_path: Path | None = None,
) -> tuple[HeteroData, dict[str, dict[str, int]], pd.DataFrame]:
    embedding_by_id, _ = load_embedding_map(embedding_path)
    host_nodes = pd.read_csv(data_dir / "host_nodes.tsv", sep="\t")
    pathogen_nodes = pd.read_csv(data_dir / "pathogen_nodes.tsv", sep="\t")
    host_ids = host_nodes["host_uniprot"].astype(str).tolist()
    pathogen_ids = pathogen_nodes["pathogen_uniprot"].astype(str).tolist()

    host_id_set = set(host_ids)
    protein_ids = host_ids + [protein_id for protein_id in pathogen_ids if protein_id not in host_id_set]
    protein_to_idx = {protein_id: idx for idx, protein_id in enumerate(protein_ids)}
    host_to_idx = {protein_id: protein_to_idx[protein_id] for protein_id in host_ids}
    pathogen_to_idx = {protein_id: protein_to_idx[protein_id] for protein_id in pathogen_ids}

    missing_embeddings = [protein_id for protein_id in protein_ids if protein_id not in embedding_by_id]
    if missing_embeddings:
        raise KeyError(f"Missing embeddings for proteins: {missing_embeddings[:5]}")

    data = HeteroData()
    data["protein"].x = torch.stack([embedding_by_id[protein_id] for protein_id in protein_ids])

    host_set = set(host_ids)
    pathogen_set = set(pathogen_ids)
    node_rows = []
    for idx, protein_id in enumerate(protein_ids):
        is_host = protein_id in host_set
        is_pathogen = protein_id in pathogen_set
        role = "both" if is_host and is_pathogen else "host" if is_host else "pathogen"
        node_rows.append(
            {
                "node_id": f"protein:{protein_id}",
                "protein_uniprot": protein_id,
                "protein_idx": idx,
                "role": role,
            }
        )
    protein_nodes = pd.DataFrame(node_rows)

    if include_hp_ppi:
        train_edges = pd.read_csv(data_dir / "message_passing_train_edges.tsv", sep="\t")
        hp_edges = [
            (host_to_idx[row.host_uniprot], pathogen_to_idx[row.pathogen_uniprot])
            for row in train_edges.itertuples(index=False)
            if row.host_uniprot in host_to_idx and row.pathogen_uniprot in pathogen_to_idx
        ]
        data["protein", "hp_ppi_observed", "protein"].edge_index = make_bidirectional_edges(
            hp_edges
        )

    if host_ppi_edge_path is not None:
        data["protein", "host_host_string_ppi", "protein"].edge_index = make_bidirectional_edges(
            _load_internal_edges(host_ppi_edge_path, protein_to_idx)
        )
    if host_similarity_edge_path is not None:
        data["protein", "host_host_esm2_sim", "protein"].edge_index = make_bidirectional_edges(
            _load_internal_edges(host_similarity_edge_path, protein_to_idx)
        )
    if host_lincs_drugbank_edge_path is not None:
        data["protein", "host_host_lincs_drugbank_sim", "protein"].edge_index = (
            make_bidirectional_edges(
                _load_internal_edges(host_lincs_drugbank_edge_path, protein_to_idx)
            )
        )
    if host_lincs_knockdown_edge_path is not None:
        data["protein", "host_host_lincs_knockdown_sim", "protein"].edge_index = (
            make_bidirectional_edges(
                _load_internal_edges(host_lincs_knockdown_edge_path, protein_to_idx)
            )
        )
    if host_lincs_overexpression_edge_path is not None:
        data["protein", "host_host_lincs_overexpression_sim", "protein"].edge_index = (
            make_bidirectional_edges(
                _load_internal_edges(host_lincs_overexpression_edge_path, protein_to_idx)
            )
        )
    if pathogen_similarity_edge_path is not None:
        data["protein", "pathogen_pathogen_sim", "protein"].edge_index = make_bidirectional_edges(
            _load_internal_edges(pathogen_similarity_edge_path, protein_to_idx)
        )

    node_maps = {
        "protein": {f"protein:{protein_id}": idx for protein_id, idx in protein_to_idx.items()},
        "host_protein": {f"host_protein:{protein_id}": idx for protein_id, idx in host_to_idx.items()},
        "pathogen_protein": {
            f"pathogen_protein:{protein_id}": idx for protein_id, idx in pathogen_to_idx.items()
        },
    }
    return data, node_maps, protein_nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("data/processed/embeddings/esm2_t12_35M_UR50D_protein_mean_embeddings.pt"),
    )
    parser.add_argument(
        "--exclude-hp-ppi",
        action="store_true",
        help="Do not include training-positive HP-PPI edges in the message-passing graph.",
    )
    parser.add_argument("--host-ppi-edge-path", type=Path)
    parser.add_argument("--host-similarity-edge-path", type=Path)
    parser.add_argument("--host-lincs-drugbank-edge-path", type=Path)
    parser.add_argument("--host-lincs-knockdown-edge-path", type=Path)
    parser.add_argument("--host-lincs-overexpression-edge-path", type=Path)
    parser.add_argument("--pathogen-similarity-edge-path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi_unified_protein_graph"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, node_maps, protein_nodes = build_unified_protein_heterodata(
        data_dir=args.data_dir,
        embedding_path=args.embedding_path,
        include_hp_ppi=not args.exclude_hp_ppi,
        host_ppi_edge_path=args.host_ppi_edge_path,
        host_similarity_edge_path=args.host_similarity_edge_path,
        host_lincs_drugbank_edge_path=args.host_lincs_drugbank_edge_path,
        host_lincs_knockdown_edge_path=args.host_lincs_knockdown_edge_path,
        host_lincs_overexpression_edge_path=args.host_lincs_overexpression_edge_path,
        pathogen_similarity_edge_path=args.pathogen_similarity_edge_path,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heterodata_path = args.output_dir / "heterodata_unified_protein.pt"
    torch.save({"data": data, "node_maps": node_maps}, heterodata_path)
    protein_nodes.to_csv(args.output_dir / "nodes_protein.tsv", sep="\t", index=False)
    summary = {
        "heterodata_path": str(heterodata_path),
        "node_types": list(data.node_types),
        "edge_types": [str(edge_type) for edge_type in data.edge_types],
        "num_nodes": {node_type: int(data[node_type].num_nodes) for node_type in data.node_types},
        "num_edges": {
            str(edge_type): int(data[edge_type].edge_index.shape[1])
            for edge_type in data.edge_types
        },
    }
    (args.output_dir / "unified_protein_graph_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
