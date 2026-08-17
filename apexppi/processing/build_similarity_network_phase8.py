#!/usr/bin/env python3
"""Build ESM2 cosine kNN similarity networks for Phase 8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F


EDGE_COLUMNS = [
    "source_id",
    "target_id",
    "source_type",
    "target_type",
    "edge_type",
    "similarity_score",
    "rank",
]


def load_embeddings(embedding_path: Path) -> tuple[torch.Tensor, list[str]]:
    obj = torch.load(embedding_path, map_location="cpu")
    return obj["embeddings"].float(), obj["protein_ids"]


def _features_for_ids(embedding_path: Path, protein_ids: list[str]) -> torch.Tensor:
    embeddings, all_ids = load_embeddings(embedding_path)
    by_id = {protein_id: embeddings[i] for i, protein_id in enumerate(all_ids)}
    return torch.stack([by_id[protein_id] for protein_id in protein_ids])


def build_symmetric_knn_edges(
    node_ids: list[str],
    embeddings: torch.Tensor,
    edge_type: str,
    source_type: str,
    target_type: str,
    k: int = 10,
    block_size: int = 1024,
    min_similarity: float | None = None,
) -> pd.DataFrame:
    if len(node_ids) != embeddings.shape[0]:
        raise ValueError("node_ids and embeddings must have the same length")
    if len(node_ids) < 2:
        return pd.DataFrame()
    k = min(k, len(node_ids) - 1)
    x = F.normalize(embeddings.float(), p=2, dim=1)
    edge_by_pair: dict[tuple[int, int], dict] = {}
    for start in range(0, x.shape[0], block_size):
        end = min(start + block_size, x.shape[0])
        scores = x[start:end] @ x.T
        row_index = torch.arange(start, end)
        scores[torch.arange(end - start), row_index] = float("-inf")
        if min_similarity is None:
            values, indices = torch.topk(scores, k=k, dim=1)
        else:
            values = []
            indices = []
            for local_i in range(end - start):
                candidate_idx = torch.nonzero(
                    scores[local_i] > min_similarity, as_tuple=False
                ).flatten()
                if candidate_idx.numel() == 0:
                    values.append(torch.empty(0, dtype=scores.dtype))
                    indices.append(torch.empty(0, dtype=torch.long))
                    continue
                candidate_scores = scores[local_i, candidate_idx]
                keep = min(k, candidate_scores.numel())
                row_values, row_order = torch.topk(candidate_scores, k=keep)
                values.append(row_values)
                indices.append(candidate_idx[row_order])
        for local_i in range(end - start):
            source_i = start + local_i
            row_values = values[local_i]
            row_indices = indices[local_i]
            for rank in range(len(row_indices)):
                target_i = int(row_indices[rank])
                score = float(row_values[rank])
                for a, b in [(source_i, target_i), (target_i, source_i)]:
                    key = (a, b)
                    existing = edge_by_pair.get(key)
                    if existing is None or score > existing["similarity_score"]:
                        edge_by_pair[key] = {
                            "source_id": node_ids[a],
                            "target_id": node_ids[b],
                            "source_type": source_type,
                            "target_type": target_type,
                            "edge_type": edge_type,
                            "similarity_score": score,
                            "rank": rank + 1,
                        }
    if not edge_by_pair:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    return pd.DataFrame(edge_by_pair.values(), columns=EDGE_COLUMNS).sort_values(
        ["source_id", "rank", "target_id"]
    )


def write_similarity_networks(
    data_dir: Path,
    embedding_path: Path,
    output_dir: Path,
    k: int = 10,
    block_size: int = 1024,
    min_similarity: float | None = None,
    node_set: str = "both",
) -> dict:
    if node_set not in {"both", "host", "pathogen"}:
        raise ValueError("node_set must be 'both', 'host', or 'pathogen'")
    host_ids = pd.read_csv(data_dir / "host_nodes.tsv", sep="\t")["host_uniprot"].astype(str).tolist()
    pathogen_ids = pd.read_csv(data_dir / "pathogen_nodes.tsv", sep="\t")[
        "pathogen_uniprot"
    ].astype(str).tolist()
    host_edges = pd.DataFrame(columns=EDGE_COLUMNS)
    pathogen_edges = pd.DataFrame(columns=EDGE_COLUMNS)
    if node_set in {"both", "host"}:
        host_x = _features_for_ids(embedding_path, host_ids)
        host_edges = build_symmetric_knn_edges(
            node_ids=[f"host_protein:{protein_id}" for protein_id in host_ids],
            embeddings=host_x,
            edge_type="similar_to",
            source_type="host_protein",
            target_type="host_protein",
            k=k,
            block_size=block_size,
            min_similarity=min_similarity,
    )
    if node_set in {"both", "pathogen"}:
        pathogen_x = _features_for_ids(embedding_path, pathogen_ids)
        pathogen_edges = build_symmetric_knn_edges(
            node_ids=[f"pathogen_protein:{protein_id}" for protein_id in pathogen_ids],
            embeddings=pathogen_x,
            edge_type="similar_to",
            source_type="pathogen_protein",
            target_type="pathogen_protein",
            k=k,
            block_size=block_size,
            min_similarity=min_similarity,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    host_edges.to_csv(
        output_dir / "edges_host_protein_similar_to_host_protein.tsv",
        sep="\t",
        index=False,
    )
    pathogen_edges.to_csv(
        output_dir / "edges_pathogen_protein_similar_to_pathogen_protein.tsv",
        sep="\t",
        index=False,
    )
    summary = {
        "k": k,
        "min_similarity": min_similarity,
        "host_nodes": len(host_ids),
        "pathogen_nodes": len(pathogen_ids),
        "host_similarity_edges": int(len(host_edges)),
        "pathogen_similarity_edges": int(len(pathogen_edges)),
        "embedding_path": str(embedding_path),
        "output_dir": str(output_dir),
        "node_set": node_set,
    }
    (output_dir / "similarity_network_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("data/processed/embeddings/esm2_t12_35M_UR50D_protein_mean_embeddings.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi_hetero_graph"),
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument(
        "--min-similarity",
        type=float,
        help="Keep only similarity edges with cosine similarity greater than this value.",
    )
    parser.add_argument(
        "--node-set",
        choices=["both", "host", "pathogen"],
        default="both",
        help="Build host, pathogen, or both similarity networks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_similarity_networks(
        data_dir=args.data_dir,
        embedding_path=args.embedding_path,
        output_dir=args.output_dir,
        k=args.k,
        block_size=args.block_size,
        min_similarity=args.min_similarity,
        node_set=args.node_set,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
