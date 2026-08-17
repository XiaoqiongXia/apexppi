#!/usr/bin/env python3
"""Build pure-threshold host ESM2 cosine similarity edges without top-k capping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from apexppi.processing.build_similarity_network_phase8 import EDGE_COLUMNS, load_embeddings


def build_host_similarity_threshold_edges(
    data_dir: Path,
    embedding_path: Path,
    output_dir: Path,
    threshold: float,
    block_size: int = 512,
) -> dict:
    host_ids = (
        pd.read_csv(data_dir / "host_nodes.tsv", sep="\t")["host_uniprot"]
        .astype(str)
        .tolist()
    )
    embeddings, protein_ids = load_embeddings(embedding_path)
    embedding_by_id = {protein_id: embeddings[i] for i, protein_id in enumerate(protein_ids)}
    missing = sorted(set(host_ids).difference(embedding_by_id))
    if missing:
        raise KeyError(f"Missing ESM2 embeddings for {len(missing)} host proteins: {missing[:5]}")

    x = torch.stack([embedding_by_id[protein_id] for protein_id in host_ids]).float()
    x = F.normalize(x, p=2, dim=1)
    rows = []
    n = x.shape[0]
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        scores = x[start:end] @ x.T
        for local_i in range(end - start):
            source_i = start + local_i
            candidate_idx = torch.nonzero(scores[local_i, source_i + 1 :] > threshold).flatten()
            if candidate_idx.numel() == 0:
                continue
            target_idx = candidate_idx + source_i + 1
            target_scores = scores[local_i, target_idx]
            order = torch.argsort(target_scores, descending=True)
            for rank, order_i in enumerate(order.tolist(), start=1):
                target_i = int(target_idx[order_i])
                rows.append(
                    {
                        "source_id": f"host_protein:{host_ids[source_i]}",
                        "target_id": f"host_protein:{host_ids[target_i]}",
                        "source_type": "host_protein",
                        "target_type": "host_protein",
                        "edge_type": "similar_to",
                        "similarity_score": float(target_scores[order_i]),
                        "rank": rank,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "edges_host_protein_similar_to_host_protein.tsv"
    pd.DataFrame(rows, columns=EDGE_COLUMNS).to_csv(edge_path, sep="\t", index=False)
    summary = {
        "data_dir": str(data_dir),
        "embedding_path": str(embedding_path),
        "edge_path": str(edge_path),
        "host_nodes": len(host_ids),
        "threshold": threshold,
        "host_similarity_edges_undirected": len(rows),
        "host_similarity_edges_bidirectional_after_graph_build": len(rows) * 2,
        "graph_rule": "pure_threshold_no_topk",
        "similarity_metric": "ESM2_t12_35M_UR50D_mean_embedding_cosine_similarity",
    }
    (output_dir / "host_similarity_threshold_edge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("data/processed/embeddings/esm2_t12_35M_UR50D_protein_mean_embeddings.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.97644)
    parser.add_argument("--block-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_host_similarity_threshold_edges(
        data_dir=args.data_dir,
        embedding_path=args.embedding_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
