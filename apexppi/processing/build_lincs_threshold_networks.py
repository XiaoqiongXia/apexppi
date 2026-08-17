#!/usr/bin/env python3
"""Build thresholded LINCS gene-gene similarity networks from dense matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_THRESHOLDS = {
    "drugbank": 0.679373,
    "knockdown": 0.592236,
    "overexpression": 0.579805,
}


def build_threshold_edges(
    similarity: np.ndarray,
    gene_ids: list[str],
    threshold: float,
    source: str,
) -> pd.DataFrame:
    if similarity.shape != (len(gene_ids), len(gene_ids)):
        raise ValueError("similarity matrix shape must match gene_ids length")
    rows, cols = np.where(np.triu(similarity, k=1) >= threshold)
    records = [
        {
            "gene_i": gene_ids[int(i)],
            "gene_j": gene_ids[int(j)],
            "similarity": float(similarity[int(i), int(j)]),
            "source": source,
        }
        for i, j in zip(rows, cols)
    ]
    edges = pd.DataFrame(records, columns=["gene_i", "gene_j", "similarity", "source"])
    if not edges.empty:
        edges = edges.sort_values(
            ["similarity", "gene_i", "gene_j"], ascending=[False, True, True]
        ).reset_index(drop=True)
    return edges


def _read_gene_ids(processed_dir: Path) -> list[str]:
    genes = pd.read_csv(processed_dir / "gene_ids.tsv", sep="\t")
    if "entrez_id" not in genes.columns:
        raise ValueError("gene_ids.tsv must contain an entrez_id column")
    return genes["entrez_id"].astype(str).tolist()


def write_lincs_threshold_networks(
    processed_dir: Path,
    thresholds: dict[str, float] | None = None,
) -> dict:
    processed_dir = Path(processed_dir)
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    gene_ids = _read_gene_ids(processed_dir)
    summary = {
        "processed_dir": str(processed_dir),
        "similarity_metric": "pearson",
        "edge_policy": "undirected thresholded edges, excluding self-loops",
        "n_genes": len(gene_ids),
        "datasets": {},
    }

    for source, threshold in thresholds.items():
        matrix_path = processed_dir / f"{source}_gene_similarity.npy"
        similarity = np.load(matrix_path)
        edges = build_threshold_edges(
            similarity=similarity,
            gene_ids=gene_ids,
            threshold=float(threshold),
            source=source,
        )
        output_path = processed_dir / f"{source}_gene_similarity_threshold_edges.tsv"
        edges.to_csv(output_path, sep="\t", index=False)
        n_edges = int(len(edges))
        summary["datasets"][source] = {
            "threshold": float(threshold),
            "matrix_path": str(matrix_path),
            "edge_path": str(output_path),
            "n_edges": n_edges,
            "density": float(n_edges / (len(gene_ids) * (len(gene_ids) - 1) / 2))
            if len(gene_ids) > 1
            else 0.0,
            "mean_degree": float(2 * n_edges / len(gene_ids)) if gene_ids else 0.0,
        }

    summary_path = processed_dir / "threshold_network_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_thresholds(value: str) -> dict[str, float]:
    thresholds = {}
    for item in value.split(","):
        if not item:
            continue
        source, threshold = item.split("=", maxsplit=1)
        thresholds[source.strip()] = float(threshold)
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/lincs_gene_similarity"),
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(
            f"{source}={threshold}" for source, threshold in DEFAULT_THRESHOLDS.items()
        ),
        help="Comma-separated source=threshold entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_lincs_threshold_networks(
        processed_dir=args.processed_dir,
        thresholds=_parse_thresholds(args.thresholds),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
