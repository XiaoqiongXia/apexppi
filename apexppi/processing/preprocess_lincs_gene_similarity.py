#!/usr/bin/env python3
"""Build LINCS-derived human gene-gene similarity matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "drugbank": "consensi-drugbank.tsv",
    "knockdown": "consensi-knockdown.tsv",
    "overexpression": "consensi-overexpression.tsv",
}


def _read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "perturbagen" not in frame.columns:
        raise ValueError(f"{path} does not contain a perturbagen column")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing values")
    return frame


def _pearson_gene_similarity(values: np.ndarray) -> np.ndarray:
    similarity = np.corrcoef(values, rowvar=False).astype(np.float32)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _topk_positive_edges(
    similarity: np.ndarray,
    gene_ids: list[str],
    top_k: int,
    source: str,
) -> pd.DataFrame:
    edge_ranks: dict[tuple[int, int], dict[int, int]] = {}
    n_genes = len(gene_ids)

    for i in range(n_genes):
        scores = similarity[i].copy()
        scores[i] = -np.inf
        positive = np.flatnonzero(scores > 0)
        if len(positive) == 0:
            continue
        ordered = positive[np.argsort(scores[positive])[::-1]]
        for rank, j in enumerate(ordered[:top_k], start=1):
            key = (min(i, j), max(i, j))
            edge_ranks.setdefault(key, {})[i] = rank

    rows = []
    for (i, j), ranks in sorted(
        edge_ranks.items(),
        key=lambda item: (-float(similarity[item[0][0], item[0][1]]), item[0][0], item[0][1]),
    ):
        rank_i = ranks.get(i)
        rank_j = ranks.get(j)
        rows.append(
            {
                "gene_i": gene_ids[i],
                "gene_j": gene_ids[j],
                "similarity": float(similarity[i, j]),
                "rank_i": rank_i,
                "rank_j": rank_j,
                "is_mutual": rank_i is not None and rank_j is not None,
                "source": source,
            }
        )
    edges = pd.DataFrame(
        rows,
        columns=[
            "gene_i",
            "gene_j",
            "similarity",
            "rank_i",
            "rank_j",
            "is_mutual",
            "source",
        ],
    )
    if not edges.empty:
        edges["rank_i"] = edges["rank_i"].astype("Int64")
        edges["rank_j"] = edges["rank_j"].astype("Int64")
    return edges


def _write_gene_ids(output_dir: Path, gene_ids: list[str]) -> None:
    pd.DataFrame(
        {
            "gene_index": list(range(len(gene_ids))),
            "entrez_id": gene_ids,
        }
    ).to_csv(output_dir / "gene_ids.tsv", sep="\t", index=False)


def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    clip_min: float = -10.0,
    clip_max: float = 10.0,
    top_k: int = 50,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    gene_columns: list[str] | None = None
    for source, filename in DATASETS.items():
        frame = _read_matrix(input_dir / filename)
        current_gene_columns = [str(column) for column in frame.columns[1:]]
        if gene_columns is None:
            gene_columns = current_gene_columns
        elif current_gene_columns != gene_columns:
            raise ValueError(f"Gene columns do not match for {source}")
        frames[source] = frame

    assert gene_columns is not None
    _write_gene_ids(output_dir, gene_columns)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_files": DATASETS,
        "similarity_metric": "pearson",
        "clip_range": [clip_min, clip_max],
        "top_k": top_k,
        "edge_policy": "undirected positive top-k per gene, excluding self-loops",
        "n_genes": len(gene_columns),
        "datasets": {},
    }

    for source, frame in frames.items():
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
        before_min = float(np.min(values))
        before_max = float(np.max(values))
        np.clip(values, clip_min, clip_max, out=values)
        after_min = float(np.min(values))
        after_max = float(np.max(values))

        similarity = _pearson_gene_similarity(values)
        np.save(output_dir / f"{source}_gene_similarity.npy", similarity)

        edges = _topk_positive_edges(
            similarity=similarity,
            gene_ids=gene_columns,
            top_k=top_k,
            source=source,
        )
        edges.to_csv(
            output_dir / f"{source}_gene_similarity_top{top_k}_positive_edges.tsv",
            sep="\t",
            index=False,
        )

        summary["datasets"][source] = {
            "input_file": str(input_dir / DATASETS[source]),
            "n_perturbagens": int(frame.shape[0]),
            "n_genes": len(gene_columns),
            "missing_values": int(frame.isna().sum().sum()),
            "value_range_before_clip": [before_min, before_max],
            "value_range_after_clip": [after_min, after_max],
            "n_positive_topk_edges": int(len(edges)),
        }

    with (output_dir / "processing_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build gene-gene Pearson similarity matrices from LINCS consensus signatures."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_data/LINCS"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/lincs_gene_similarity"),
    )
    parser.add_argument("--clip-min", type=float, default=-10.0)
    parser.add_argument("--clip-max", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_preprocessing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
