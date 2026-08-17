#!/usr/bin/env python3
"""Evaluate pathogen-to-human ranking for ApexPPI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import torch

from apexppi.inference.predict_interaction import ApexPPIPredictor


def ndcg_at_k(ranked_relevance: list[int], k: int) -> float:
    gains = ranked_relevance[:k]
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(gains))
    ideal = sorted(ranked_relevance, reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def compute_query_ranking(
    pathogen_uniprot: str,
    host_ids: list[str],
    scores: torch.Tensor,
    positive_hosts: set[str],
    ks: Iterable[int] = (1, 5, 10, 50, 100),
) -> dict[str, float | int | str]:
    if len(host_ids) != int(scores.numel()):
        raise ValueError("host_ids and scores must have the same length")
    if not positive_hosts:
        raise ValueError("positive_hosts must not be empty")

    ordered_indices = torch.argsort(scores, descending=True).tolist()
    ranked_hosts = [host_ids[i] for i in ordered_indices]
    rank_by_host = {host_id: rank + 1 for rank, host_id in enumerate(ranked_hosts)}
    positive_ranks = sorted(rank_by_host[host_id] for host_id in positive_hosts)
    ranked_relevance = [1 if host_id in positive_hosts else 0 for host_id in ranked_hosts]

    row: dict[str, float | int | str] = {
        "pathogen_uniprot": pathogen_uniprot,
        "n_candidates": len(host_ids),
        "n_positives": len(positive_hosts),
        "best_rank": int(positive_ranks[0]),
        "mean_rank": float(sum(positive_ranks) / len(positive_ranks)),
        "median_rank": float(pd.Series(positive_ranks).median()),
        "mrr": float(1.0 / positive_ranks[0]),
    }
    for k in ks:
        positives_in_top_k = sum(1 for rank in positive_ranks if rank <= k)
        row[f"hits_at_{k}"] = float(positives_in_top_k > 0)
        row[f"recall_at_{k}"] = float(positives_in_top_k / len(positive_hosts))
        row[f"ndcg_at_{k}"] = ndcg_at_k(ranked_relevance, k)
    return row


def summarize_per_query(per_query: pd.DataFrame, ks: Iterable[int]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "queries": int(len(per_query)),
        "positives": int(per_query["n_positives"].sum()),
        "mean_best_rank": float(per_query["best_rank"].mean()),
        "median_best_rank": float(per_query["best_rank"].median()),
        "mean_positive_rank": float(per_query["mean_rank"].mean()),
        "mean_mrr": float(per_query["mrr"].mean()),
    }
    for k in ks:
        summary[f"mean_hits_at_{k}"] = float(per_query[f"hits_at_{k}"].mean())
        summary[f"mean_recall_at_{k}"] = float(per_query[f"recall_at_{k}"].mean())
        summary[f"mean_ndcg_at_{k}"] = float(per_query[f"ndcg_at_{k}"].mean())
    return summary


def evaluate_grouped_rankings(
    positives: pd.DataFrame,
    host_ids: list[str],
    score_pathogen: Callable[[str], torch.Tensor],
    ks: Iterable[int] = (1, 5, 10, 50, 100),
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = []
    for pathogen_uniprot, group in positives.groupby("pathogen_uniprot", sort=True):
        positive_hosts = set(group["host_uniprot"])
        scores = score_pathogen(pathogen_uniprot)
        rows.append(
            compute_query_ranking(
                pathogen_uniprot=pathogen_uniprot,
                host_ids=host_ids,
                scores=scores,
                positive_hosts=positive_hosts,
                ks=ks,
            )
        )
    per_query = pd.DataFrame(rows)
    return per_query, summarize_per_query(per_query, ks)


def evaluate_model_ranking(
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    positives_path: Path,
    heterodata_path: Path | None = None,
    device_name: str = "cpu",
    batch_size: int = 8192,
    ks: Iterable[int] = (1, 5, 10, 50, 100),
) -> dict:
    predictor = ApexPPIPredictor(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        heterodata_path=heterodata_path,
        device_name=device_name,
    )
    positives = pd.read_csv(positives_path, sep="\t")

    def score_pathogen(pathogen_uniprot: str) -> torch.Tensor:
        ranking = predictor.score_pathogen_against_hosts(
            pathogen_uniprot=pathogen_uniprot,
            candidate_host_ids=predictor.host_ids,
            batch_size=batch_size,
        )
        scores_by_host = dict(zip(ranking["host_uniprot"], ranking["logit"]))
        return torch.tensor([scores_by_host[host_id] for host_id in predictor.host_ids])

    per_query, summary = evaluate_grouped_rankings(
        positives=positives,
        host_ids=predictor.host_ids,
        score_pathogen=score_pathogen,
        ks=ks,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "apexppi_ranking"
    per_query_path = output_dir / f"{prefix}_per_pathogen.tsv"
    summary_path = output_dir / f"{prefix}_summary.json"
    per_query.to_csv(per_query_path, sep="\t", index=False)
    result = {
        "model_type": "apexppi",
        "checkpoint_path": str(checkpoint_path),
        "positives_path": str(positives_path),
        "candidate_hosts": len(predictor.host_ids),
        "pathogen_queries": int(len(per_query)),
        "ks": list(ks),
        "per_query_path": str(per_query_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument(
        "--positives-path",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi/test_positive_edges.tsv"),
    )
    parser.add_argument("--heterodata-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/ranking"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ks", default="1,5,10,50,100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = tuple(int(item) for item in args.ks.split(",") if item)
    result = evaluate_model_ranking(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        positives_path=args.positives_path,
        heterodata_path=args.heterodata_path,
        device_name=args.device,
        batch_size=args.batch_size,
        ks=ks,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
