#!/usr/bin/env python3
"""Build query-aware blinded-edge HPIDB splits for unknown-edge discovery."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd


EDGE_COLUMNS = ["host_uniprot", "pathogen_uniprot"]


def _sample_negative_edges(
    hosts: list[str],
    pathogens: list[str],
    all_positive: set[tuple[str, str]],
    count: int,
    rng: random.Random,
) -> pd.DataFrame:
    capacity = len(hosts) * len(pathogens) - len(all_positive)
    if count > capacity:
        raise ValueError(f"Cannot sample {count} negatives; only {capacity} candidates exist")
    negatives: set[tuple[str, str]] = set()
    while len(negatives) < count:
        candidate = (rng.choice(hosts), rng.choice(pathogens))
        if candidate not in all_positive:
            negatives.add(candidate)
    return pd.DataFrame(sorted(negatives), columns=EDGE_COLUMNS).assign(label=0)


def _copy_node_tables(source_dir: Path, output_dir: Path) -> None:
    for name in ("host_nodes.tsv", "pathogen_nodes.tsv", "viral_protein_nodes.tsv"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def _make_supervision(
    positives: pd.DataFrame,
    hosts: list[str],
    pathogens: list[str],
    all_positive: set[tuple[str, str]],
    negatives_per_positive: int,
    rng: random.Random,
    seed: int,
) -> pd.DataFrame:
    pos = positives[EDGE_COLUMNS].copy()
    pos["label"] = 1
    neg = _sample_negative_edges(
        hosts=hosts,
        pathogens=pathogens,
        all_positive=all_positive,
        count=len(pos) * negatives_per_positive,
        rng=rng,
    )
    frame = pd.concat([pos, neg], ignore_index=True)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _select_blinded_edges(
    positive_edges: pd.DataFrame,
    mode: str,
    seed: int,
    min_query_degree: int,
    valid_edges_per_query: int,
    test_edges_per_query: int,
    min_train_edges_per_endpoint: int,
    max_queries: int | None,
) -> tuple[set[int], set[int], pd.DataFrame]:
    query_col = "host_uniprot" if mode == "host" else "pathogen_uniprot"
    host_degree = positive_edges.groupby("host_uniprot").size().to_dict()
    pathogen_degree = positive_edges.groupby("pathogen_uniprot").size().to_dict()

    query_degree = positive_edges.groupby(query_col).size()
    queries = query_degree[query_degree >= min_query_degree].index.astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(queries)
    if max_queries is not None:
        queries = queries[:max_queries]

    valid_indices: set[int] = set()
    test_indices: set[int] = set()
    query_rows = []

    def can_hide(row: pd.Series, h_degree: dict[str, int], p_degree: dict[str, int]) -> bool:
        host = str(row["host_uniprot"])
        pathogen = str(row["pathogen_uniprot"])
        return (
            h_degree[host] - 1 >= min_train_edges_per_endpoint
            and p_degree[pathogen] - 1 >= min_train_edges_per_endpoint
        )

    for query in queries:
        rows = positive_edges[positive_edges[query_col].astype(str) == query]
        candidate_indices = list(rows.index)
        rng.shuffle(candidate_indices)

        tmp_host_degree = dict(host_degree)
        tmp_pathogen_degree = dict(pathogen_degree)
        tmp_valid: list[int] = []
        tmp_test: list[int] = []
        for split_name, target, bucket in (
            ("valid", valid_edges_per_query, tmp_valid),
            ("test", test_edges_per_query, tmp_test),
        ):
            for idx in candidate_indices:
                if idx in tmp_valid or idx in tmp_test:
                    continue
                row = positive_edges.loc[idx]
                if not can_hide(row, tmp_host_degree, tmp_pathogen_degree):
                    continue
                host = str(row["host_uniprot"])
                pathogen = str(row["pathogen_uniprot"])
                tmp_host_degree[host] -= 1
                tmp_pathogen_degree[pathogen] -= 1
                bucket.append(int(idx))
                if len(bucket) == target:
                    break
            if len(bucket) < target:
                break

        if len(tmp_valid) != valid_edges_per_query or len(tmp_test) != test_edges_per_query:
            continue

        host_degree = tmp_host_degree
        pathogen_degree = tmp_pathogen_degree
        valid_indices.update(tmp_valid)
        test_indices.update(tmp_test)
        query_rows.append(
            {
                "query_mode": mode,
                "query_uniprot": query,
                "valid_blinded_edges": len(tmp_valid),
                "test_blinded_edges": len(tmp_test),
            }
        )

    if not valid_indices or not test_indices:
        raise ValueError("No blinded edges selected; relax degree or endpoint constraints")
    return valid_indices, test_indices, pd.DataFrame(query_rows)


def build_blinded_edge_split(
    source_dir: Path,
    output_dir: Path,
    mode: str,
    seed: int,
    min_query_degree: int,
    valid_edges_per_query: int,
    test_edges_per_query: int,
    min_train_edges_per_endpoint: int,
    max_queries: int | None,
    negatives_per_positive: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_edges = pd.read_csv(source_dir / "positive_edges.tsv", sep="\t")
    positive_edges.to_csv(output_dir / "positive_edges.tsv", sep="\t", index=False)
    _copy_node_tables(source_dir, output_dir)

    valid_indices, test_indices, query_frame = _select_blinded_edges(
        positive_edges=positive_edges,
        mode=mode,
        seed=seed,
        min_query_degree=min_query_degree,
        valid_edges_per_query=valid_edges_per_query,
        test_edges_per_query=test_edges_per_query,
        min_train_edges_per_endpoint=min_train_edges_per_endpoint,
        max_queries=max_queries,
    )
    hidden_indices = valid_indices | test_indices
    train_positive = positive_edges.loc[
        [idx for idx in positive_edges.index if idx not in hidden_indices]
    ].copy()
    valid_positive = positive_edges.loc[sorted(valid_indices)].copy()
    test_positive = positive_edges.loc[sorted(test_indices)].copy()

    hidden_pairs = set(zip(valid_positive["host_uniprot"], valid_positive["pathogen_uniprot"]))
    hidden_pairs |= set(zip(test_positive["host_uniprot"], test_positive["pathogen_uniprot"]))
    train_pairs = set(zip(train_positive["host_uniprot"], train_positive["pathogen_uniprot"]))
    leakage = hidden_pairs & train_pairs
    if leakage:
        raise AssertionError(f"Hidden positives leaked into train positives: {sorted(leakage)[:5]}")

    hosts = sorted(positive_edges["host_uniprot"].astype(str).unique())
    pathogens = sorted(positive_edges["pathogen_uniprot"].astype(str).unique())
    all_positive = set(zip(positive_edges["host_uniprot"], positive_edges["pathogen_uniprot"]))
    rng = random.Random(seed)
    for split, positives in (
        ("train", train_positive),
        ("valid", valid_positive),
        ("test", test_positive),
    ):
        frame = _make_supervision(
            positives=positives,
            hosts=hosts,
            pathogens=pathogens,
            all_positive=all_positive,
            negatives_per_positive=negatives_per_positive,
            rng=rng,
            seed=seed,
        )
        frame.to_csv(output_dir / f"{split}_edges.tsv", sep="\t", index=False)
        positives.to_csv(output_dir / f"{split}_positive_edges.tsv", sep="\t", index=False)

    train_positive[EDGE_COLUMNS].to_csv(
        output_dir / "message_passing_train_edges.tsv", sep="\t", index=False
    )
    query_frame.to_csv(output_dir / "blinded_edge_queries.tsv", sep="\t", index=False)

    train_host_degree = train_positive.groupby("host_uniprot").size()
    train_pathogen_degree = train_positive.groupby("pathogen_uniprot").size()
    hidden_hosts = set(valid_positive["host_uniprot"]) | set(test_positive["host_uniprot"])
    hidden_pathogens = set(valid_positive["pathogen_uniprot"]) | set(test_positive["pathogen_uniprot"])
    missing_hidden_hosts = sorted(host for host in hidden_hosts if host not in train_host_degree)
    missing_hidden_pathogens = sorted(
        pathogen for pathogen in hidden_pathogens if pathogen not in train_pathogen_degree
    )
    if missing_hidden_hosts or missing_hidden_pathogens:
        raise AssertionError("At least one hidden-edge endpoint is absent from the training HP-PPI graph")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "split_type": "query_aware_blinded_edge",
        "query_mode": mode,
        "seed": seed,
        "min_query_degree": min_query_degree,
        "valid_edges_per_query": valid_edges_per_query,
        "test_edges_per_query": test_edges_per_query,
        "min_train_edges_per_endpoint": min_train_edges_per_endpoint,
        "max_queries": max_queries,
        "negatives_per_positive": negatives_per_positive,
        "total_positive_edges": len(positive_edges),
        "train_positive_edges": len(train_positive),
        "valid_positive_edges": len(valid_positive),
        "test_positive_edges": len(test_positive),
        "train_negative_edges": len(train_positive) * negatives_per_positive,
        "valid_negative_edges": len(valid_positive) * negatives_per_positive,
        "test_negative_edges": len(test_positive) * negatives_per_positive,
        "query_count": len(query_frame),
        "hidden_edge_train_leakage_count": len(leakage),
        "hidden_endpoint_missing_from_train_hp_graph_count": len(missing_hidden_hosts)
        + len(missing_hidden_pathogens),
        "total_host_nodes": int(positive_edges["host_uniprot"].nunique()),
        "total_pathogen_nodes": int(positive_edges["pathogen_uniprot"].nunique()),
    }
    (output_dir / "blinded_edge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--mode", choices=["host", "pathogen", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-query-degree", type=int, default=3)
    parser.add_argument("--valid-edges-per-query", type=int, default=1)
    parser.add_argument("--test-edges-per-query", type=int, default=1)
    parser.add_argument("--min-train-edges-per-endpoint", type=int, default=1)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ["host", "pathogen"] if args.mode == "both" else [args.mode]
    summaries = []
    for mode in modes:
        output_dir = args.output_root / f"hpidb_human_ppi_blinded_edge_{mode}_seed{args.seed}"
        summaries.append(
            build_blinded_edge_split(
                source_dir=args.source_dir,
                output_dir=output_dir,
                mode=mode,
                seed=args.seed,
                min_query_degree=args.min_query_degree,
                valid_edges_per_query=args.valid_edges_per_query,
                test_edges_per_query=args.test_edges_per_query,
                min_train_edges_per_endpoint=args.min_train_edges_per_endpoint,
                max_queries=args.max_queries,
                negatives_per_positive=args.negatives_per_positive,
            )
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
