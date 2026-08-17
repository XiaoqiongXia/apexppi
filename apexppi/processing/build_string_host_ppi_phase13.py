#!/usr/bin/env python3
"""Build host-host PPI edges from STRING human PPI using exact sequence mapping."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(chunks)
    return sequences


def build_exact_sequence_mapping(
    host_nodes: pd.DataFrame, string_sequences: dict[str, str]
) -> dict[str, str]:
    sequence_to_hosts: dict[str, list[str]] = defaultdict(list)
    for row in host_nodes[["host_uniprot", "sequence"]].dropna().itertuples(index=False):
        sequence = str(row.sequence).strip()
        if sequence:
            sequence_to_hosts[sequence].append(str(row.host_uniprot))
    unique_sequence_to_host = {
        sequence: hosts[0] for sequence, hosts in sequence_to_hosts.items() if len(set(hosts)) == 1
    }
    mapping = {}
    for string_id, sequence in string_sequences.items():
        host_uniprot = unique_sequence_to_host.get(sequence)
        if host_uniprot is not None:
            mapping[string_id] = host_uniprot
    return mapping


def build_string_host_ppi_edges(
    host_nodes: pd.DataFrame,
    string_sequences: dict[str, str],
    links_path: Path,
    min_score: int = 700,
) -> tuple[pd.DataFrame, dict]:
    mapping = build_exact_sequence_mapping(host_nodes, string_sequences)
    host_set = set(host_nodes["host_uniprot"].astype(str))
    edge_by_pair: dict[tuple[str, str], int] = {}
    total_links = 0
    passing_score_links = 0
    mapped_links = 0
    with links_path.open() as handle:
        header = handle.readline().strip().split()
        col_index = {name: idx for idx, name in enumerate(header)}
        for line in handle:
            total_links += 1
            fields = line.strip().split()
            if not fields:
                continue
            score = int(fields[col_index["combined_score"]])
            if score < min_score:
                continue
            passing_score_links += 1
            host1 = mapping.get(fields[col_index["protein1"]])
            host2 = mapping.get(fields[col_index["protein2"]])
            if host1 is None or host2 is None or host1 == host2:
                continue
            if host1 not in host_set or host2 not in host_set:
                continue
            mapped_links += 1
            for source, target in [(host1, host2), (host2, host1)]:
                key = (source, target)
                edge_by_pair[key] = max(score, edge_by_pair.get(key, -1))

    records = [
        {
            "source_id": f"host_protein:{source}",
            "target_id": f"host_protein:{target}",
            "source_type": "host_protein",
            "target_type": "host_protein",
            "edge_type": "string_interacts",
            "host_uniprot_1": source,
            "host_uniprot_2": target,
            "combined_score": score,
        }
        for (source, target), score in sorted(edge_by_pair.items())
    ]
    edges = pd.DataFrame(
        records,
        columns=[
            "source_id",
            "target_id",
            "source_type",
            "target_type",
            "edge_type",
            "host_uniprot_1",
            "host_uniprot_2",
            "combined_score",
        ],
    )
    mapped_host_counts = Counter(mapping.values())
    summary = {
        "min_score": min_score,
        "host_nodes": int(len(host_nodes)),
        "string_sequences": int(len(string_sequences)),
        "mapped_string_proteins": int(len(mapping)),
        "mapped_host_proteins": int(len(mapped_host_counts)),
        "total_string_links": int(total_links),
        "passing_score_links": int(passing_score_links),
        "mapped_passing_score_links": int(mapped_links),
        "host_ppi_edges": int(len(edges)),
    }
    return edges, summary


def build_string_host_ppi_tables(
    data_dir: Path,
    string_dir: Path,
    output_dir: Path,
    min_score: int = 700,
) -> dict:
    host_nodes = pd.read_csv(data_dir / "host_nodes.tsv", sep="\t")
    string_sequences = parse_fasta_sequences(string_dir / "9606.protein.sequences.v12.0.fa")
    edges, summary = build_string_host_ppi_edges(
        host_nodes=host_nodes,
        string_sequences=string_sequences,
        links_path=string_dir / "9606.protein.links.v12.0.txt",
        min_score=min_score,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(
        output_dir / "edges_host_protein_interacts_host_protein_string.tsv",
        sep="\t",
        index=False,
    )
    summary.update(
        {
            "data_dir": str(data_dir),
            "string_dir": str(string_dir),
            "output_dir": str(output_dir),
            "edge_table": "edges_host_protein_interacts_host_protein_string.tsv",
            "mapping": "exact_protein_sequence_match_only",
        }
    )
    (output_dir / "string_host_ppi_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument("--string-dir", type=Path, default=Path("data/raw_data/PPI_human"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi_hetero_graph"),
    )
    parser.add_argument("--min-score", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_string_host_ppi_tables(
        data_dir=args.data_dir,
        string_dir=args.string_dir,
        output_dir=args.output_dir,
        min_score=args.min_score,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
