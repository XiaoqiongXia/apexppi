#!/usr/bin/env python3
"""Map LINCS Entrez gene-gene similarity edges to current host UniProt protein edges."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


EDGE_COLUMNS = [
    "source_id",
    "target_id",
    "source_type",
    "target_type",
    "edge_type",
    "similarity_score",
    "source",
]


def load_entrez_to_host_uniprot(
    host_nodes_path: Path,
    uniprot_geneid_mapping_path: Path,
) -> dict[str, list[str]]:
    host_ids = set(
        pd.read_csv(host_nodes_path, sep="\t")["host_uniprot"].astype(str).tolist()
    )
    mapping = pd.read_csv(uniprot_geneid_mapping_path, sep="\t", dtype=str).fillna("")
    if "Entry" not in mapping.columns or "GeneID" not in mapping.columns:
        raise ValueError("UniProt mapping must contain Entry and GeneID columns")
    entrez_to_uniprot: dict[str, set[str]] = defaultdict(set)
    for row in mapping[["Entry", "GeneID"]].itertuples(index=False):
        uniprot = str(row.Entry)
        if uniprot not in host_ids:
            continue
        for gene_id in str(row.GeneID).replace(";", " ").replace(",", " ").split():
            gene_id = gene_id.strip()
            if gene_id:
                entrez_to_uniprot[gene_id].add(uniprot)
    return {
        gene_id: sorted(uniprot_ids)
        for gene_id, uniprot_ids in sorted(entrez_to_uniprot.items())
    }


def map_lincs_edges(
    edge_path: Path,
    entrez_to_uniprot: dict[str, list[str]],
    output_path: Path,
    relation_source: str,
) -> dict:
    edges = pd.read_csv(edge_path, sep="\t", dtype={"gene_i": str, "gene_j": str})
    edge_by_pair: dict[tuple[str, str], float] = {}
    mapped_gene_edges = 0
    expanded_edges = 0
    for row in edges.itertuples(index=False):
        source_proteins = entrez_to_uniprot.get(str(row.gene_i), [])
        target_proteins = entrez_to_uniprot.get(str(row.gene_j), [])
        if not source_proteins or not target_proteins:
            continue
        mapped_gene_edges += 1
        score = float(row.similarity)
        for source in source_proteins:
            for target in target_proteins:
                if source == target:
                    continue
                a, b = sorted((source, target))
                key = (a, b)
                expanded_edges += 1
                if key not in edge_by_pair or score > edge_by_pair[key]:
                    edge_by_pair[key] = score

    records = [
        {
            "source_id": f"host_protein:{source}",
            "target_id": f"host_protein:{target}",
            "source_type": "host_protein",
            "target_type": "host_protein",
            "edge_type": "similar_to",
            "similarity_score": score,
            "source": relation_source,
        }
        for (source, target), score in sorted(edge_by_pair.items())
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=EDGE_COLUMNS).to_csv(output_path, sep="\t", index=False)
    return {
        "input_edge_path": str(edge_path),
        "output_edge_path": str(output_path),
        "source": relation_source,
        "input_gene_edges": int(len(edges)),
        "mapped_gene_edges": int(mapped_gene_edges),
        "expanded_protein_edge_candidates": int(expanded_edges),
        "deduplicated_protein_edges": int(len(records)),
    }


def map_all_lincs_sources(
    lincs_dir: Path,
    host_nodes_path: Path,
    uniprot_geneid_mapping_path: Path,
    output_dir: Path,
) -> dict:
    entrez_to_uniprot = load_entrez_to_host_uniprot(
        host_nodes_path=host_nodes_path,
        uniprot_geneid_mapping_path=uniprot_geneid_mapping_path,
    )
    summary = {
        "lincs_dir": str(lincs_dir),
        "host_nodes_path": str(host_nodes_path),
        "uniprot_geneid_mapping_path": str(uniprot_geneid_mapping_path),
        "output_dir": str(output_dir),
        "mapped_entrez_genes": len(entrez_to_uniprot),
        "sources": {},
    }
    for source in ["drugbank", "knockdown", "overexpression"]:
        edge_path = lincs_dir / f"{source}_gene_similarity_threshold_edges.tsv"
        output_path = output_dir / f"edges_host_protein_lincs_{source}_similar_to_host_protein.tsv"
        summary["sources"][source] = map_lincs_edges(
            edge_path=edge_path,
            entrez_to_uniprot=entrez_to_uniprot,
            output_path=output_path,
            relation_source=source,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lincs_host_protein_edge_mapping_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lincs-dir",
        type=Path,
        default=Path("data/processed/lincs_gene_similarity"),
    )
    parser.add_argument(
        "--host-nodes-path",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi/host_nodes.tsv"),
    )
    parser.add_argument(
        "--uniprot-geneid-mapping-path",
        type=Path,
        default=Path("data/external/uniprot/uniprot_human_geneid_mapping.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/lincs_host_protein_similarity_threshold"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_all_lincs_sources(
        lincs_dir=args.lincs_dir,
        host_nodes_path=args.host_nodes_path,
        uniprot_geneid_mapping_path=args.uniprot_geneid_mapping_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
