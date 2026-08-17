#!/usr/bin/env python3
"""Build NCBI taxonomy lineage tables for HPIDB pathogen taxa."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip"


def extract_numeric_taxids(taxid_set: str) -> list[str]:
    seen = set()
    taxids = []
    for taxid in re.findall(r"taxid:([^|(]+)", str(taxid_set)):
        taxid = taxid.strip()
        if taxid and taxid not in seen:
            seen.add(taxid)
            taxids.append(taxid)
    return taxids


def _split_dmp_line(line: str) -> list[str]:
    return [part.strip() for part in line.rstrip("\n").split("|")][:-1]


def parse_nodes_dmp(path: Path) -> dict[str, dict[str, str]]:
    nodes = {}
    with path.open() as handle:
        for line in handle:
            parts = _split_dmp_line(line)
            if len(parts) < 3:
                continue
            nodes[parts[0]] = {"parent_taxid": parts[1], "rank": parts[2]}
    return nodes


def parse_names_dmp(path: Path) -> dict[str, str]:
    names = {}
    with path.open() as handle:
        for line in handle:
            parts = _split_dmp_line(line)
            if len(parts) < 4:
                continue
            if parts[3] == "scientific name":
                names[parts[0]] = parts[1]
    return names


def ensure_taxdump(taxonomy_dir: Path, url: str = TAXDUMP_URL) -> None:
    taxonomy_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = taxonomy_dir / "nodes.dmp"
    names_path = taxonomy_dir / "names.dmp"
    if nodes_path.exists() and names_path.exists():
        return
    archive_path = taxonomy_dir / "taxdmp.zip"
    if not archive_path.exists():
        urllib.request.urlretrieve(url, archive_path)
    with zipfile.ZipFile(archive_path) as zf:
        for member in ["nodes.dmp", "names.dmp"]:
            zf.extract(member, taxonomy_dir)


def lineage_for_taxid(taxid: str, nodes: dict[str, dict[str, str]]) -> list[str]:
    lineage = []
    current = taxid
    seen = set()
    while current and current not in seen and current in nodes:
        seen.add(current)
        lineage.append(current)
        parent = nodes[current]["parent_taxid"]
        if parent == current:
            break
        current = parent
    return lineage


def build_taxonomy_lineage_tables(
    pathogen_nodes: pd.DataFrame,
    taxonomy_dir: Path,
    output_dir: Path,
) -> dict:
    nodes = parse_nodes_dmp(taxonomy_dir / "nodes.dmp")
    names = parse_names_dmp(taxonomy_dir / "names.dmp")
    observed_taxids: set[str] = set()
    protein_edges = []
    for row in pathogen_nodes.to_dict("records"):
        pathogen_uniprot = row["pathogen_uniprot"]
        for taxid in extract_numeric_taxids(row.get("pathogen_taxid_set", "")):
            observed_taxids.add(taxid)
            protein_edges.append(
                {
                    "source_id": f"pathogen_protein:{pathogen_uniprot}",
                    "target_id": f"taxon:taxid:{taxid}",
                    "source_type": "pathogen_protein",
                    "target_type": "taxon",
                    "edge_type": "belongs_to_taxon",
                    "pathogen_uniprot": pathogen_uniprot,
                    "taxon_id": f"taxid:{taxid}",
                }
            )

    all_taxids: set[str] = set()
    parent_edges = []
    missing_taxids = []
    for taxid in sorted(observed_taxids):
        lineage = lineage_for_taxid(taxid, nodes)
        if not lineage:
            missing_taxids.append(taxid)
            continue
        all_taxids.update(lineage)
        for child, parent in zip(lineage, lineage[1:]):
            parent_edges.append(
                {
                    "source_id": f"taxon:taxid:{child}",
                    "target_id": f"taxon:taxid:{parent}",
                    "source_type": "taxon",
                    "target_type": "taxon",
                    "edge_type": "is_a_parent",
                    "child_taxon_id": f"taxid:{child}",
                    "parent_taxon_id": f"taxid:{parent}",
                }
            )

    taxon_nodes = pd.DataFrame(
        [
            {
                "node_id": f"taxon:taxid:{taxid}",
                "node_type": "taxon",
                "taxon_id": f"taxid:{taxid}",
                "taxon_name": names.get(taxid, ""),
                "rank": nodes.get(taxid, {}).get("rank", ""),
                "is_observed_pathogen_taxon": int(taxid in observed_taxids),
            }
            for taxid in sorted(all_taxids, key=lambda value: int(value) if value.isdigit() else value)
        ]
    )
    parent_edges_frame = pd.DataFrame(parent_edges).drop_duplicates()
    protein_edges_frame = pd.DataFrame(protein_edges).drop_duplicates()

    output_dir.mkdir(parents=True, exist_ok=True)
    taxon_nodes.to_csv(output_dir / "nodes_taxon_lineage.tsv", sep="\t", index=False)
    parent_edges_frame.to_csv(output_dir / "edges_taxon_child_parent.tsv", sep="\t", index=False)
    protein_edges_frame.to_csv(
        output_dir / "edges_pathogen_protein_belongs_to_taxon_lineage.tsv",
        sep="\t",
        index=False,
    )
    summary = {
        "observed_taxa": len(observed_taxids),
        "lineage_taxon_nodes": int(len(taxon_nodes)),
        "taxon_parent_edges": int(len(parent_edges_frame)),
        "pathogen_protein_to_taxon_edges": int(len(protein_edges_frame)),
        "missing_taxids": missing_taxids,
        "taxonomy_dir": str(taxonomy_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "taxonomy_lineage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument(
        "--taxonomy-dir",
        type=Path,
        default=Path("data/external/ncbi_taxonomy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi_hetero_graph"),
    )
    parser.add_argument("--taxdump-url", default=TAXDUMP_URL)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_download:
        ensure_taxdump(args.taxonomy_dir, args.taxdump_url)
    pathogen_nodes = pd.read_csv(args.data_dir / "pathogen_nodes.tsv", sep="\t")
    summary = build_taxonomy_lineage_tables(
        pathogen_nodes=pathogen_nodes,
        taxonomy_dir=args.taxonomy_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
