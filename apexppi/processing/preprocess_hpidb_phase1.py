#!/usr/bin/env python3
"""Phase 1 preprocessing for HPIDB human-pathogen PPI link prediction."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MITAB_COLUMNS = [
    "protein_xref_1",
    "protein_xref_2",
    "alternative_identifiers_1",
    "alternative_identifiers_2",
    "protein_alias_1",
    "protein_alias_2",
    "detection_method",
    "author_name",
    "pmid",
    "protein_taxid_1",
    "protein_taxid_2",
    "interaction_type",
    "source_database_id",
    "database_identifier",
    "confidence",
]


@dataclass(frozen=True)
class FastaRecord:
    accession: str
    fasta_header: str
    protein_name: str
    organism: str
    taxid: str
    sequence: str


def extract_uniprot_accession(field: str) -> str | None:
    """Return the last UniProt accession in a MITAB xref field, without isoform."""
    hits = []
    for part in str(field).split("|"):
        if part.startswith("uniprotkb:"):
            accession = part.split(":", 1)[1].split("-", 1)[0]
            if re.fullmatch(r"[A-Z0-9]+", accession):
                hits.append(accession)
    return hits[-1] if hits else None


def _parse_fasta_header(header: str) -> tuple[str, str, str, str]:
    token, _, description = header.partition(" ")
    parts = token.split("|")
    accession = parts[1] if len(parts) >= 2 else parts[0]
    accession = accession.split("-", 1)[0]

    protein_name = description
    organism = ""
    taxid = ""
    if " OS=" in description:
        protein_name, rest = description.split(" OS=", 1)
        if " OX=" in rest:
            organism, rest = rest.split(" OX=", 1)
            taxid = rest.split()[0]
        else:
            organism = rest
    return accession, protein_name.strip(), organism.strip(), taxid.strip()


def parse_fasta_records(fasta_path: Path) -> dict[str, FastaRecord]:
    records: dict[str, FastaRecord] = {}
    current_header: str | None = None
    sequence_parts: list[str] = []

    def flush_record() -> None:
        if current_header is None:
            return
        accession, protein_name, organism, taxid = _parse_fasta_header(current_header)
        records[accession] = FastaRecord(
            accession=accession,
            fasta_header=current_header,
            protein_name=protein_name,
            organism=organism,
            taxid=taxid,
            sequence="".join(sequence_parts),
        )

    with fasta_path.open(errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush_record()
                current_header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line.strip())
    flush_record()
    return records


def parse_fasta_accessions(fasta_path: Path) -> set[str]:
    return set(parse_fasta_records(fasta_path))


def _read_mitab_rows(mitab_path: Path):
    with mitab_path.open(errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames and reader.fieldnames[0].startswith("# "):
            reader.fieldnames[0] = reader.fieldnames[0][2:]
        for row in reader:
            yield row


def _join_unique(values) -> str:
    cleaned = sorted({str(value) for value in values if value and str(value) != "-"})
    return "|".join(cleaned)


def _taxid_names(taxid_set: str) -> str:
    names = []
    for match in re.finditer(r"taxid:[^(]+\(([^)]*)\)", str(taxid_set)):
        name = match.group(1)
        if "|" in name:
            name = name.split("|")[-1]
        names.append(name.strip())
    return _join_unique(names)


def _is_viral_candidate(*fields: str) -> int:
    text = " ".join(str(field).lower() for field in fields if field)
    viral_patterns = [
        r"\bvirus\b",
        r"\bviral\b",
        r"\bvirion\b",
        r"\bviroid\b",
        r"\bhiv\b",
        r"\bsiv\b",
        r"\bhtlv\b",
        r"\bhpv\b",
        r"\bhcv\b",
        r"\bhhv\b",
        r"\bhbv\b",
        r"\badenovirus\b",
        r"\bherpesvirus\b",
        r"\bpapillomavirus\b",
        r"\binfluenza\b",
        r"\bdengue\b",
        r"\bzika\b",
        r"\bebola\b",
        r"\bcoronavirus\b",
        r"\brotavirus\b",
        r"\bretrovirus\b",
        r"\bpolyomavirus\b",
        r"\bpoxvirus\b",
    ]
    return int(any(re.search(pattern, text) for pattern in viral_patterns))


def _fasta_metadata(accession: str, fasta_records: dict[str, FastaRecord]) -> dict:
    record = fasta_records.get(accession)
    if record is None:
        return {
            "protein_name": "",
            "organism": "",
            "fasta_taxid": "",
            "sequence_length": 0,
            "sequence": "",
            "fasta_header": "",
        }
    return {
        "protein_name": record.protein_name,
        "organism": record.organism,
        "fasta_taxid": record.taxid,
        "sequence_length": len(record.sequence),
        "sequence": record.sequence,
        "fasta_header": record.fasta_header,
    }


def write_node_annotation_tables(
    positive_edges: pd.DataFrame, fasta_records: dict[str, FastaRecord], output_dir: Path
) -> dict:
    host_counts = positive_edges.groupby("host_uniprot").size().to_dict()
    pathogen_counts = positive_edges.groupby("pathogen_uniprot").size().to_dict()

    host_records = []
    for accession, count in sorted(host_counts.items()):
        row = {"host_uniprot": accession, "n_positive_edges": int(count)}
        row.update(_fasta_metadata(accession, fasta_records))
        host_records.append(row)

    pathogen_records = []
    for accession, group in positive_edges.groupby("pathogen_uniprot", sort=True):
        taxid_set = _join_unique(group["pathogen_taxid_set"])
        name_set = _taxid_names(taxid_set)
        row = {
            "pathogen_uniprot": accession,
            "pathogen_taxid_set": taxid_set,
            "pathogen_name_set": name_set,
            "n_positive_edges": int(pathogen_counts[accession]),
        }
        row.update(_fasta_metadata(accession, fasta_records))
        row["is_viral_candidate"] = _is_viral_candidate(
            row["pathogen_taxid_set"],
            row["pathogen_name_set"],
            row["organism"],
        )
        pathogen_records.append(row)

    host_nodes = pd.DataFrame.from_records(host_records)
    pathogen_nodes = pd.DataFrame.from_records(pathogen_records)
    viral_nodes = pathogen_nodes[pathogen_nodes["is_viral_candidate"] == 1].copy()
    if not viral_nodes.empty:
        viral_nodes = viral_nodes.rename(
            columns={
                "pathogen_uniprot": "viral_uniprot",
                "pathogen_taxid_set": "viral_taxid_set",
                "pathogen_name_set": "viral_name_set",
            }
        )
    else:
        viral_nodes = pd.DataFrame(
            columns=[
                "viral_uniprot",
                "viral_taxid_set",
                "viral_name_set",
                "n_positive_edges",
                "protein_name",
                "organism",
                "fasta_taxid",
                "sequence_length",
                "sequence",
                "fasta_header",
                "is_viral_candidate",
            ]
        )

    host_nodes.to_csv(output_dir / "host_nodes.tsv", sep="\t", index=False)
    pathogen_nodes.to_csv(output_dir / "pathogen_nodes.tsv", sep="\t", index=False)
    viral_nodes.to_csv(output_dir / "viral_protein_nodes.tsv", sep="\t", index=False)

    return {
        "host_node_table_rows": len(host_nodes),
        "pathogen_node_table_rows": len(pathogen_nodes),
        "viral_candidate_node_table_rows": len(viral_nodes),
    }


def build_positive_edges(mitab_path: Path, fasta_path: Path) -> tuple[pd.DataFrame, dict]:
    fasta_accessions = parse_fasta_accessions(fasta_path)
    edge_metadata = defaultdict(lambda: defaultdict(list))
    summary = {
        "total_rows": 0,
        "homo_sapiens_rows": 0,
        "both_uniprot_rows": 0,
        "fasta_backed_rows": 0,
        "missing_host_sequence_rows": 0,
        "missing_pathogen_sequence_rows": 0,
        "fasta_accessions": len(fasta_accessions),
    }

    for row in _read_mitab_rows(mitab_path):
        summary["total_rows"] += 1
        if not row.get("protein_taxid_1", "").startswith("taxid:9606("):
            continue
        summary["homo_sapiens_rows"] += 1

        host = extract_uniprot_accession(row.get("protein_xref_1", ""))
        pathogen = extract_uniprot_accession(row.get("protein_xref_2", ""))
        if not host or not pathogen:
            continue
        summary["both_uniprot_rows"] += 1

        host_has_sequence = host in fasta_accessions
        pathogen_has_sequence = pathogen in fasta_accessions
        if not host_has_sequence:
            summary["missing_host_sequence_rows"] += 1
        if not pathogen_has_sequence:
            summary["missing_pathogen_sequence_rows"] += 1
        if not host_has_sequence or not pathogen_has_sequence:
            continue
        summary["fasta_backed_rows"] += 1

        key = (host, pathogen)
        metadata = edge_metadata[key]
        metadata["pmid"].append(row.get("pmid", ""))
        metadata["source_database_id"].append(row.get("source_database_id", ""))
        metadata["detection_method"].append(row.get("detection_method", ""))
        metadata["interaction_type"].append(row.get("interaction_type", ""))
        metadata["database_identifier"].append(row.get("database_identifier", ""))
        metadata["confidence"].append(row.get("confidence", ""))
        metadata["protein_taxid_2"].append(row.get("protein_taxid_2", ""))

    records = []
    for (host, pathogen), metadata in sorted(edge_metadata.items()):
        records.append(
            {
                "host_uniprot": host,
                "pathogen_uniprot": pathogen,
                "evidence_count": len(metadata["pmid"]),
                "pmid_set": _join_unique(metadata["pmid"]),
                "source_database_set": _join_unique(metadata["source_database_id"]),
                "detection_method_set": _join_unique(metadata["detection_method"]),
                "interaction_type_set": _join_unique(metadata["interaction_type"]),
                "database_identifier_set": _join_unique(metadata["database_identifier"]),
                "confidence_set": _join_unique(metadata["confidence"]),
                "pathogen_taxid_set": _join_unique(metadata["protein_taxid_2"]),
            }
        )

    positive_edges = pd.DataFrame.from_records(records)
    summary["unique_positive_edges"] = len(positive_edges)
    summary["host_nodes"] = int(positive_edges["host_uniprot"].nunique()) if len(positive_edges) else 0
    summary["pathogen_nodes"] = (
        int(positive_edges["pathogen_uniprot"].nunique()) if len(positive_edges) else 0
    )
    return positive_edges, summary


def _split_positive_edges(
    positive_edges: pd.DataFrame,
    seed: int,
    train_frac: float,
    valid_frac: float,
    test_frac: float,
) -> dict[str, pd.DataFrame]:
    if abs((train_frac + valid_frac + test_frac) - 1.0) > 1e-8:
        raise ValueError("train_frac + valid_frac + test_frac must equal 1.0")

    rng = random.Random(seed)
    indices = list(positive_edges.index)
    rng.shuffle(indices)

    uncovered_hosts = set(positive_edges["host_uniprot"])
    uncovered_pathogens = set(positive_edges["pathogen_uniprot"])
    required_train: set[int] = set()
    for idx in indices:
        row = positive_edges.loc[idx]
        covers_new_node = (
            row["host_uniprot"] in uncovered_hosts
            or row["pathogen_uniprot"] in uncovered_pathogens
        )
        if not covers_new_node:
            continue
        required_train.add(int(idx))
        uncovered_hosts.discard(row["host_uniprot"])
        uncovered_pathogens.discard(row["pathogen_uniprot"])
        if not uncovered_hosts and not uncovered_pathogens:
            break
    if uncovered_hosts or uncovered_pathogens:
        raise ValueError("Could not create a train split covering all nodes")

    target_train = max(int(round(len(indices) * train_frac)), len(required_train))
    target_valid = int(round(len(indices) * valid_frac))
    train_indices = set(required_train)
    remaining = [idx for idx in indices if idx not in train_indices]

    for idx in remaining:
        if len(train_indices) >= target_train:
            break
        train_indices.add(idx)
    remaining = [idx for idx in remaining if idx not in train_indices]

    valid_count = min(target_valid, len(remaining))
    valid_indices = set(remaining[:valid_count])
    test_indices = set(remaining[valid_count:])

    return {
        "train": positive_edges.loc[sorted(train_indices)].copy(),
        "valid": positive_edges.loc[sorted(valid_indices)].copy(),
        "test": positive_edges.loc[sorted(test_indices)].copy(),
    }


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

    return pd.DataFrame(
        sorted(negatives), columns=["host_uniprot", "pathogen_uniprot"]
    ).assign(label=0)


def run_preprocessing(
    mitab_path: Path,
    fasta_path: Path,
    output_dir: Path,
    seed: int = 42,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    negatives_per_positive: int = 1,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_edges, summary = build_positive_edges(mitab_path, fasta_path)
    if positive_edges.empty:
        raise ValueError("No positive edges remained after preprocessing")

    positive_edges.to_csv(output_dir / "positive_edges.tsv", sep="\t", index=False)
    fasta_records = parse_fasta_records(fasta_path)
    node_summary = write_node_annotation_tables(positive_edges, fasta_records, output_dir)

    split_positives = _split_positive_edges(
        positive_edges, seed=seed, train_frac=train_frac, valid_frac=valid_frac, test_frac=test_frac
    )
    all_positive = set(zip(positive_edges["host_uniprot"], positive_edges["pathogen_uniprot"]))
    hosts = sorted(positive_edges["host_uniprot"].unique())
    pathogens = sorted(positive_edges["pathogen_uniprot"].unique())
    rng = random.Random(seed)

    split_counts = {}
    for split, pos_frame in split_positives.items():
        pos_supervision = pos_frame[["host_uniprot", "pathogen_uniprot"]].copy()
        pos_supervision["label"] = 1
        neg_count = len(pos_supervision) * negatives_per_positive
        neg_supervision = _sample_negative_edges(hosts, pathogens, all_positive, neg_count, rng)
        split_frame = pd.concat([pos_supervision, neg_supervision], ignore_index=True)
        split_frame = split_frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_frame.to_csv(output_dir / f"{split}_edges.tsv", sep="\t", index=False)
        pos_frame.to_csv(output_dir / f"{split}_positive_edges.tsv", sep="\t", index=False)
        split_counts[f"{split}_positive_edges"] = len(pos_frame)
        split_counts[f"{split}_negative_edges"] = len(neg_supervision)

    train_positive = split_positives["train"][["host_uniprot", "pathogen_uniprot"]]
    train_positive.to_csv(output_dir / "message_passing_train_edges.tsv", sep="\t", index=False)

    summary.update(split_counts)
    summary.update(node_summary)
    summary.update(
        {
            "seed": seed,
            "train_frac": train_frac,
            "valid_frac": valid_frac,
            "test_frac": test_frac,
            "negatives_per_positive": negatives_per_positive,
        }
    )
    with (output_dir / "preprocessing_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mitab",
        type=Path,
        default=Path("data/raw_data/data_processed/hpidb2_protein_taxid_1_homo_sapiens.tsv"),
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/raw_data/HPIDB/idmapping_2024_09_19.fasta/idmapping_2024_09_19.fasta"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preprocessing(
        mitab_path=args.mitab,
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        seed=args.seed,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
        negatives_per_positive=args.negatives_per_positive,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
