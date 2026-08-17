#!/usr/bin/env python3
"""Generate mean-pooled ESM2 embeddings for HPIDB protein nodes."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch


MODEL_NAME = "esm2_t12_35M_UR50D"
MODEL_LAYER = 12


def project_cache_paths(project_root: Path, model_name: str = MODEL_NAME) -> dict[str, Path]:
    return {
        "model_dir": project_root / "models" / model_name,
        "torch_home": project_root / "models" / "torch",
        "output_dir": project_root / "data" / "processed" / "embeddings",
    }


def _clean_sequence(sequence: str) -> str:
    return "".join(str(sequence).upper().split())


def build_embedding_records(input_dir: Path, max_residues: int = 1022) -> list[dict]:
    host_path = input_dir / "host_nodes.tsv"
    pathogen_path = input_dir / "pathogen_nodes.tsv"
    if not host_path.exists():
        raise FileNotFoundError(host_path)
    if not pathogen_path.exists():
        raise FileNotFoundError(pathogen_path)

    records_by_id: dict[str, dict] = {}
    for node_type, path, id_column in [
        ("host", host_path, "host_uniprot"),
        ("pathogen", pathogen_path, "pathogen_uniprot"),
    ]:
        frame = pd.read_csv(path, sep="\t")
        for row in frame.itertuples(index=False):
            row_dict = row._asdict()
            protein_id = str(row_dict[id_column])
            sequence = _clean_sequence(row_dict.get("sequence", ""))
            if not sequence:
                continue
            original_sequence_length = len(sequence)
            sequence_for_embedding = sequence[:max_residues]
            if protein_id not in records_by_id:
                records_by_id[protein_id] = {
                    "protein_id": protein_id,
                    "sequence": sequence_for_embedding,
                    "original_sequence_length": original_sequence_length,
                    "sequence_length": len(sequence_for_embedding),
                    "was_truncated": int(original_sequence_length > len(sequence_for_embedding)),
                    "node_types": set(),
                }
            records_by_id[protein_id]["node_types"].add(node_type)

    records = []
    for protein_id in sorted(records_by_id):
        record = records_by_id[protein_id]
        record = {
            **record,
            "node_types": "|".join(sorted(record["node_types"])),
        }
        records.append(record)
    return records


def mean_pool_residue_representations(
    token_representations: torch.Tensor, sequence_lengths: list[int]
) -> torch.Tensor:
    pooled = []
    for i, sequence_length in enumerate(sequence_lengths):
        pooled.append(token_representations[i, 1 : sequence_length + 1].mean(dim=0))
    return torch.stack(pooled, dim=0)


def make_token_limited_batches(
    records: list[dict], max_tokens_per_batch: int
) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_max_length = 0

    for record in sorted(records, key=lambda item: item["sequence_length"]):
        proposed_max_length = max(current_max_length, record["sequence_length"])
        proposed_tokens = proposed_max_length * (len(current) + 1)
        if current and proposed_tokens > max_tokens_per_batch:
            batches.append(current)
            current = [record]
            current_max_length = record["sequence_length"]
        else:
            current.append(record)
            current_max_length = proposed_max_length
    if current:
        batches.append(current)
    return batches


def _load_esm2_model(model_name: str, project_root: Path, device: torch.device):
    paths = project_cache_paths(project_root, model_name)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["torch_home"].mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(paths["torch_home"])
    torch.hub.set_dir(str(paths["torch_home"] / "hub"))

    try:
        import esm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The `esm` package is not installed. Install fair-esm before running "
            "Phase 2 embedding generation."
        ) from exc

    if model_name != MODEL_NAME:
        raise ValueError(f"Unsupported model_name={model_name!r}; expected {MODEL_NAME!r}")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model = model.eval().to(device)
    return model, alphabet


def generate_embeddings(
    records: list[dict],
    project_root: Path,
    output_dir: Path,
    batch_size: int = 8,
    max_tokens_per_batch: int = 4096,
    device_name: str = "auto",
    model_name: str = MODEL_NAME,
) -> dict:
    if not records:
        raise ValueError("No protein records were available for embedding generation")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model, alphabet = _load_esm2_model(model_name, project_root, device)
    batch_converter = alphabet.get_batch_converter()

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = output_dir / f"{model_name}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    embeddings = []
    metadata_rows = []
    batches = make_token_limited_batches(records, max_tokens_per_batch)
    started_at = time.time()

    with torch.no_grad():
        for batch_index, batch_records in enumerate(batches, start=1):
            if batch_size > 0 and len(batch_records) > batch_size:
                sub_batches = [
                    batch_records[start : start + batch_size]
                    for start in range(0, len(batch_records), batch_size)
                ]
            else:
                sub_batches = [batch_records]
            for sub_index, sub_batch_records in enumerate(sub_batches, start=1):
                batch_records = sub_batch_records
                labels_and_sequences = [
                    (record["protein_id"], record["sequence"]) for record in batch_records
                ]
                _, _, tokens = batch_converter(labels_and_sequences)
                tokens = tokens.to(device)
                result = model(tokens, repr_layers=[MODEL_LAYER], return_contacts=False)
                token_representations = result["representations"][MODEL_LAYER].detach().cpu()
                sequence_lengths = [record["sequence_length"] for record in batch_records]
                batch_embeddings = mean_pool_residue_representations(
                    token_representations, sequence_lengths
                )
                embeddings.append(batch_embeddings)
                metadata_rows.extend(
                    {
                        "protein_id": record["protein_id"],
                        "node_types": record["node_types"],
                        "original_sequence_length": record["original_sequence_length"],
                        "sequence_length": record["sequence_length"],
                        "was_truncated": record["was_truncated"],
                        "embedding_model": model_name,
                        "pooling_method": "residue_mean_excluding_special_tokens",
                        "embedding_dim": token_representations.shape[-1],
                    }
                    for record in batch_records
                )

            if batch_index % 25 == 0 or batch_index == len(batches):
                processed = len(metadata_rows)
                elapsed = time.time() - started_at
                print(
                    f"[embedding] batch_group={batch_index}/{len(batches)} "
                    f"proteins={processed}/{len(records)} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )
                torch.save(
                    {
                        "embeddings": torch.cat(embeddings, dim=0),
                        "metadata": metadata_rows,
                        "processed": processed,
                    },
                    chunk_dir / "latest.pt",
                )

    embedding_tensor = torch.cat(embeddings, dim=0)
    tensor_path = output_dir / f"{model_name}_protein_mean_embeddings.pt"
    metadata_path = output_dir / f"{model_name}_protein_metadata.tsv"
    summary_path = output_dir / f"{model_name}_embedding_summary.json"

    torch.save(
        {
            "embeddings": embedding_tensor,
            "protein_ids": [row["protein_id"] for row in metadata_rows],
            "node_types": [row["node_types"] for row in metadata_rows],
            "model_name": model_name,
            "pooling_method": "residue_mean_excluding_special_tokens",
        },
        tensor_path,
    )
    pd.DataFrame(metadata_rows).to_csv(metadata_path, sep="\t", index=False)

    summary = {
        "model_name": model_name,
        "model_layer": MODEL_LAYER,
        "n_proteins": len(records),
        "n_truncated": int(sum(record["was_truncated"] for record in records)),
        "max_tokens_per_batch": max_tokens_per_batch,
        "n_batch_groups": len(batches),
        "embedding_dim": int(embedding_tensor.shape[1]),
        "device": str(device),
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
        "torch_home": str(project_cache_paths(project_root, model_name)["torch_home"]),
        "model_dir": str(project_cache_paths(project_root, model_name)["model_dir"]),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/hpidb_human_ppi"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/embeddings"),
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens-per-batch", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write the protein list that would be embedded without loading ESM2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    records = build_embedding_records(args.input_dir, max_residues=args.max_residues)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata_only:
        metadata_path = args.output_dir / f"{args.model_name}_input_proteins.tsv"
        pd.DataFrame(records).to_csv(metadata_path, sep="\t", index=False)
        print(json.dumps({"n_proteins": len(records), "metadata_path": str(metadata_path)}))
        return

    summary = generate_embeddings(
        records=records,
        project_root=project_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_tokens_per_batch=args.max_tokens_per_batch,
        device_name=args.device,
        model_name=args.model_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
