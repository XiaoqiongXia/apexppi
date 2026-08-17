#!/usr/bin/env python3
"""Score host-pathogen protein pairs with a trained ApexPPI checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from apexppi.models import ApexPPI
from apexppi.models.data import load_graph


RANKING_COLUMNS = [
    "host_uniprot",
    "pathogen_uniprot",
    "known_hpidb_positive",
    "logit",
    "interaction_probability",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


class ApexPPIPredictor:
    def __init__(
        self,
        checkpoint_path: Path,
        data_dir: Path,
        heterodata_path: Path | None = None,
        device_name: str = "cpu",
    ):
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        data_dir = Path(data_dir).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False"
            )
        self.device = torch.device(device_name)
        self.data_dir = data_dir
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        supported_model_types = {
            "apexppi",
            "unified_protein_lorentz_learnable_curvature_relation_attention",
        }
        if self.checkpoint.get("model_type") not in supported_model_types:
            raise ValueError("Checkpoint is not a final ApexPPI checkpoint")

        graph_path = self._resolve_graph_path(
            checkpoint_path=checkpoint_path,
            data_dir=data_dir,
            heterodata_path=heterodata_path,
        )
        self.data, self.node_maps = load_graph(graph_path)
        self.data = self.data.to(self.device)
        self.host_to_idx = {
            key.removeprefix("host_protein:"): value
            for key, value in self.node_maps["host_protein"].items()
        }
        self.pathogen_to_idx = {
            key.removeprefix("pathogen_protein:"): value
            for key, value in self.node_maps["pathogen_protein"].items()
        }
        self.host_ids = list(self.host_to_idx)

        self.model = ApexPPI(
            edge_types=[
                tuple(edge_type) for edge_type in self.checkpoint["edge_types"]
            ],
            input_dim=self.checkpoint["input_dim"],
            hidden_dim=self.checkpoint["hidden_dim"],
            dropout=self.checkpoint["dropout"],
            num_layers=self.checkpoint.get("num_layers", 2),
            max_tangent_norm=self.checkpoint.get("max_tangent_norm", 5.0),
            initial_curvature=self.checkpoint.get("initial_curvature", 1.0),
            min_curvature=self.checkpoint.get("min_curvature", 1e-4),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        self._encoded: torch.Tensor | None = None

        self.host_annotations = self._read_optional_table(data_dir / "host_nodes.tsv")
        self.pathogen_annotations = self._read_optional_table(
            data_dir / "pathogen_nodes.tsv"
        )
        positives = self._read_optional_table(data_dir / "positive_edges.tsv")
        self.known_positive_pairs = (
            set(zip(positives["host_uniprot"], positives["pathogen_uniprot"]))
            if not positives.empty
            else set()
        )

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: Path | str,
        device_name: str = "cpu",
    ) -> "ApexPPIPredictor":
        """Load the conventional model/data layout documented by ApexPPI."""
        bundle_dir = Path(bundle_dir).expanduser().resolve()
        return cls(
            checkpoint_path=bundle_dir / "models" / "apexppi" / "apexppi_best.pt",
            data_dir=bundle_dir / "data" / "processed" / "hpidb_human_ppi",
            heterodata_path=(
                bundle_dir
                / "data"
                / "processed"
                / "hpidb_human_ppi_unified_protein_graph"
                / "heterodata_unified_protein.pt"
            ),
            device_name=device_name,
        )

    def _resolve_graph_path(
        self,
        checkpoint_path: Path,
        data_dir: Path,
        heterodata_path: Path | None,
    ) -> Path:
        candidates: list[Path] = []
        if heterodata_path is not None:
            candidates.append(Path(heterodata_path).expanduser())
        checkpoint_graph = self.checkpoint.get("heterodata_path")
        if checkpoint_graph:
            candidates.append(Path(checkpoint_graph).expanduser())
        candidates.extend(
            [
                data_dir.parent
                / "hpidb_human_ppi_unified_protein_graph"
                / "heterodata_unified_protein.pt",
                checkpoint_path.parent / "heterodata_unified_protein.pt",
            ]
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        searched = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "A matching processed graph was not found. Pass heterodata_path explicitly. "
            f"Searched:\n  - {searched}"
        )

    @staticmethod
    def _read_optional_table(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()

    def encode(self) -> torch.Tensor:
        if self._encoded is None:
            with torch.no_grad():
                self._encoded = self.model.encode(self.data)
        return self._encoded

    def score_indices(
        self, host_idx: torch.Tensor, pathogen_idx: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model.decode(
                self.encode(),
                host_idx.to(self.device),
                pathogen_idx.to(self.device),
            )
        return logits.cpu()

    def score_pair(self, host_uniprot: str, pathogen_uniprot: str) -> dict:
        if host_uniprot not in self.host_to_idx:
            raise KeyError(f"Unknown host UniProt ID: {host_uniprot}")
        if pathogen_uniprot not in self.pathogen_to_idx:
            raise KeyError(f"Unknown pathogen UniProt ID: {pathogen_uniprot}")
        host_idx = torch.tensor([self.host_to_idx[host_uniprot]])
        pathogen_idx = torch.tensor([self.pathogen_to_idx[pathogen_uniprot]])
        logit = float(self.score_indices(host_idx, pathogen_idx)[0])
        result = {
            "model_type": "apexppi",
            "host_uniprot": host_uniprot,
            "pathogen_uniprot": pathogen_uniprot,
            "known_hpidb_positive": (host_uniprot, pathogen_uniprot)
            in self.known_positive_pairs,
            "logit": logit,
            "interaction_probability": float(torch.sigmoid(torch.tensor(logit))),
        }
        result.update(self._annotation(self.host_annotations, "host", host_uniprot))
        result.update(
            self._annotation(self.pathogen_annotations, "pathogen", pathogen_uniprot)
        )
        return result

    def score_pathogen_against_hosts(
        self,
        pathogen_uniprot: str,
        candidate_host_ids: list[str] | None = None,
        batch_size: int = 8192,
    ) -> pd.DataFrame:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if pathogen_uniprot not in self.pathogen_to_idx:
            raise KeyError(f"Unknown pathogen UniProt ID: {pathogen_uniprot}")
        host_ids = self.host_ids if candidate_host_ids is None else candidate_host_ids
        if not host_ids:
            return pd.DataFrame(columns=RANKING_COLUMNS)
        unknown = [
            protein_id for protein_id in host_ids if protein_id not in self.host_to_idx
        ]
        if unknown:
            raise KeyError(f"Unknown host UniProt IDs: {unknown[:5]}")

        rows = []
        pathogen_index = self.pathogen_to_idx[pathogen_uniprot]
        for start in range(0, len(host_ids), batch_size):
            batch_ids = host_ids[start : start + batch_size]
            host_idx = torch.tensor(
                [self.host_to_idx[protein_id] for protein_id in batch_ids]
            )
            pathogen_idx = torch.full((len(batch_ids),), pathogen_index)
            logits = self.score_indices(host_idx, pathogen_idx)
            for protein_id, logit, probability in zip(
                batch_ids, logits.tolist(), torch.sigmoid(logits).tolist()
            ):
                rows.append(
                    {
                        "host_uniprot": protein_id,
                        "pathogen_uniprot": pathogen_uniprot,
                        "known_hpidb_positive": (protein_id, pathogen_uniprot)
                        in self.known_positive_pairs,
                        "logit": float(logit),
                        "interaction_probability": float(probability),
                    }
                )
        return pd.DataFrame(rows).sort_values(
            ["interaction_probability", "host_uniprot"], ascending=[False, True]
        )

    @staticmethod
    def _annotation(table: pd.DataFrame, prefix: str, protein_id: str) -> dict:
        id_column = f"{prefix}_uniprot"
        if (
            table.empty
            or id_column not in table
            or protein_id not in set(table[id_column])
        ):
            return {}
        record = table.loc[table[id_column] == protein_id].iloc[0].to_dict()
        return {
            f"{prefix}_{key}": value
            for key, value in record.items()
            if key != id_column
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="Portable asset bundle root; overrides checkpoint, data, and graph paths.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/apexppi/apexppi_best.pt")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi")
    )
    parser.add_argument("--heterodata-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host-uniprot")
    parser.add_argument("--pathogen-uniprot", required=True)
    parser.add_argument("--top-k", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int, default=8192)
    parser.add_argument("--output-tsv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bundle_dir:
        predictor = ApexPPIPredictor.from_bundle(
            args.bundle_dir,
            device_name=args.device,
        )
    else:
        predictor = ApexPPIPredictor(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            heterodata_path=args.heterodata_path,
            device_name=args.device,
        )
    if args.host_uniprot:
        print(
            json.dumps(
                predictor.score_pair(args.host_uniprot, args.pathogen_uniprot), indent=2
            )
        )
        return
    ranking = predictor.score_pathogen_against_hosts(
        args.pathogen_uniprot,
        batch_size=args.batch_size,
    )
    if args.top_k:
        ranking = ranking.head(args.top_k)
    if args.output_tsv:
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(args.output_tsv, sep="\t", index=False)
    else:
        print(ranking.to_csv(sep="\t", index=False))


if __name__ == "__main__":
    main()
