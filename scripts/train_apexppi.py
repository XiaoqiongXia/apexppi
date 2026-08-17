#!/usr/bin/env python3
"""Train the final ApexPPI model."""

import argparse
import json
from pathlib import Path

from apexppi.models.training import train_apexppi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/hpidb_human_ppi"))
    parser.add_argument(
        "--heterodata-path",
        type=Path,
        default=Path(
            "data/processed/hpidb_human_ppi_unified_protein_graph/heterodata_unified_protein.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/apexppi"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-negatives", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--p2h-infonce-weight", type=float, default=0.25)
    parser.add_argument("--h2p-infonce-weight", type=float, default=0.25)
    parser.add_argument("--max-tangent-norm", type=float, default=5.0)
    parser.add_argument("--initial-curvature", type=float, default=1.0)
    parser.add_argument("--min-curvature", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_apexppi(
        data_dir=args.data_dir,
        heterodata_path=args.heterodata_path,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_negatives=args.num_negatives,
        temperature=args.temperature,
        p2h_infonce_weight=args.p2h_infonce_weight,
        h2p_infonce_weight=args.h2p_infonce_weight,
        max_tangent_norm=args.max_tangent_norm,
        initial_curvature=args.initial_curvature,
        min_curvature=args.min_curvature,
        grad_clip_norm=args.grad_clip_norm,
        patience=args.patience,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
