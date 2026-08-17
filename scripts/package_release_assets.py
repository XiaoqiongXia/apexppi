#!/usr/bin/env python3
"""Build a versioned, checksummed ApexPPI inference bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from apexppi import ApexPPIPredictor


ANNOTATION_FILES = ("host_nodes.tsv", "pathogen_nodes.tsv", "positive_edges.tsv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def package_release_assets(
    checkpoint: Path,
    graph: Path,
    data_dir: Path,
    output_dir: Path,
    metrics: Path | None = None,
    version: str = "0.1.0",
) -> tuple[Path, Path]:
    checkpoint = require_file(checkpoint, "Checkpoint")
    graph = require_file(graph, "Processed graph")
    data_dir = data_dir.expanduser().resolve()
    annotation_paths = [
        require_file(data_dir / filename, filename) for filename in ANNOTATION_FILES
    ]
    metrics = require_file(metrics, "Metrics") if metrics is not None else None
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_name = f"apexppi-bundle-v{version}"
    archive_path = output_dir / f"{bundle_name}.tar.gz"
    archive_checksum_path = output_dir / f"{archive_path.name}.sha256"

    with tempfile.TemporaryDirectory(prefix="apexppi-bundle-") as temporary_dir:
        bundle_root = Path(temporary_dir) / bundle_name
        model_dir = bundle_root / "models" / "apexppi"
        graph_dir = (
            bundle_root / "data" / "processed" / "hpidb_human_ppi_unified_protein_graph"
        )
        annotation_dir = bundle_root / "data" / "processed" / "hpidb_human_ppi"
        model_dir.mkdir(parents=True)
        graph_dir.mkdir(parents=True)
        annotation_dir.mkdir(parents=True)

        shutil.copy2(checkpoint, model_dir / "apexppi_best.pt")
        shutil.copy2(graph, graph_dir / "heterodata_unified_protein.pt")
        for source in annotation_paths:
            shutil.copy2(source, annotation_dir / source.name)
        if metrics is not None:
            shutil.copy2(metrics, model_dir / "model_metrics.json")

        predictor = ApexPPIPredictor.from_bundle(bundle_root)
        manifest = {
            "bundle_version": version,
            "model_type": predictor.checkpoint["model_type"],
            "host_candidates": len(predictor.host_to_idx),
            "pathogen_candidates": len(predictor.pathogen_to_idx),
            "known_positive_pairs": len(predictor.known_positive_pairs),
        }
        (bundle_root / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (bundle_root / "README.txt").write_text(
            "ApexPPI inference asset bundle. Verify checksums.sha256 before use.\n"
            "Load with ApexPPIPredictor.from_bundle(<this directory>).\n",
            encoding="utf-8",
        )

        bundled_files = sorted(
            path for path in bundle_root.rglob("*") if path.is_file()
        )
        checksum_lines = [
            f"{sha256(path)}  {path.relative_to(bundle_root).as_posix()}"
            for path in bundled_files
        ]
        (bundle_root / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="utf-8",
        )

        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(bundle_root, arcname=bundle_name)

    archive_checksum_path.write_text(
        f"{sha256(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return archive_path, archive_checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--version", default="0.1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path, checksum_path = package_release_assets(
        checkpoint=args.checkpoint,
        graph=args.graph,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metrics=args.metrics,
        version=args.version,
    )
    print(
        json.dumps(
            {"archive": str(archive_path), "checksum": str(checksum_path)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
