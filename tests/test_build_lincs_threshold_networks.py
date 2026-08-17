import json

import numpy as np
import pandas as pd

from apexppi.processing.build_lincs_threshold_networks import (
    build_threshold_edges,
    write_lincs_threshold_networks,
)


def test_build_threshold_edges_uses_ge_threshold_and_excludes_self_loops():
    similarity = np.array(
        [
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.8],
            [0.5, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    edges = build_threshold_edges(
        similarity=similarity,
        gene_ids=["G1", "G2", "G3"],
        threshold=0.7,
        source="toy",
    )

    assert edges.to_dict("records") == [
        {"gene_i": "G2", "gene_j": "G3", "similarity": float(np.float32(0.8)), "source": "toy"},
        {"gene_i": "G1", "gene_j": "G2", "similarity": float(np.float32(0.7)), "source": "toy"},
    ]


def test_write_lincs_threshold_networks_reads_dense_matrices_and_writes_summary(tmp_path):
    processed_dir = tmp_path / "lincs"
    processed_dir.mkdir()
    pd.DataFrame(
        {"gene_index": [0, 1, 2], "entrez_id": ["101", "102", "103"]}
    ).to_csv(processed_dir / "gene_ids.tsv", sep="\t", index=False)
    similarity = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.6],
            [0.1, 0.6, 1.0],
        ],
        dtype=np.float32,
    )
    for source in ["drugbank", "knockdown", "overexpression"]:
        np.save(processed_dir / f"{source}_gene_similarity.npy", similarity)

    summary = write_lincs_threshold_networks(
        processed_dir=processed_dir,
        thresholds={"drugbank": 0.8, "knockdown": 0.6, "overexpression": 0.95},
    )

    drugbank_edges = pd.read_csv(
        processed_dir / "drugbank_gene_similarity_threshold_edges.tsv", sep="\t"
    )
    assert drugbank_edges.to_dict("records") == [
        {"gene_i": 101, "gene_j": 102, "similarity": float(np.float32(0.9)), "source": "drugbank"}
    ]
    knockdown_edges = pd.read_csv(
        processed_dir / "knockdown_gene_similarity_threshold_edges.tsv", sep="\t"
    )
    assert len(knockdown_edges) == 2
    overexpression_edges = pd.read_csv(
        processed_dir / "overexpression_gene_similarity_threshold_edges.tsv", sep="\t"
    )
    assert overexpression_edges.empty
    assert summary["datasets"]["drugbank"]["threshold"] == 0.8
    assert summary["datasets"]["drugbank"]["n_edges"] == 1
    assert summary["datasets"]["knockdown"]["mean_degree"] == 4 / 3
    assert json.loads(
        (processed_dir / "threshold_network_summary.json").read_text()
    ) == summary
