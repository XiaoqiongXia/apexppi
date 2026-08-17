import json

import numpy as np
import pandas as pd

from apexppi.processing.preprocess_lincs_gene_similarity import run_preprocessing


def _write_lincs_matrix(path, rows, genes):
    frame = pd.DataFrame(rows, columns=["perturbagen", *genes])
    frame.to_csv(path, sep="\t", index=False)


def test_lincs_gene_similarity_preprocessing_outputs_dense_and_topk_edges(tmp_path):
    input_dir = tmp_path / "LINCS"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    genes = ["101", "102", "103", "104"]
    _write_lincs_matrix(
        input_dir / "consensi-drugbank.tsv",
        [
            ["DB1", 1.0, 2.0, -1.0, 100.0],
            ["DB2", 2.0, 4.0, -2.0, -100.0],
            ["DB3", 3.0, 6.0, -3.0, 0.0],
            ["DB4", 4.0, 8.0, -4.0, 1.0],
        ],
        genes,
    )
    _write_lincs_matrix(
        input_dir / "consensi-knockdown.tsv",
        [
            ["2", 1.0, 1.0, 4.0, 4.0],
            ["9", 2.0, 2.0, 3.0, 3.0],
            ["16", 3.0, 3.0, 2.0, 2.0],
            ["18", 4.0, 4.0, 1.0, 1.0],
        ],
        genes,
    )
    _write_lincs_matrix(
        input_dir / "consensi-overexpression.tsv",
        [
            ["2", 1.0, 0.0, 0.0, -1.0],
            ["9", 0.0, 1.0, 0.0, -1.0],
            ["18", 0.0, 0.0, 1.0, -1.0],
            ["25", 1.0, 1.0, 1.0, -3.0],
        ],
        genes,
    )

    summary = run_preprocessing(
        input_dir=input_dir,
        output_dir=output_dir,
        clip_min=-10.0,
        clip_max=10.0,
        top_k=1,
    )

    gene_ids = pd.read_csv(output_dir / "gene_ids.tsv", sep="\t")
    assert gene_ids.to_dict("records") == [
        {"gene_index": 0, "entrez_id": 101},
        {"gene_index": 1, "entrez_id": 102},
        {"gene_index": 2, "entrez_id": 103},
        {"gene_index": 3, "entrez_id": 104},
    ]

    dense = np.load(output_dir / "drugbank_gene_similarity.npy")
    assert dense.dtype == np.float32
    assert dense.shape == (4, 4)
    assert np.allclose(dense, dense.T)
    assert np.allclose(np.diag(dense), 1.0)
    assert dense[0, 1] > 0.99
    assert dense[0, 2] < -0.99

    edges = pd.read_csv(
        output_dir / "drugbank_gene_similarity_top1_positive_edges.tsv",
        sep="\t",
    )
    assert list(edges.columns) == [
        "gene_i",
        "gene_j",
        "similarity",
        "rank_i",
        "rank_j",
        "is_mutual",
        "source",
    ]
    assert not any(edges["gene_i"] == edges["gene_j"])
    assert set(edges["source"]) == {"drugbank"}
    assert (edges["similarity"] > 0).all()
    edge_pairs = set(zip(edges["gene_i"].astype(str), edges["gene_j"].astype(str)))
    assert ("101", "102") in edge_pairs
    mutual_edge = edges.query("gene_i == 101 and gene_j == 102").iloc[0]
    assert mutual_edge["rank_i"] == 1
    assert mutual_edge["rank_j"] == 1
    assert bool(mutual_edge["is_mutual"])

    summary_path = json.loads((output_dir / "processing_summary.json").read_text())
    assert summary == summary_path
    assert summary["similarity_metric"] == "pearson"
    assert summary["clip_range"] == [-10.0, 10.0]
    assert summary["top_k"] == 1
    assert summary["datasets"]["drugbank"]["n_perturbagens"] == 4
    assert summary["datasets"]["drugbank"]["value_range_before_clip"] == [-100.0, 100.0]
    assert summary["datasets"]["drugbank"]["value_range_after_clip"] == [-10.0, 10.0]


def test_lincs_gene_similarity_rejects_mismatched_gene_columns(tmp_path):
    input_dir = tmp_path / "LINCS"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    _write_lincs_matrix(
        input_dir / "consensi-drugbank.tsv",
        [["DB1", 1.0, 2.0], ["DB2", 2.0, 3.0]],
        ["101", "102"],
    )
    _write_lincs_matrix(
        input_dir / "consensi-knockdown.tsv",
        [["2", 1.0, 2.0], ["9", 2.0, 3.0]],
        ["101", "103"],
    )
    _write_lincs_matrix(
        input_dir / "consensi-overexpression.tsv",
        [["2", 1.0, 2.0], ["9", 2.0, 3.0]],
        ["101", "102"],
    )

    try:
        run_preprocessing(input_dir=input_dir, output_dir=output_dir)
    except ValueError as exc:
        assert "Gene columns do not match" in str(exc)
    else:
        raise AssertionError("Expected mismatched gene columns to be rejected")
