import pandas as pd
import torch

from apexppi.processing.build_similarity_network_phase8 import (
    build_symmetric_knn_edges,
    write_similarity_networks,
)


def test_build_symmetric_knn_edges_excludes_self_and_writes_both_directions():
    node_ids = ["protein:A", "protein:B", "protein:C"]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    edges = build_symmetric_knn_edges(
        node_ids=node_ids,
        embeddings=embeddings,
        edge_type="similar_to",
        source_type="protein",
        target_type="protein",
        k=1,
    )

    assert not (edges["source_id"] == edges["target_id"]).any()
    assert ("protein:A", "protein:B") in set(zip(edges["source_id"], edges["target_id"]))
    assert ("protein:B", "protein:A") in set(zip(edges["source_id"], edges["target_id"]))
    assert set(edges["edge_type"]) == {"similar_to"}
    assert "similarity_score" in edges.columns
    assert "rank" in edges.columns


def test_build_symmetric_knn_edges_filters_by_min_similarity():
    node_ids = ["protein:A", "protein:B", "protein:C"]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    edges = build_symmetric_knn_edges(
        node_ids=node_ids,
        embeddings=embeddings,
        edge_type="similar_to",
        source_type="protein",
        target_type="protein",
        k=2,
        min_similarity=0.8,
    )

    pairs = set(zip(edges["source_id"], edges["target_id"]))
    assert pairs == {("protein:A", "protein:B"), ("protein:B", "protein:A")}
    assert (edges["similarity_score"] > 0.8).all()


def test_write_similarity_networks_uses_host_and_pathogen_embeddings(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    data_dir.mkdir()
    pd.DataFrame(
        [{"host_uniprot": "H1"}, {"host_uniprot": "H2"}, {"host_uniprot": "H3"}]
    ).to_csv(data_dir / "host_nodes.tsv", sep="\t", index=False)
    pd.DataFrame(
        [{"pathogen_uniprot": "P1"}, {"pathogen_uniprot": "P2"}, {"pathogen_uniprot": "P3"}]
    ).to_csv(data_dir / "pathogen_nodes.tsv", sep="\t", index=False)
    embedding_path = tmp_path / "embeddings.pt"
    torch.save(
        {
            "protein_ids": ["H1", "H2", "H3", "P1", "P2", "P3"],
            "embeddings": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.9, 0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        embedding_path,
    )

    summary = write_similarity_networks(
        data_dir=data_dir,
        embedding_path=embedding_path,
        output_dir=output_dir,
        k=1,
        min_similarity=0.0,
    )

    assert summary["host_similarity_edges"] > 0
    assert summary["pathogen_similarity_edges"] > 0
    assert summary["min_similarity"] == 0.0
    assert (output_dir / "edges_host_protein_similar_to_host_protein.tsv").exists()
    assert (output_dir / "edges_pathogen_protein_similar_to_pathogen_protein.tsv").exists()
