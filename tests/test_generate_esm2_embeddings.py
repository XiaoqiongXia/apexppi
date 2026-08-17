import pandas as pd
import torch

from apexppi.embeddings.generate_esm2_embeddings import (
    build_embedding_records,
    mean_pool_residue_representations,
    project_cache_paths,
)


def test_project_cache_paths_stay_inside_project(tmp_path):
    paths = project_cache_paths(tmp_path, "esm2_t12_35M_UR50D")

    assert paths["model_dir"] == tmp_path / "models" / "esm2_t12_35M_UR50D"
    assert paths["torch_home"] == tmp_path / "models" / "torch"
    assert paths["output_dir"] == tmp_path / "data" / "processed" / "embeddings"


def test_build_embedding_records_deduplicates_host_and_pathogen_nodes(tmp_path):
    input_dir = tmp_path / "hpidb_human_ppi"
    input_dir.mkdir()
    pd.DataFrame(
        [
            {"host_uniprot": "H1", "sequence": "AAAA", "sequence_length": 4},
            {"host_uniprot": "SHARED", "sequence": "CCCC", "sequence_length": 4},
        ]
    ).to_csv(input_dir / "host_nodes.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {"pathogen_uniprot": "P1", "sequence": "VVVV", "sequence_length": 4},
            {"pathogen_uniprot": "SHARED", "sequence": "CCCC", "sequence_length": 4},
        ]
    ).to_csv(input_dir / "pathogen_nodes.tsv", sep="\t", index=False)

    records = build_embedding_records(input_dir, max_residues=3)

    assert [record["protein_id"] for record in records] == ["H1", "P1", "SHARED"]
    shared = [record for record in records if record["protein_id"] == "SHARED"][0]
    assert shared["node_types"] == "host|pathogen"
    assert shared["original_sequence_length"] == 4
    assert shared["sequence_length"] == 3
    assert shared["was_truncated"] == 1


def test_mean_pool_residue_representations_excludes_special_tokens():
    token_representations = torch.tensor(
        [
            [
                [100.0, 100.0],
                [1.0, 3.0],
                [5.0, 7.0],
                [200.0, 200.0],
            ]
        ]
    )

    pooled = mean_pool_residue_representations(token_representations, [2])

    assert torch.equal(pooled, torch.tensor([[3.0, 5.0]]))
