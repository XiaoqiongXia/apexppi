import pandas as pd

from apexppi.processing.build_string_host_ppi_phase13 import (
    build_exact_sequence_mapping,
    build_string_host_ppi_edges,
    parse_fasta_sequences,
)


def test_parse_fasta_sequences_reads_string_ids(tmp_path):
    fasta_path = tmp_path / "seq.fa"
    fasta_path.write_text(">9606.ENSP1\nMAA\nAA\n>9606.ENSP2 description\nMCC\n")

    sequences = parse_fasta_sequences(fasta_path)

    assert sequences == {"9606.ENSP1": "MAAAA", "9606.ENSP2": "MCC"}


def test_build_exact_sequence_mapping_maps_unique_sequence_matches():
    host_nodes = pd.DataFrame(
        [
            {"host_uniprot": "H1", "sequence": "AAAA"},
            {"host_uniprot": "H2", "sequence": "CCCC"},
            {"host_uniprot": "H3", "sequence": "CCCC"},
        ]
    )
    string_sequences = {
        "9606.ENSP1": "AAAA",
        "9606.ENSP2": "CCCC",
        "9606.ENSP3": "DDDD",
    }

    mapping = build_exact_sequence_mapping(host_nodes, string_sequences)

    assert mapping == {"9606.ENSP1": "H1"}


def test_build_string_host_ppi_edges_filters_score_and_unmapped_ids(tmp_path):
    host_nodes = pd.DataFrame(
        [
            {"host_uniprot": "H1", "sequence": "AAAA"},
            {"host_uniprot": "H2", "sequence": "BBBB"},
            {"host_uniprot": "H3", "sequence": "CCCC"},
        ]
    )
    string_sequences = {
        "9606.ENSP1": "AAAA",
        "9606.ENSP2": "BBBB",
        "9606.ENSP3": "CCCC",
        "9606.ENSP4": "DDDD",
    }
    links_path = tmp_path / "links.txt"
    links_path.write_text(
        "protein1 protein2 combined_score\n"
        "9606.ENSP1 9606.ENSP2 900\n"
        "9606.ENSP1 9606.ENSP3 650\n"
        "9606.ENSP1 9606.ENSP4 950\n"
    )

    edges, summary = build_string_host_ppi_edges(
        host_nodes=host_nodes,
        string_sequences=string_sequences,
        links_path=links_path,
        min_score=700,
    )

    assert summary["mapped_string_proteins"] == 3
    assert summary["host_ppi_edges"] == 2
    assert set(zip(edges["source_id"], edges["target_id"])) == {
        ("host_protein:H1", "host_protein:H2"),
        ("host_protein:H2", "host_protein:H1"),
    }
    assert set(edges["combined_score"]) == {900}
