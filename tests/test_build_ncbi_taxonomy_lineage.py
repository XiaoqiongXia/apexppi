import pandas as pd

from apexppi.processing.build_ncbi_taxonomy_lineage import (
    build_taxonomy_lineage_tables,
    extract_numeric_taxids,
    parse_names_dmp,
    parse_nodes_dmp,
)


def test_parse_ncbi_taxdump_nodes_and_names(tmp_path):
    nodes = tmp_path / "nodes.dmp"
    names = tmp_path / "names.dmp"
    nodes.write_text(
        "1\t|\t1\t|\tno rank\t|\n"
        "2\t|\t1\t|\tsuperkingdom\t|\n"
        "10\t|\t2\t|\tfamily\t|\n"
        "11\t|\t10\t|\tspecies\t|\n"
    )
    names.write_text(
        "1\t|\troot\t|\t\t|\tscientific name\t|\n"
        "2\t|\tViruses\t|\t\t|\tscientific name\t|\n"
        "10\t|\tExampleviridae\t|\t\t|\tscientific name\t|\n"
        "11\t|\tExample virus\t|\t\t|\tscientific name\t|\n"
    )

    assert parse_nodes_dmp(nodes)["11"] == {"parent_taxid": "10", "rank": "species"}
    assert parse_names_dmp(names)["10"] == "Exampleviridae"


def test_extract_numeric_taxids_from_hpidb_taxid_set():
    value = "taxid:11(Example virus)|taxid:22(other|Other bacterium)"

    assert extract_numeric_taxids(value) == ["11", "22"]


def test_build_taxonomy_lineage_tables_writes_observed_taxa_and_ancestors(tmp_path):
    taxonomy_dir = tmp_path / "taxonomy"
    output_dir = tmp_path / "out"
    taxonomy_dir.mkdir()
    (taxonomy_dir / "nodes.dmp").write_text(
        "1\t|\t1\t|\tno rank\t|\n"
        "2\t|\t1\t|\tsuperkingdom\t|\n"
        "10\t|\t2\t|\tfamily\t|\n"
        "11\t|\t10\t|\tspecies\t|\n"
    )
    (taxonomy_dir / "names.dmp").write_text(
        "1\t|\troot\t|\t\t|\tscientific name\t|\n"
        "2\t|\tViruses\t|\t\t|\tscientific name\t|\n"
        "10\t|\tExampleviridae\t|\t\t|\tscientific name\t|\n"
        "11\t|\tExample virus\t|\t\t|\tscientific name\t|\n"
    )
    pathogen_nodes = pd.DataFrame(
        [
            {
                "pathogen_uniprot": "P1",
                "pathogen_taxid_set": "taxid:11(Example virus)",
            }
        ]
    )

    summary = build_taxonomy_lineage_tables(
        pathogen_nodes=pathogen_nodes,
        taxonomy_dir=taxonomy_dir,
        output_dir=output_dir,
    )

    nodes = pd.read_csv(output_dir / "nodes_taxon_lineage.tsv", sep="\t")
    parent_edges = pd.read_csv(output_dir / "edges_taxon_child_parent.tsv", sep="\t")
    protein_edges = pd.read_csv(
        output_dir / "edges_pathogen_protein_belongs_to_taxon_lineage.tsv", sep="\t"
    )

    assert summary["observed_taxa"] == 1
    assert set(nodes["taxon_id"]) == {"taxid:1", "taxid:2", "taxid:10", "taxid:11"}
    assert ("taxid:11", "taxid:10") in set(zip(parent_edges["child_taxon_id"], parent_edges["parent_taxon_id"]))
    assert protein_edges.iloc[0]["source_id"] == "pathogen_protein:P1"
