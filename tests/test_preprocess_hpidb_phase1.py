import pandas as pd

from apexppi.processing.preprocess_hpidb_phase1 import (
    extract_uniprot_accession,
    run_preprocessing,
)


def test_extract_uniprot_accession_prefers_last_uniprot_and_strips_isoform():
    field = "intact:EBI-1|uniprotkb:P12345-2|refseq:XP_1|uniprotkb:Q99999"

    assert extract_uniprot_accession(field) == "Q99999"


def test_phase1_preprocessing_deduplicates_edges_preserves_metadata_and_splits(tmp_path):
    mitab = tmp_path / "mini.mitab.tsv"
    fasta = tmp_path / "mini.fasta"
    output_dir = tmp_path / "out"

    mitab.write_text(
        "\t".join(
            [
                "# protein_xref_1",
                "protein_xref_2",
                "alternative_identifiers_1",
                "alternative_identifiers_2",
                "protein_alias_1",
                "protein_alias_2",
                "detection_method",
                "author_name",
                "pmid",
                "protein_taxid_1",
                "protein_taxid_2",
                "interaction_type",
                "source_database_id",
                "database_identifier",
                "confidence",
            ]
        )
        + "\n"
        + "\n".join(
            [
                "uniprotkb:H1\tuniprotkb:P1\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:1\ttaxid:9606(Homo sapiens)\ttaxid:1(Pathogen one)\ttype:x\tdb:A\tdb:edge1\tscore:0.1",
                "uniprotkb:H1\tuniprotkb:P1\t-\t-\t-\t-\tmethod:b\tAuthor B\tpubmed:2\ttaxid:9606(human|Homo sapiens)\ttaxid:1(Pathogen one)\ttype:x\tdb:B\tdb:edge2\tscore:0.2",
                "uniprotkb:H1\tuniprotkb:P2\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:3\ttaxid:9606(Homo sapiens)\ttaxid:2(Pathogen two)\ttype:y\tdb:A\tdb:edge3\tscore:0.3",
                "uniprotkb:H2\tuniprotkb:P1\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:4\ttaxid:9606(Homo sapiens)\ttaxid:1(Pathogen one)\ttype:y\tdb:A\tdb:edge4\tscore:0.4",
                "uniprotkb:H2\tuniprotkb:P3\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:5\ttaxid:9606(Homo sapiens)\ttaxid:3(Pathogen three)\ttype:y\tdb:A\tdb:edge5\tscore:0.5",
                "uniprotkb:H3\tuniprotkb:P2\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:6\ttaxid:9606(Homo sapiens)\ttaxid:2(Pathogen two)\ttype:y\tdb:A\tdb:edge6\tscore:0.6",
                "uniprotkb:H3\tuniprotkb:P3\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:7\ttaxid:9606(Homo sapiens)\ttaxid:3(Pathogen three)\ttype:y\tdb:A\tdb:edge7\tscore:0.7",
                "uniprotkb:H4\tuniprotkb:P4\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:8\ttaxid:9606(Homo sapiens)\ttaxid:4(Pathogen four)\ttype:y\tdb:A\tdb:edge8\tscore:0.8",
                "uniprotkb:H3\tuniprotkb:P3\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:6\ttaxid:10090(Mus musculus)\ttaxid:3(Pathogen three)\ttype:y\tdb:A\tdb:edge6\tscore:0.6",
                "uniprotkb:H4\trefseq:BAD\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:7\ttaxid:9606(Homo sapiens)\ttaxid:4(Pathogen four)\ttype:y\tdb:A\tdb:edge7\tscore:0.7",
                "uniprotkb:H5\tuniprotkb:P5\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:8\ttaxid:9606(Homo sapiens)\ttaxid:5(Pathogen five)\ttype:y\tdb:A\tdb:edge8\tscore:0.8",
            ]
        )
        + "\n"
    )
    fasta.write_text(
        ">sp|H1|H1_HUMAN\nAAAA\n"
        ">sp|H2|H2_HUMAN\nAAAA\n"
        ">sp|H3|H3_HUMAN\nAAAA\n"
        ">sp|H4|H4_HUMAN\nAAAA\n"
        ">sp|P1|P1_PATH\nAAAA\n"
        ">sp|P2|P2_PATH\nAAAA\n"
        ">sp|P3|P3_PATH\nAAAA\n"
        ">sp|P4|P4_PATH\nAAAA\n"
    )

    summary = run_preprocessing(
        mitab_path=mitab,
        fasta_path=fasta,
        output_dir=output_dir,
        seed=7,
        train_frac=0.5,
        valid_frac=0.25,
        test_frac=0.25,
        negatives_per_positive=1,
    )

    assert summary["homo_sapiens_rows"] == 10
    assert summary["both_uniprot_rows"] == 9
    assert summary["fasta_backed_rows"] == 8
    assert summary["unique_positive_edges"] == 7

    positive_edges = pd.read_csv(output_dir / "positive_edges.tsv", sep="\t")
    assert set(positive_edges.columns) >= {
        "host_uniprot",
        "pathogen_uniprot",
        "evidence_count",
        "pmid_set",
        "source_database_set",
    }
    h1p1 = positive_edges.query(
        "host_uniprot == 'H1' and pathogen_uniprot == 'P1'"
    ).iloc[0]
    assert h1p1["evidence_count"] == 2
    assert h1p1["pmid_set"] == "pubmed:1|pubmed:2"
    assert h1p1["source_database_set"] == "db:A|db:B"

    splits = {
        split: pd.read_csv(output_dir / f"{split}_edges.tsv", sep="\t")
        for split in ["train", "valid", "test"]
    }
    assert sum(len(frame) for frame in splits.values()) == 14
    assert all(
        set(frame["label"]) == {0, 1}
        for frame in splits.values()
        if len(frame) > 0
    )

    all_positive = set(zip(positive_edges["host_uniprot"], positive_edges["pathogen_uniprot"]))
    for frame in splits.values():
        negatives = frame[frame["label"] == 0]
        assert not any(
            (row.host_uniprot, row.pathogen_uniprot) in all_positive
            for row in negatives.itertuples()
        )

    train_pos = splits["train"][splits["train"]["label"] == 1]
    train_hosts = set(train_pos["host_uniprot"])
    train_pathogens = set(train_pos["pathogen_uniprot"])
    for split in ["valid", "test"]:
        split_pos = splits[split][splits[split]["label"] == 1]
        assert set(split_pos["host_uniprot"]).issubset(train_hosts)
        assert set(split_pos["pathogen_uniprot"]).issubset(train_pathogens)


def test_phase1b_writes_host_pathogen_and_viral_node_annotation_tables(tmp_path):
    mitab = tmp_path / "mini.mitab.tsv"
    fasta = tmp_path / "mini.fasta"
    output_dir = tmp_path / "out"

    header = [
        "# protein_xref_1",
        "protein_xref_2",
        "alternative_identifiers_1",
        "alternative_identifiers_2",
        "protein_alias_1",
        "protein_alias_2",
        "detection_method",
        "author_name",
        "pmid",
        "protein_taxid_1",
        "protein_taxid_2",
        "interaction_type",
        "source_database_id",
        "database_identifier",
        "confidence",
    ]
    rows = [
        "uniprotkb:H1\tuniprotkb:V1\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:1\ttaxid:9606(Homo sapiens)\ttaxid:11111(Example virus strain A)\ttype:x\tdb:A\tdb:edge1\tscore:0.1",
        "uniprotkb:H2\tuniprotkb:B1\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:2\ttaxid:9606(Homo sapiens)\ttaxid:22222(Example bacterium)\ttype:x\tdb:A\tdb:edge2\tscore:0.1",
        "uniprotkb:H1\tuniprotkb:B1\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:3\ttaxid:9606(Homo sapiens)\ttaxid:22222(Example bacterium)\ttype:x\tdb:A\tdb:edge3\tscore:0.1",
        "uniprotkb:H2\tuniprotkb:HI1\t-\t-\t-\t-\tmethod:a\tAuthor A\tpubmed:4\ttaxid:9606(Homo sapiens)\ttaxid:727(haeif|Haemophilus influenzae)\ttype:x\tdb:A\tdb:edge4\tscore:0.1",
    ]
    mitab.write_text("\t".join(header) + "\n" + "\n".join(rows) + "\n")
    fasta.write_text(
        ">sp|H1|H1_HUMAN Host protein 1 OS=Homo sapiens OX=9606\nAAAA\n"
        ">sp|H2|H2_HUMAN Host protein 2 OS=Homo sapiens OX=9606\nAAAAA\n"
        ">sp|V1|V1_VIRUS Viral capsid protein OS=Example virus strain A OX=11111\nVVVV\n"
        ">sp|B1|B1_BACT Bacterial protein OS=Example bacterium OX=22222\nBBBB\n"
        ">sp|HI1|HI1_HAEIF Haemophilus protein OS=Haemophilus influenzae OX=727\nHHHH\n"
    )

    run_preprocessing(
        mitab_path=mitab,
        fasta_path=fasta,
        output_dir=output_dir,
        seed=3,
        train_frac=1.0,
        valid_frac=0.0,
        test_frac=0.0,
        negatives_per_positive=0,
    )

    host_nodes = pd.read_csv(output_dir / "host_nodes.tsv", sep="\t")
    pathogen_nodes = pd.read_csv(output_dir / "pathogen_nodes.tsv", sep="\t")
    viral_nodes = pd.read_csv(output_dir / "viral_protein_nodes.tsv", sep="\t")

    assert set(host_nodes["host_uniprot"]) == {"H1", "H2"}
    assert set(pathogen_nodes["pathogen_uniprot"]) == {"V1", "B1", "HI1"}
    assert set(viral_nodes["viral_uniprot"]) == {"V1"}

    v1 = viral_nodes.iloc[0]
    assert v1["viral_taxid_set"] == "taxid:11111(Example virus strain A)"
    assert v1["viral_name_set"] == "Example virus strain A"
    assert v1["protein_name"] == "Viral capsid protein"
    assert v1["sequence_length"] == 4
    assert v1["n_positive_edges"] == 1

    b1 = pathogen_nodes.query("pathogen_uniprot == 'B1'").iloc[0]
    assert b1["is_viral_candidate"] == 0
    assert b1["n_positive_edges"] == 2

    hi1 = pathogen_nodes.query("pathogen_uniprot == 'HI1'").iloc[0]
    assert hi1["pathogen_name_set"] == "Haemophilus influenzae"
    assert hi1["is_viral_candidate"] == 0
