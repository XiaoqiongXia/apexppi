import json
import subprocess
import sys

from apexppi import ApexPPIPredictor


def test_predictor_loads_portable_bundle_and_scores(tiny_bundle):
    predictor = ApexPPIPredictor.from_bundle(tiny_bundle)
    result = predictor.score_pair("H1", "P1")
    assert result["known_hpidb_positive"] is True
    assert 0.0 <= result["interaction_probability"] <= 1.0
    assert result["host_protein_name"] == "Host one"


def test_empty_candidate_list_returns_empty_ranking(tiny_bundle):
    predictor = ApexPPIPredictor.from_bundle(tiny_bundle)
    ranking = predictor.score_pathogen_against_hosts("P1", [])
    assert ranking.empty
    assert "interaction_probability" in ranking.columns


def test_cli_scores_pair_from_bundle(tiny_bundle):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "apexppi.inference.predict_interaction",
            "--bundle-dir",
            str(tiny_bundle),
            "--host-uniprot",
            "H1",
            "--pathogen-uniprot",
            "P1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["host_uniprot"] == "H1"
    assert result["pathogen_uniprot"] == "P1"


def test_cli_rejects_nonpositive_top_k(tiny_bundle):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "apexppi.inference.predict_interaction",
            "--bundle-dir",
            str(tiny_bundle),
            "--pathogen-uniprot",
            "P1",
            "--top-k",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "must be at least 1" in completed.stderr
