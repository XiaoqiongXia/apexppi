import pandas as pd
import torch

from apexppi.evaluation.evaluate_ranking import (
    compute_query_ranking,
    evaluate_grouped_rankings,
    ndcg_at_k,
)


def test_compute_query_ranking_reports_positive_ranks_and_topk_metrics():
    host_ids = ["H1", "H2", "H3", "H4"]
    scores = torch.tensor([0.2, 0.9, 0.8, 0.1])

    row = compute_query_ranking(
        pathogen_uniprot="P1",
        host_ids=host_ids,
        scores=scores,
        positive_hosts={"H1", "H3"},
        ks=(1, 2, 3),
    )

    assert row["n_positives"] == 2
    assert row["best_rank"] == 2
    assert row["mean_rank"] == 2.5
    assert row["mrr"] == 0.5
    assert row["hits_at_1"] == 0.0
    assert row["hits_at_2"] == 1.0
    assert row["recall_at_2"] == 0.5


def test_ndcg_at_k_handles_binary_relevance():
    ranked_relevance = [0, 1, 1, 0]

    score = ndcg_at_k(ranked_relevance, k=3)

    assert round(score, 4) == 0.6934


def test_evaluate_grouped_rankings_calls_scorer_once_per_pathogen():
    positives = pd.DataFrame(
        [
            {"pathogen_uniprot": "P1", "host_uniprot": "H2"},
            {"pathogen_uniprot": "P1", "host_uniprot": "H3"},
            {"pathogen_uniprot": "P2", "host_uniprot": "H1"},
        ]
    )
    host_ids = ["H1", "H2", "H3"]
    calls = []

    def scorer(pathogen_uniprot: str) -> torch.Tensor:
        calls.append(pathogen_uniprot)
        if pathogen_uniprot == "P1":
            return torch.tensor([0.1, 0.8, 0.7])
        return torch.tensor([0.9, 0.2, 0.1])

    per_query, summary = evaluate_grouped_rankings(
        positives=positives, host_ids=host_ids, score_pathogen=scorer, ks=(1, 2)
    )

    assert calls == ["P1", "P2"]
    assert per_query.shape[0] == 2
    assert summary["queries"] == 2
    assert summary["mean_recall_at_1"] == 0.75
