from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from apexppi.models import ApexPPI


@pytest.fixture
def tiny_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "apexppi-bundle-v0.1.0"
    model_dir = bundle / "models" / "apexppi"
    graph_dir = bundle / "data" / "processed" / "hpidb_human_ppi_unified_protein_graph"
    data_dir = bundle / "data" / "processed" / "hpidb_human_ppi"
    model_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    torch.manual_seed(7)
    edge_type = ("protein", "hp_ppi_observed", "protein")
    data = HeteroData()
    data["protein"].x = torch.randn(3, 4)
    data[edge_type].edge_index = torch.tensor([[0, 2], [2, 0]])
    node_maps = {
        "protein": {"protein:H1": 0, "protein:H2": 1, "protein:P1": 2},
        "host_protein": {"host_protein:H1": 0, "host_protein:H2": 1},
        "pathogen_protein": {"pathogen_protein:P1": 2},
    }
    torch.save(
        {"data": data, "node_maps": node_maps},
        graph_dir / "heterodata_unified_protein.pt",
    )

    model = ApexPPI(
        edge_types=[edge_type],
        input_dim=4,
        hidden_dim=4,
        dropout=0.0,
        num_layers=1,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "apexppi",
            "input_dim": 4,
            "hidden_dim": 4,
            "dropout": 0.0,
            "num_layers": 1,
            "edge_types": [edge_type],
            "heterodata_path": "/nonportable/original/graph.pt",
        },
        model_dir / "apexppi_best.pt",
    )

    pd.DataFrame(
        [
            {"host_uniprot": "H1", "protein_name": "Host one"},
            {"host_uniprot": "H2", "protein_name": "Host two"},
        ]
    ).to_csv(data_dir / "host_nodes.tsv", sep="\t", index=False)
    pd.DataFrame([{"pathogen_uniprot": "P1", "protein_name": "Pathogen one"}]).to_csv(
        data_dir / "pathogen_nodes.tsv", sep="\t", index=False
    )
    pd.DataFrame([{"host_uniprot": "H1", "pathogen_uniprot": "P1"}]).to_csv(
        data_dir / "positive_edges.tsv", sep="\t", index=False
    )
    return bundle
