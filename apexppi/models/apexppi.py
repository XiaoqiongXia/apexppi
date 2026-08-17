"""Final ApexPPI architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData

from .geometry import (
    clamp_tangent_norm,
    expmap0,
    inverse_softplus,
    logmap0,
    lorentz_distance,
    positive_curvature,
)


def edge_type_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


def aggregate_mean(
    source_features: torch.Tensor,
    edge_index: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    output = source_features.new_zeros((num_targets, source_features.shape[1]))
    if edge_index.numel() == 0:
        return output
    source, target = edge_index
    output.index_add_(0, target, source_features[source])
    counts = source_features.new_zeros((num_targets, 1))
    counts.index_add_(0, target, torch.ones_like(target, dtype=source_features.dtype)[:, None])
    return output / counts.clamp_min(1.0)


class RelationAttentionLayer(nn.Module):
    """Aggregate typed graph messages in the tangent space."""

    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        hidden_dim: int,
        dropout: float = 0.2,
        max_tangent_norm: float = 5.0,
    ):
        super().__init__()
        self.edge_types = list(edge_types)
        self.relation_transforms = nn.ModuleDict(
            {
                edge_type_key(edge_type): nn.Linear(hidden_dim, hidden_dim, bias=False)
                for edge_type in self.edge_types
            }
        )
        self.relation_logits = nn.Parameter(torch.zeros(len(self.edge_types)))
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_tangent_norm = max_tangent_norm

    def forward(
        self,
        protein_points: torch.Tensor,
        data: HeteroData,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        protein_tangent = logmap0(protein_points, curvature)
        weights = torch.softmax(self.relation_logits, dim=0)
        relation_sum = torch.zeros_like(protein_tangent)
        for weight, edge_type in zip(weights, self.edge_types):
            aggregate = aggregate_mean(
                protein_tangent,
                data[edge_type].edge_index,
                num_targets=protein_tangent.shape[0],
            )
            message = self.relation_transforms[edge_type_key(edge_type)](aggregate)
            relation_sum = relation_sum + weight * message
        delta = self.update(torch.cat([protein_tangent, relation_sum], dim=1))
        updated = protein_tangent + self.dropout(F.relu(delta))
        updated = clamp_tangent_norm(torch.tanh(updated), self.max_tangent_norm)
        return expmap0(updated, curvature, self.max_tangent_norm)

    def attention_weights(self) -> dict[str, float]:
        weights = torch.softmax(self.relation_logits.detach().cpu(), dim=0)
        return {
            edge_type[1]: float(weight)
            for edge_type, weight in zip(self.edge_types, weights)
        }


class ApexPPI(nn.Module):
    """Learnable-curvature Lorentz GNN with relation attention and gated decoder."""

    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_layers: int = 2,
        max_tangent_norm: float = 5.0,
        initial_curvature: float = 1.0,
        min_curvature: float = 1e-4,
    ):
        super().__init__()
        self.edge_types = list(edge_types)
        self.max_tangent_norm = max_tangent_norm
        self.min_curvature = min_curvature
        self.hidden_dim = hidden_dim
        raw_init = inverse_softplus(max(initial_curvature - min_curvature, min_curvature))
        self.raw_curvature = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                RelationAttentionLayer(
                    edge_types=self.edge_types,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    max_tangent_norm=max_tangent_norm,
                )
                for _ in range(num_layers)
            ]
        )

        base_feature_dim = hidden_dim * 4 + 1
        self.decoder_gate = nn.Sequential(
            nn.Linear(base_feature_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.decoder_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def curvature(self) -> torch.Tensor:
        return positive_curvature(self.raw_curvature, self.min_curvature)

    def encode(self, data: HeteroData) -> torch.Tensor:
        curvature = self.curvature()
        tangent = clamp_tangent_norm(
            torch.tanh(self.input_proj(data["protein"].x)),
            self.max_tangent_norm,
        )
        protein_points = expmap0(tangent, curvature, self.max_tangent_norm)
        for layer in self.layers:
            protein_points = layer(protein_points, data, curvature)
        return protein_points

    def build_decoder_features(
        self,
        protein_points: torch.Tensor,
        host_idx: torch.Tensor,
        pathogen_idx: torch.Tensor,
    ) -> torch.Tensor:
        curvature = self.curvature()
        host_points = protein_points[host_idx]
        pathogen_points = protein_points[pathogen_idx]
        host_tangent = logmap0(host_points, curvature)
        pathogen_tangent = logmap0(pathogen_points, curvature)
        distance = lorentz_distance(host_points, pathogen_points, curvature)
        return torch.cat(
            [
                distance.unsqueeze(1),
                host_tangent,
                pathogen_tangent,
                torch.abs(host_tangent - pathogen_tangent),
                host_tangent * pathogen_tangent,
            ],
            dim=1,
        )

    def decode(
        self,
        protein_points: torch.Tensor,
        host_idx: torch.Tensor,
        pathogen_idx: torch.Tensor,
    ) -> torch.Tensor:
        features = self.build_decoder_features(protein_points, host_idx, pathogen_idx)
        hidden_dim = self.hidden_dim
        host_tangent = features[:, 1 : 1 + hidden_dim]
        pathogen_tangent = features[:, 1 + hidden_dim : 1 + 2 * hidden_dim]
        product = features[:, 1 + 3 * hidden_dim : 1 + 4 * hidden_dim]
        gated_product = self.decoder_gate(features) * product
        mlp_logit = self.decoder(torch.cat([features, gated_product], dim=1)).squeeze(-1)
        bilinear_logit = self.decoder_bilinear(host_tangent, pathogen_tangent).squeeze(-1)
        return mlp_logit + bilinear_logit

    def forward(
        self,
        data: HeteroData,
        host_idx: torch.Tensor,
        pathogen_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode(self.encode(data), host_idx, pathogen_idx)

    def relation_attention(self) -> list[dict[str, float]]:
        return [layer.attention_weights() for layer in self.layers]
