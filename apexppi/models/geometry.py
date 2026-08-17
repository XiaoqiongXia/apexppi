"""Lorentz-manifold operations used by ApexPPI."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def clamp_tangent_norm(tangent: torch.Tensor, max_norm: float = 5.0) -> torch.Tensor:
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norm.clamp_min(1e-15), max=1.0)
    return tangent * scale


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def positive_curvature(
    raw_curvature: torch.Tensor, min_curvature: float = 1e-4
) -> torch.Tensor:
    return F.softplus(raw_curvature) + min_curvature


def expmap0(
    tangent: torch.Tensor,
    curvature: torch.Tensor,
    max_norm: float = 5.0,
) -> torch.Tensor:
    tangent = clamp_tangent_norm(tangent, max_norm=max_norm)
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(curvature).clamp_min(1e-8)
    scaled_norm = (sqrt_c * norm).clamp_max(sqrt_c * max_norm)
    direction = tangent / norm.clamp_min(1e-15)
    spatial = (torch.sinh(scaled_norm) / sqrt_c) * direction
    spatial = torch.where(norm > 0, spatial, torch.zeros_like(spatial))
    time = torch.cosh(scaled_norm) / sqrt_c
    return torch.cat([time, spatial], dim=-1)


def logmap0(point: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    spatial = point[..., 1:]
    spatial_norm = torch.linalg.norm(spatial, dim=-1, keepdim=True)
    sqrt_c = torch.sqrt(curvature).clamp_min(1e-8)
    distance = torch.acosh((sqrt_c * point[..., :1]).clamp_min(1.0)) / sqrt_c
    tangent = (distance / spatial_norm.clamp_min(1e-15)) * spatial
    return torch.where(spatial_norm > 0, tangent, torch.zeros_like(tangent))


def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(x[..., 0] * y[..., 0]) + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_distance(
    x: torch.Tensor, y: torch.Tensor, curvature: torch.Tensor
) -> torch.Tensor:
    sqrt_c = torch.sqrt(curvature).clamp_min(1e-8)
    argument = (-curvature * minkowski_dot(x, y)).clamp_min(1.0)
    return torch.acosh(argument) / sqrt_c
