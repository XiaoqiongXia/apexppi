"""Inference helpers for trained ApexPPI models."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .predict_interaction import ApexPPIPredictor

__all__ = ["ApexPPIPredictor"]


def __getattr__(name: str) -> Any:
    if name == "ApexPPIPredictor":
        from .predict_interaction import ApexPPIPredictor

        return ApexPPIPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
