"""Public ApexPPI Python interface."""

from typing import TYPE_CHECKING, Any

from apexppi.models import ApexPPI

if TYPE_CHECKING:
    from apexppi.inference.predict_interaction import ApexPPIPredictor

__all__ = ["ApexPPI", "ApexPPIPredictor"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "ApexPPIPredictor":
        from apexppi.inference.predict_interaction import ApexPPIPredictor

        return ApexPPIPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
