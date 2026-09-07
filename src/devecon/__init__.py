from .poverty import fgt, alkire_foster, AFResult
from .deprivation import deprive, deprive_in
from .build import build_nested_weights, build_deprivation_matrix

__version__ = "0.0.1"
__all__ = [
    "fgt", "alkire_foster", "AFResult",
    "deprive", "deprive_in",
    "build_nested_weights", "build_deprivation_matrix",
]