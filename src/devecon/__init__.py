from .build import build_deprivation_matrix, build_nested_weights
from .deprivation import deprive, deprive_in
from .inequality import atkinson, generalized_entropy, gini
from .poverty import AFResult, alkire_foster, fgt

__version__ = "0.0.1"
__all__ = [
    "AFResult",
    "alkire_foster",
    "atkinson",
    "build_deprivation_matrix",
    "build_nested_weights",
    "deprive",
    "deprive_in",
    "fgt",
    "generalized_entropy",
    "gini",
]