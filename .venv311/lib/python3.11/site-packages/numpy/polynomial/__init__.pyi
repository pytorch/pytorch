from typing import Final, Literal

from . import chebyshev, hermite, hermite_e, laguerre, legendre, polynomial
from .chebyshev import Chebyshev
from .hermite import Hermite
from .hermite_e import HermiteE
from .laguerre import Laguerre
from .legendre import Legendre
from .polynomial import Polynomial

__all__ = [
    "set_default_printstyle",
    "polynomial", "Polynomial",
    "chebyshev", "Chebyshev",
    "legendre", "Legendre",
    "hermite", "Hermite",
    "hermite_e", "HermiteE",
    "laguerre", "Laguerre",
]

def set_default_printstyle(style: Literal["ascii", "unicode"]) -> None: ...

from numpy._pytesttester import PytestTester as _PytestTester

test: Final[_PytestTester]
