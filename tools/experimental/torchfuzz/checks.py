"""Check abstractions for different execution modes and validations.

Concrete :class:`Check` subclasses are owned by device plugins (see
``torchfuzz/cuda/_checks.py`` for the CUDA reference implementation).
"""

from abc import ABC, abstractmethod


class Check(ABC):
    """Base class for execution checks."""

    @abstractmethod
    def codegen(self, args_tuple: str) -> list[str]:
        """Generate code lines for this check."""
