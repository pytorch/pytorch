__all__ = ["Sequence"]

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
