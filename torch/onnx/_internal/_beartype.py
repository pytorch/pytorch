"""An internal wrapper for the beartype library.

The module returns a no-op decorator when the beartype library is not installed.
"""
from torch.onnx._globals import GLOBALS


def _no_op_decorator(func):
    return func


if not GLOBALS.runtime_type_check:
    beartype = _no_op_decorator
else:
    try:
        import warnings

        from beartype import beartype, roar as _roar  # type: ignore[import,no-redef]

        warnings.filterwarnings(
            "ignore",
            category=_roar.BeartypeDecorHintPep585DeprecationWarning,
        )
    except ImportError:
        beartype = _no_op_decorator
