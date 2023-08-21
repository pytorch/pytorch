from typing import List, Callable
import importlib
import warnings


_MESSAGE_TEMPLATE = r"Usage of '{old_location}' is deprecated; please use '{new_location}' instead."

def lazy_deprecated_import(all: List[str], old_module: str, new_module: str) -> Callable:
    r"""Import utility to lazily import deprecated packages / modules / functional.

    The old_module and new_module are also used in the deprecation warning defined
    by the `_MESSAGE_TEMPLATE`.

    Args:
        all: The list of the functions that are imported. Generally, the module's
            __all__ list of the module.
        old_module: Old module location
        new_module: New module location / Migrated location

    Returns:
        Callable to assign to the `__getattr__`

    Usage:

        # In the `torch/nn/quantized/functional.py`
        from torch.nn.utils._deprecation_utils import lazy_deprecated_import
        _MIGRATED_TO = "torch.ao.nn.quantized.functional"
        __getattr__ = lazy_deprecated_import(
            all=__all__,
            old_module=__name__,
            new_module=_MIGRATED_TO)
    """
    warning_message = _MESSAGE_TEMPLATE.format(
        old_location=old_module,
        new_location=new_module)

    def getattr_dunder(name):
        if name in all:
            # We are using the "RuntimeWarning" to make sure it is not
            # ignored by default.
            warnings.warn(warning_message, RuntimeWarning, stacklevel=TO_BE_DETERMINED)
            package = importlib.import_module(new_module)
            return getattr(package, name)
        raise AttributeError(f"Module {new_module!r} has no attribute {name!r}.")
    return getattr_dunder
