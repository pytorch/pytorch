import sys
from typing import Callable
import warnings

_AO_MIGRATION_DEPRECATED_NAME_PREFIX = "_deprecated"

# helper function for warnings for AO migration, see
# https://github.com/pytorch/pytorch/issues/81667 for context
def _get_ao_migration_warning_str(
    module__name__,
    object_name,
) -> str:
    old_name = f"{module__name__}.{object_name}"
    new_root = module__name__.replace("torch", "torch.ao")
    new_name = f"{new_root}.{object_name}"
    s = (
        f"`{old_name}` has been moved to `{new_name}`, and the "
        f"`{old_name}` syntax will be removed in a future version "
        "of PyTorch. Please see "
        "https://github.com/pytorch/pytorch/issues/81667 for detailed "
        "migration instructions."
    )
    return s

def _import_names_with_prefix(
    cur_module__name__,
    target_module_obj,
    _deprecated_names,
) -> None:
    """
    For each name in `_deprecated_names` describing objects
    in `target_module_obj`, assigns the object to `cur_module__name__`
    with a prefix.

    This is similar to doing

      from target_module_obj import *

    except the names get a prefix.
    """
    for orig_name in _deprecated_names:
        target_obj_name = \
            f"{_AO_MIGRATION_DEPRECATED_NAME_PREFIX}_{orig_name}"
        target_obj = getattr(target_module_obj, orig_name)
        setattr(sys.modules[cur_module__name__], target_obj_name, target_obj)

def _get_module_getattr_override(
    module__name__,
    _deprecated_names,
) -> Callable:
    def __getattr__(name):
        if name in _deprecated_names:
            warnings.warn(_get_ao_migration_warning_str(module__name__, name))
            return getattr(
                sys.modules[module__name__],
                f"{_AO_MIGRATION_DEPRECATED_NAME_PREFIX}_{name}",
            )
        raise AttributeError(f"module {module__name__!r} has no attribute {name!r}")

    return __getattr__
