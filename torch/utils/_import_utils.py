import functools
import importlib.util
from types import ModuleType
from typing import Optional


def _check_module_exists(name: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    try:
        spec = importlib.util.find_spec(name)
        return spec is not None
    except ImportError:
        return False


@functools.lru_cache
def dill_available() -> bool:
    return _check_module_exists("dill")


@functools.lru_cache
def import_dill() -> Optional[ModuleType]:
    if not dill_available():
        return None

    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    return dill
