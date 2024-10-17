from types import ModuleType
from typing import Any

from .._mangling import is_mangled


def is_from_package(obj: Any) -> bool:
    """
    Return whether an object was loaded from a package.

    Note: packaged objects from externed modules will return ``False``.
    """
    if type(obj) == ModuleType:
        return is_mangled(obj.__name__)
    else:
        return is_mangled(type(obj).__module__)
