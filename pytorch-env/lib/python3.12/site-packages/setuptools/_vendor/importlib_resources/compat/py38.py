import os
import sys

from typing import Union


if sys.version_info >= (3, 9):
    StrPath = Union[str, os.PathLike[str]]
else:
    # PathLike is only subscriptable at runtime in 3.9+
    StrPath = Union[str, "os.PathLike[str]"]
