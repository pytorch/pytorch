"""
A platform independent file lock that supports the with-statement.

.. autodata:: filelock.__version__
   :no-value:

"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

from ._api import AcquireReturnProxy, BaseFileLock
from ._error import Timeout
from ._soft import SoftFileLock
from ._unix import UnixFileLock, has_fcntl
from ._windows import WindowsFileLock
from .asyncio import (
    AsyncAcquireReturnProxy,
    AsyncSoftFileLock,
    AsyncUnixFileLock,
    AsyncWindowsFileLock,
    BaseAsyncFileLock,
)
from .version import version

#: version of the project as a string
__version__: str = version


if sys.platform == "win32":  # pragma: win32 cover
    _FileLock: type[BaseFileLock] = WindowsFileLock
    _AsyncFileLock: type[BaseAsyncFileLock] = AsyncWindowsFileLock
else:  # pragma: win32 no cover # noqa: PLR5501
    if has_fcntl:
        _FileLock: type[BaseFileLock] = UnixFileLock
        _AsyncFileLock: type[BaseAsyncFileLock] = AsyncUnixFileLock
    else:
        _FileLock = SoftFileLock
        _AsyncFileLock = AsyncSoftFileLock
        if warnings is not None:
            warnings.warn("only soft file lock is available", stacklevel=2)

if TYPE_CHECKING:
    FileLock = SoftFileLock
    AsyncFileLock = AsyncSoftFileLock
else:
    #: Alias for the lock, which should be used for the current platform.
    FileLock = _FileLock
    AsyncFileLock = _AsyncFileLock


__all__ = [
    "AcquireReturnProxy",
    "AsyncAcquireReturnProxy",
    "AsyncFileLock",
    "AsyncSoftFileLock",
    "AsyncUnixFileLock",
    "AsyncWindowsFileLock",
    "BaseAsyncFileLock",
    "BaseFileLock",
    "FileLock",
    "SoftFileLock",
    "Timeout",
    "UnixFileLock",
    "WindowsFileLock",
    "__version__",
]
