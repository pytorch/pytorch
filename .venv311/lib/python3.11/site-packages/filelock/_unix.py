from __future__ import annotations

import os
import sys
from contextlib import suppress
from errno import ENOSYS
from pathlib import Path
from typing import cast

from ._api import BaseFileLock
from ._util import ensure_directory_exists

#: a flag to indicate if the fcntl API is available
has_fcntl = False
if sys.platform == "win32":  # pragma: win32 cover

    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise NotImplementedError

else:  # pragma: win32 no cover
    try:
        import fcntl

        _ = (fcntl.flock, fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN)
    except (ImportError, AttributeError):
        pass
    else:
        has_fcntl = True

    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""

        def _acquire(self) -> None:
            ensure_directory_exists(self.lock_file)
            open_flags = os.O_RDWR | os.O_TRUNC
            if not Path(self.lock_file).exists():
                open_flags |= os.O_CREAT
            fd = os.open(self.lock_file, open_flags, self._context.mode)
            with suppress(PermissionError):  # This locked is not owned by this UID
                os.fchmod(fd, self._context.mode)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exception:
                os.close(fd)
                if exception.errno == ENOSYS:  # NotImplemented error
                    msg = "FileSystem does not appear to support flock; use SoftFileLock instead"
                    raise NotImplementedError(msg) from exception
            else:
                self._context.lock_file_fd = fd

        def _release(self) -> None:
            # Do not remove the lockfile:
            #   https://github.com/tox-dev/py-filelock/issues/31
            #   https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
            fd = cast("int", self._context.lock_file_fd)
            self._context.lock_file_fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


__all__ = [
    "UnixFileLock",
    "has_fcntl",
]
