from __future__ import annotations

import os
import sys
from contextlib import suppress
from errno import EACCES, EEXIST
from pathlib import Path

from ._api import BaseFileLock
from ._util import ensure_directory_exists, raise_on_not_writable_file


class SoftFileLock(BaseFileLock):
    """Simply watches the existence of the lock file."""

    def _acquire(self) -> None:
        raise_on_not_writable_file(self.lock_file)
        ensure_directory_exists(self.lock_file)
        # first check for exists and read-only mode as the open will mask this case as EEXIST
        flags = (
            os.O_WRONLY  # open for writing only
            | os.O_CREAT
            | os.O_EXCL  # together with above raise EEXIST if the file specified by filename exists
            | os.O_TRUNC  # truncate the file to zero byte
        )
        try:
            file_handler = os.open(self.lock_file, flags, self._context.mode)
        except OSError as exception:  # re-raise unless expected exception
            if not (
                exception.errno == EEXIST  # lock already exist
                or (exception.errno == EACCES and sys.platform == "win32")  # has no access to this lock
            ):  # pragma: win32 no cover
                raise
        else:
            self._context.lock_file_fd = file_handler

    def _release(self) -> None:
        assert self._context.lock_file_fd is not None  # noqa: S101
        os.close(self._context.lock_file_fd)  # the lock file is definitely not None
        self._context.lock_file_fd = None
        with suppress(OSError):  # the file is already deleted and that's what we want
            Path(self.lock_file).unlink()


__all__ = [
    "SoftFileLock",
]
