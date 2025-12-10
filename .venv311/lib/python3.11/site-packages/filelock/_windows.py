from __future__ import annotations

import os
import sys
from contextlib import suppress
from errno import EACCES
from pathlib import Path
from typing import cast

from ._api import BaseFileLock
from ._util import ensure_directory_exists, raise_on_not_writable_file

if sys.platform == "win32":  # pragma: win32 cover
    import msvcrt

    class WindowsFileLock(BaseFileLock):
        """Uses the :func:`msvcrt.locking` function to hard lock the lock file on Windows systems."""

        def _acquire(self) -> None:
            raise_on_not_writable_file(self.lock_file)
            ensure_directory_exists(self.lock_file)
            flags = (
                os.O_RDWR  # open for read and write
                | os.O_CREAT  # create file if not exists
                | os.O_TRUNC  # truncate file if not empty
            )
            try:
                fd = os.open(self.lock_file, flags, self._context.mode)
            except OSError as exception:
                if exception.errno != EACCES:  # has no access to this lock
                    raise
            else:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError as exception:
                    os.close(fd)  # close file first
                    if exception.errno != EACCES:  # file is already locked
                        raise
                else:
                    self._context.lock_file_fd = fd

        def _release(self) -> None:
            fd = cast("int", self._context.lock_file_fd)
            self._context.lock_file_fd = None
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            os.close(fd)

            with suppress(OSError):  # Probably another instance of the application hat acquired the file lock.
                Path(self.lock_file).unlink()

else:  # pragma: win32 no cover

    class WindowsFileLock(BaseFileLock):
        """Uses the :func:`msvcrt.locking` function to hard lock the lock file on Windows systems."""

        def _acquire(self) -> None:
            raise NotImplementedError

        def _release(self) -> None:
            raise NotImplementedError


__all__ = [
    "WindowsFileLock",
]
