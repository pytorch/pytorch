from __future__ import annotations

import os
import stat
import sys
from errno import EACCES, EISDIR
from pathlib import Path


def raise_on_not_writable_file(filename: str) -> None:
    """
    Raise an exception if attempting to open the file for writing would fail.

    This is done so files that will never be writable can be separated from files that are writable but currently
    locked.

    :param filename: file to check
    :raises OSError: as if the file was opened for writing.

    """
    try:  # use stat to do exists + can write to check without race condition
        file_stat = os.stat(filename)  # noqa: PTH116
    except OSError:
        return  # swallow does not exist or other errors

    if file_stat.st_mtime != 0:  # if os.stat returns but modification is zero that's an invalid os.stat - ignore it
        if not (file_stat.st_mode & stat.S_IWUSR):
            raise PermissionError(EACCES, "Permission denied", filename)

        if stat.S_ISDIR(file_stat.st_mode):
            if sys.platform == "win32":  # pragma: win32 cover
                # On Windows, this is PermissionError
                raise PermissionError(EACCES, "Permission denied", filename)
            else:  # pragma: win32 no cover # noqa: RET506
                # On linux / macOS, this is IsADirectoryError
                raise IsADirectoryError(EISDIR, "Is a directory", filename)


def ensure_directory_exists(filename: Path | str) -> None:
    """
    Ensure the directory containing the file exists (create it if necessary).

    :param filename: file.

    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ensure_directory_exists",
    "raise_on_not_writable_file",
]
