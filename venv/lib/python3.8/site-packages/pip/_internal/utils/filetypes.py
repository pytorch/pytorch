"""Filetype information.
"""
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from typing import Tuple

WHEEL_EXTENSION = '.whl'
BZ2_EXTENSIONS = ('.tar.bz2', '.tbz')  # type: Tuple[str, ...]
XZ_EXTENSIONS = ('.tar.xz', '.txz', '.tlz',
                 '.tar.lz', '.tar.lzma')  # type: Tuple[str, ...]
ZIP_EXTENSIONS = ('.zip', WHEEL_EXTENSION)  # type: Tuple[str, ...]
TAR_EXTENSIONS = ('.tar.gz', '.tgz', '.tar')  # type: Tuple[str, ...]
ARCHIVE_EXTENSIONS = (
    ZIP_EXTENSIONS + BZ2_EXTENSIONS + TAR_EXTENSIONS + XZ_EXTENSIONS
)
