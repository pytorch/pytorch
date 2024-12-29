"""Read resources contained within a package."""

from ._common import (
    as_file,
    files,
    Package,
    Anchor,
)

from .functional import (
    contents,
    is_resource,
    open_binary,
    open_text,
    path,
    read_binary,
    read_text,
)

from .abc import ResourceReader


__all__ = [
    'Package',
    'Anchor',
    'ResourceReader',
    'as_file',
    'files',
    'contents',
    'is_resource',
    'open_binary',
    'open_text',
    'path',
    'read_binary',
    'read_text',
]
