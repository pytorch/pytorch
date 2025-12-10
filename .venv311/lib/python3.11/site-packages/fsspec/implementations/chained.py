from typing import ClassVar

from fsspec import AbstractFileSystem

__all__ = ("ChainedFileSystem",)


class ChainedFileSystem(AbstractFileSystem):
    """Chained filesystem base class.

    A chained filesystem is designed to be layered over another FS.
    This is useful to implement things like caching.

    This base class does very little on its own, but is used as a marker
    that the class is designed for chaining.

    Right now this is only used in `url_to_fs` to provide the path argument
    (`fo`) to the chained filesystem from the underlying filesystem.

    Additional functionality may be added in the future.
    """

    protocol: ClassVar[str] = "chained"
