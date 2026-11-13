import warnings

from .fsspec_filesystem import _FileSystem, FsspecReader, FsspecWriter


__all__ = ["FsspecWriter", "FsspecReader"]

# Keep old name for backward compatibility
FileSystem = _FileSystem

warnings.warn(
    "torch.distributed.checkpoint._fsspec_filesystem is deprecated and will be removed in a future release. "
    "Please use torch.distributed.checkpoint.fsspec_filesystem instead.",
    DeprecationWarning,
    stacklevel=2,
)
