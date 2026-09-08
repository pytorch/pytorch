import warnings


_warned = False


def install() -> None:
    from .filesystem import FileSystemReader

    original_read_metadata = FileSystemReader.read_metadata

    def read_metadata(self, *args, **kwargs):
        global _warned
        if not _warned:
            _warned = True
            warnings.warn(
                "Distributed Checkpoint metadata is deserialized with pickle. "
                "Only load checkpoints from trusted sources.",
                UserWarning,
                stacklevel=2,
            )
        return original_read_metadata(self, *args, **kwargs)

    FileSystemReader.read_metadata = read_metadata


__all__ = ["install"]
