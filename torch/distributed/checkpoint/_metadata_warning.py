import threading
import warnings


_warned = False
_warning_lock = threading.Lock()


def install() -> None:
    from .filesystem import FileSystemReader

    original_read_metadata = FileSystemReader.read_metadata

    def read_metadata(self, *args, **kwargs):
        global _warned
        if not _warned:
            with _warning_lock:
                if not _warned:
                    warnings.warn(
                        "Distributed Checkpoint metadata is deserialized with pickle. "
                        "Only load checkpoints from trusted sources.",
                        UserWarning,
                        stacklevel=2,
                    )
                    _warned = True
        return original_read_metadata(self, *args, **kwargs)

    FileSystemReader.read_metadata = read_metadata

__all__ = ["install"]
