import threading
import warnings


_warned = False
_warning_lock = threading.Lock()
_install_lock = threading.Lock()


def install() -> None:
    from .filesystem import FileSystemReader

    if getattr(FileSystemReader, "_native_neo_metadata_warning_installed", False):
        return
    with _install_lock:
        if getattr(FileSystemReader, "_native_neo_metadata_warning_installed", False):
            return

        original_read_metadata = FileSystemReader.read_metadata

        def read_metadata(self, *args, **kwargs):
            global _warned
            result = original_read_metadata(self, *args, **kwargs)
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
            return result

        FileSystemReader.read_metadata = read_metadata
        FileSystemReader._native_neo_metadata_warning_installed = True


__all__ = ["install"]
