# Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors

import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, TYPE_CHECKING, Union

from fsspec.core import url_to_fs

from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
)


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


__all__ = [
    "FsspecWriter",
    "FsspecReader",
]


class FileSystem(FileSystemBase):
    def __init__(self) -> None:
        self.fs: Optional[AbstractFileSystem] = None

    @contextmanager
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        assert self.fs is not None
        path = os.fspath(path)

        # fsspec does not support concurrent transactions, and not all
        # AbstractFileSystem have working rollback implementations, so
        # just manually delete the file if necessary on errors.
        with self.fs.open(path, mode) as stream:
            try:
                yield stream
            except:  # noqa: B001,E722
                if "w" or "+" or "a" in mode:  # cleanup file if not read-only
                    try:
                        self.rm_file(path)
                    except:  # noqa: B001,E722
                        pass
                raise

    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        return os.path.join(path, suffix)

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        self.fs, _ = url_to_fs(path)
        return path

    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        self.fs.rename(path, new_path)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        self.fs.makedirs(path, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        if isinstance(checkpoint_id, Path):
            return False

        try:
            url_to_fs(checkpoint_id)
        except ValueError:
            return False

        return True

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        return self.fs.exists(path)

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        self.fs.rm(path)


# TODO: add the dcp.async_save mixin
class FsspecWriter(FileSystemWriter):
    """
    Basic implementation of StorageWriter using FFspec.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__(
            path,
            single_file_per_rank,
            sync_files,
            thread_count,
            per_thread_copy_ahead,
            overwrite=overwrite,
        )
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FsspecReader(FileSystemReader):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        super().__init__(path)
        self.fs = FileSystem()
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
