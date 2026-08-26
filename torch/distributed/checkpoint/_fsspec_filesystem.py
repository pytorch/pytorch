# Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors

import concurrent.futures
import io
import os
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from torch.futures import Future
from fsspec.core import url_to_fs

from torch.distributed.checkpoint._extension import StreamTransformExtension
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
    SerializationFormat,
)
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


__all__ = [
    "FsspecWriter",
    "FsspecReader",
]


class FileSystem(FileSystemBase):
    def __init__(self) -> None:
        self.fs: AbstractFileSystem | None = None

    @contextmanager
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]:
        if self.fs is None:
            raise AssertionError("fs should not be None")
        path = os.fspath(path)

        # fsspec does not support concurrent transactions, and not all
        # AbstractFileSystem have working rollback implementations, so
        # just manually delete the file if necessary on errors.
        with self.fs.open(path, mode) as stream:
            try:
                yield stream
            except:
                if any(ch in mode for ch in "w+a"):  # cleanup file if not read-only
                    try:
                        self.rm_file(path)
                    except:  # noqa: E722
                        pass
                raise

    def concat_path(self, path: str | os.PathLike, suffix: str) -> str | os.PathLike:
        return os.path.join(path, suffix)

    def init_path(self, path: str | os.PathLike, **kwargs) -> str | os.PathLike:
        # Disable fsspec internal caching by default to avoid redundant memory copies and high RAM usage during batched range reads.
        kwargs.setdefault("cache_type", "none")
        self.fs, _ = url_to_fs(path, **kwargs)
        return path

    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None:
        self.fs.rename(path, new_path)

    def mkdir(self, path: str | os.PathLike) -> None:
        self.fs.makedirs(path, exist_ok=True)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        if isinstance(checkpoint_id, Path):
            return False

        try:
            url_to_fs(checkpoint_id)
        except ValueError:
            return False

        return True

    def exists(self, path: str | os.PathLike) -> bool:
        return self.fs.exists(path)

    def rm_file(self, path: str | os.PathLike) -> None:
        self.fs.rm(path)

    def ls(self, path: str | os.PathLike) -> list[str]:
        # setting detail to False explicitly to keep the list[str] return type,
        # instead of the list[Dict] return type when detail=True
        return self.fs.ls(path, detail=False)


# TODO: add the dcp.async_save mixin
class FsspecWriter(FileSystemWriter):
    """
    Basic implementation of StorageWriter using fsspec.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: str | os.PathLike,
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
        **kwargs,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving them. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__(
            path,
            single_file_per_rank,
            sync_files,
            thread_count,
            per_thread_copy_ahead,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)


class FsspecReader(FileSystemReader):
    def __init__(
        self,
        path: str | os.PathLike,
        max_batch_size: int = 64,
        cpu_workers: int | None = None,
        io_workers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(path)
        self.max_batch_size = max(1, max_batch_size)
        self.cpu_workers = max(
            1, cpu_workers if cpu_workers is not None else min(16, os.cpu_count() or 4)
        )
        self.io_workers = max(1, io_workers)
        self.fs = FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        reqs = plan.items
        if not reqs:
            fut: Future[None] = Future()
            fut.set_result(None)
            return fut

        # If the underlying fsspec filesystem supports cat_ranges, use batched range reading
        if self.fs and self.fs.fs and hasattr(self.fs.fs, "cat_ranges"):
            batches = []
            for i in range(0, len(reqs), self.max_batch_size):
                batch = reqs[i : i + self.max_batch_size]
                paths = []
                starts = []
                ends = []
                for req in batch:
                    item_md = self.storage_data[req.storage_index]
                    paths.append(self.fs.concat_path(self.path, item_md.relative_path))
                    starts.append(item_md.offset)
                    ends.append(item_md.offset + item_md.length)
                batches.append((paths, starts, ends, batch))

            def fetch_batch(b):
                bp, bs, be, br = b
                chunks = self.fs.fs.cat_ranges(bp, bs, be, on_error="raise")
                return chunks, br

            def process_chunk(req, chunk_data):
                self._load_item(req, io.BytesIO(chunk_data), planner)

            with (
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.cpu_workers
                ) as cpu_executor,
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.io_workers
                ) as io_executor,
            ):
                try:
                    next_io = io_executor.submit(fetch_batch, batches[0])

                    for idx, batch in enumerate(batches):
                        chunks, b_reqs = next_io.result()

                        if idx + 1 < len(batches):
                            next_io = io_executor.submit(fetch_batch, batches[idx + 1])

                        futures = [
                            cpu_executor.submit(process_chunk, req, chunk_data)
                            for req, chunk_data in zip(b_reqs, chunks)
                        ]
                        for f in futures:
                            f.result()

                        del chunks
                        del b_reqs
                finally:
                    if sys.version_info >= (3, 9):
                        cpu_executor.shutdown(wait=True, cancel_futures=True)
                        io_executor.shutdown(wait=True, cancel_futures=True)
                    else:
                        cpu_executor.shutdown(wait=True)
                        io_executor.shutdown(wait=True)

            fut: Future[None] = Future()
            fut.set_result(None)
            return fut

        return super().read_data(plan, planner)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
