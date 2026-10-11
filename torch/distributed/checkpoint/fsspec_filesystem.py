# Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors

import concurrent.futures
import io
import itertools
import os
import sys
import warnings
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch import Tensor
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint._extension import StreamTransformExtension
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
    SerializationFormat,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.futures import Future
from torch.serialization import _load, _open_zipfile_reader


if TYPE_CHECKING:
    from fsspec import AbstractFileSystem


__all__ = [
    "FsspecWriter",
    "FsspecReader",
]


class _FileSystem(FileSystemBase):
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
        from fsspec.core import url_to_fs

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
            from fsspec.core import url_to_fs
        except ImportError:
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
        cache_staged_state_dict: bool = False,
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
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increased memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and reused for multiple dcp.async_save calls. Default to False.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.
            _extensions: Extensions to apply to output streams (EXPERIMENTAL)

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__(
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            cache_staged_state_dict=cache_staged_state_dict,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        self.fs = _FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return _FileSystem.validate_checkpoint_id(checkpoint_id)


def _destinations_disjoint(targets: list[Tensor]) -> bool:
    """Whether every destination is a contiguous plain CPU tensor with disjoint bytes.

    Overlap means two items would race, which happens when items narrow into
    overlapping regions of the same tensor. Non-contiguous, non-CPU, or
    custom tensor subclasses are rejected so ``[data_ptr, data_ptr + nbytes)``
    is always an exact host memory extent.
    """
    spans = []
    for t in targets:
        try:
            if not (type(t) is Tensor and t.is_cpu and t.is_contiguous()):
                return False
            start = t.data_ptr()
            nbytes = t.numel() * t.element_size()
            if start == 0 and nbytes > 0:
                return False
            spans.append((start, start + nbytes))
        except Exception:
            return False
    spans.sort()
    return all(end <= nxt for (_, end), (nxt, _) in itertools.pairwise(spans))


def _load_aliased(buf: bytes | bytearray | memoryview) -> Tensor:
    """Deserialize one DCP tensor record, aliasing ``buf`` if the byte order matches.

    A foreign byte order is swapped in place, so that case still gets a copy.
    """
    with _open_zipfile_reader(io.BytesIO(buf)) as zf:
        storage = None
        if (
            zf.has_record("byteorder")
            and zf.get_record("byteorder") == sys.byteorder.encode()
        ):
            storage = torch.frombuffer(buf, dtype=torch.uint8).untyped_storage()
        return _load(
            zf,
            "cpu",
            _weights_only_unpickler,
            overall_storage=storage,
            weights_only=True,
            encoding="utf-8",
        )


class FsspecReader(FileSystemReader):
    def __init__(
        self,
        path: str | os.PathLike,
        *,
        max_batch_size: int = 1024,
        max_batch_bytes: int = 256 * 1024 * 1024,
        cpu_workers: int | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the FsspecReader pointing to `path`.

        Args:
            path: directory or URL where the checkpoint will be read from.
            max_batch_size: Maximum number of read items per batched cat_ranges call.
                Defaults to 1024, so batches are normally bounded by bytes. Small
                items, such as per-parameter optimizer steps spread over every
                rank's file, then share one call instead of paying for opening
                their files in each of several calls.
            max_batch_bytes: Maximum cumulative byte size requested per batched
                cat_ranges call. Defaults to 256 MiB. This caps one request, not
                resident memory: the next two batches are fetched while the current
                one is still being decoded and copied, so expect a small multiple of
                this to be live at peak.
            cpu_workers: Number of worker threads for parallel CPU deserialization.
                Defaults to min(4, max(1, cpu_count // local_world_size)).
            **kwargs: Additional storage options passed to fsspec url_to_fs.
        """
        super().__init__(path)
        self.max_batch_size = max(1, max_batch_size)
        self.max_batch_bytes = max(1, max_batch_bytes)
        if cpu_workers is None:
            local_world_size = max(1, int(os.environ.get("LOCAL_WORLD_SIZE") or 1))
            total_cpus = os.cpu_count() or 4
            cpu_workers = min(4, max(1, total_cpus // local_world_size))
        self.cpu_workers = max(1, cpu_workers)
        self.fs = _FileSystem()
        self.path = self.fs.init_path(path, **kwargs)

    def _supports_batched_cat_ranges(self) -> bool:
        import fsspec
        import fsspec.asyn
        from fsspec.implementations.cached import CachingFileSystem

        # read_data runs two cat_ranges calls at once, which only an
        # AsyncFileSystem is guaranteed to handle. Wrappers such as
        # DirFileSystem are async even over a sync or caching filesystem, so
        # the innermost filesystem decides.
        curr_fs = self.fs.fs
        if not isinstance(curr_fs, fsspec.asyn.AsyncFileSystem):
            return False
        seen: set[int] = set()
        while True:
            if isinstance(curr_fs, CachingFileSystem):
                return False
            seen.add(id(curr_fs))
            inner = getattr(curr_fs, "fs", None)
            if not isinstance(inner, fsspec.AbstractFileSystem):
                break
            if id(inner) in seen:
                return False
            curr_fs = inner
        return bool(getattr(curr_fs, "async_impl", True))

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        if not plan.items or not self._supports_batched_cat_ranges():
            return super().read_data(plan, planner)

        reqs = sorted(
            plan.items,
            key=lambda req: (
                self.storage_data[req.storage_index].relative_path,
                self.storage_data[req.storage_index].offset,
            ),
        )

        batches = []
        batch = []
        batch_bytes = 0
        for req in reqs:
            length = self.storage_data[req.storage_index].length
            if batch and (
                len(batch) >= self.max_batch_size
                or batch_bytes + length > self.max_batch_bytes
            ):
                batches.append(batch)
                batch = []
                batch_bytes = 0
            batch.append(req)
            batch_bytes += length
        if batch:
            batches.append(batch)

        def fetch_batch(b_reqs):
            mds = [self.storage_data[req.storage_index] for req in b_reqs]
            paths = [self.fs.concat_path(self.path, md.relative_path) for md in mds]
            starts = [md.offset for md in mds]
            ends = [md.offset + md.length for md in mds]
            chunks = self.fs.fs.cat_ranges(paths, starts, ends, on_error="raise")
            # A short list means some ranges were dropped (``on_error="omit"``).
            # Left unchecked, the zip below would silently skip those items and
            # leave their tensors at whatever the caller initialized them to.
            if len(chunks) != len(b_reqs):
                raise RuntimeError(
                    f"cat_ranges returned {len(chunks)} chunks for {len(b_reqs)} ranges"
                )
            # ``on_error`` is advisory: fsspec honors it only since 2026.6.0 and
            # other backends may ignore it, returning exceptions in-band.
            for path, start, end, chunk in zip(paths, starts, ends, chunks):
                if isinstance(chunk, BaseException):
                    raise RuntimeError(
                        f"Failed to read bytes [{start}, {end}) from {path}"
                    ) from chunk
                if len(chunk) != end - start:
                    raise RuntimeError(
                        f"Read {len(chunk)} bytes for [{start}, {end}) from {path}"
                    )
            return chunks

        def decode(req, chunk):
            if self.storage_data[req.storage_index].transform_descriptors:
                return self._decode_item(req, io.BytesIO(chunk))
            if req.type == LoadItemType.BYTE_IO:
                return io.BytesIO(chunk)
            tensor = _load_aliased(chunk)
            return narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)

        inference_mode = torch.is_inference_mode_enabled()

        def _copy(dst: Tensor, src: Tensor) -> None:
            with torch.inference_mode(inference_mode):
                dst.copy_(src)

        # Resolving a whole batch before committing any of it is only safe
        # when resolve_tensor and commit_tensor keep their default behavior.
        planner_cls = type(planner)
        parallel_copy = (
            isinstance(planner, DefaultLoadPlanner)
            and planner_cls.resolve_tensor is DefaultLoadPlanner.resolve_tensor
            and planner_cls.commit_tensor is DefaultLoadPlanner.commit_tensor
        )

        with (
            warnings.catch_warnings(),
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.cpu_workers, thread_name_prefix="FsspecReader-cpu"
            ) as cpu_executor,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="FsspecReader-io"
            ) as io_executor,
        ):
            # _load_aliased wraps the read-only fetched bytes without copying.
            # The filter is process-wide so that it also covers the pool threads.
            warnings.filterwarnings(
                "ignore",
                message=".*The given buffer is not writable.*",
                category=UserWarning,
            )
            # Two fetches run at once, so one batch's slowest streams overlap
            # the next batch's start instead of idling the connection.
            inflight = [io_executor.submit(fetch_batch, b) for b in batches[:2]]
            try:
                for idx, b_reqs in enumerate(batches):
                    chunks = inflight.pop(0).result()
                    if idx + 2 < len(batches):
                        inflight.append(
                            io_executor.submit(fetch_batch, batches[idx + 2])
                        )

                    decoded = [
                        cpu_executor.submit(decode, req, chunk)
                        for req, chunk in zip(b_reqs, chunks)
                    ]
                    del chunks

                    # Every planner hook runs on this thread, so planners need
                    # not be thread safe. Only decoding above and the copies
                    # below go to the pool.
                    if parallel_copy:
                        pending: list[tuple[ReadItem, Tensor, Tensor]] = []
                        for i, req in enumerate(b_reqs):
                            f = decoded[i]
                            decoded[i] = None
                            item = f.result()
                            if req.type == LoadItemType.BYTE_IO:
                                planner.load_bytes(req, item)
                            else:
                                pending.append(
                                    (req, self._resolve_item(req, item, planner), item)
                                )

                        if len(pending) > 1 and _destinations_disjoint(
                            [dst for _, dst, _ in pending]
                        ):
                            copies = [
                                cpu_executor.submit(_copy, dst, src)
                                for _, dst, src in pending
                            ]
                            for c in copies:
                                c.result()
                            for req, dst, _ in pending:
                                planner.commit_tensor(req, dst)
                        else:
                            for req, dst, src in pending:
                                _copy(dst, src)
                                planner.commit_tensor(req, dst)
                    else:
                        for i, req in enumerate(b_reqs):
                            f = decoded[i]
                            decoded[i] = None
                            item = f.result()
                            if req.type == LoadItemType.BYTE_IO:
                                planner.load_bytes(req, item)
                            else:
                                dst = self._resolve_item(req, item, planner)
                                _copy(dst, item)
                                planner.commit_tensor(req, dst)
            finally:
                for fut in inflight:
                    fut.cancel()
                cpu_executor.shutdown(wait=False, cancel_futures=True)
                io_executor.shutdown(wait=False, cancel_futures=True)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return _FileSystem.validate_checkpoint_id(checkpoint_id)
