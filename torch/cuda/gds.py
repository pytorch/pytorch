import functools
import os
from typing import Any

import torch
from torch.types import Storage


__all__: list[str] = [
    "gds_register_buffer",
    "gds_deregister_buffer",
    "GdsFile",
    "is_available",
    "save",
    "load",
]


@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """Returns ``True`` if GPUDirect Storage is available.

    GDS requires a CUDA-capable device, a PyTorch build with cuFile support
    (``USE_CUFILE=1``), and a functional cuFile driver at runtime.
    """
    return torch.cuda.is_available() and torch._C._gds_is_available()


def gds_register_buffer(s: Storage) -> None:
    """Registers a storage on a CUDA device as a cuFile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)

    Args:
        s (Storage): Buffer to register.
    """
    torch._C._gds_register_buffer(s)


def gds_deregister_buffer(s: Storage) -> None:
    """Deregisters a previously registered storage on a CUDA device as a cuFile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)
        >>> gds_deregister_buffer(s)

    Args:
        s (Storage): Buffer to deregister.
    """
    torch._C._gds_deregister_buffer(s)


class GdsFile:
    r"""Wrapper around cuFile.

    cuFile is a file-like interface to the GPUDirect Storage (GDS) API.

    See the `cuFile docs <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api>`_
    for more details.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to ``os.open`` when opening the file. ``os.O_DIRECT`` will
            be added automatically.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src1 = torch.randn(1024, device="cuda")
        >>> src2 = torch.randn(2, 1024, device="cuda")
        >>> file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
        >>> file.save_storage(src1.untyped_storage(), offset=0)
        >>> file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
        >>> dest1 = torch.empty(1024, device="cuda")
        >>> dest2 = torch.empty(2, 1024, device="cuda")
        >>> file.load_storage(dest1.untyped_storage(), offset=0)
        >>> file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        >>> torch.equal(src1, dest1)
        True
        >>> torch.equal(src2, dest2)
        True

    """

    def __init__(self, filename: str, flags: int):
        self.filename = filename
        self.flags = flags
        self.fd = os.open(filename, flags | os.O_DIRECT)  # type: ignore[attr-defined]
        self.handle: int | None = None
        self.register_handle()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Closes the GDS file, deregistering the handle and file descriptor."""
        try:
            if self.handle is not None:
                self.deregister_handle()
        finally:
            if getattr(self, "fd", -1) >= 0:
                os.close(self.fd)
                self.fd = -1

    def __enter__(self) -> "GdsFile":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def register_handle(self) -> None:
        """Registers file descriptor to cuFile driver.

        This is a wrapper around ``cuFileHandleRegister``.
        """
        if self.handle is not None:
            raise AssertionError("Cannot register a handle that is already registered.")
        self.handle = torch._C._gds_register_handle(self.fd)

    def deregister_handle(self) -> None:
        """Deregisters file descriptor from cuFile driver.

        This is a wrapper around ``cuFileHandleDeregister``.
        """
        if self.handle is None:
            raise AssertionError("Cannot deregister a handle that is not registered.")
        torch._C._gds_deregister_handle(self.handle)
        self.handle = None

    def load_storage(
        self, storage: Storage | torch.UntypedStorage, offset: int = 0
    ) -> None:
        """Loads data from the file into the storage.

        This is a wrapper around ``cuFileRead``. ``storage.nbytes()`` of data
        will be loaded from the file at ``offset`` into the storage.

        Args:
            storage (Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot load data from a file that is not registered.")
        torch._C._gds_load_storage(self.handle, storage, offset)

    def save_storage(
        self, storage: Storage | torch.UntypedStorage, offset: int = 0
    ) -> None:
        """Saves data from the storage into the file.

        This is a wrapper around ``cuFileWrite``. All bytes of the storage
        will be written to the file at ``offset``.

        Args:
            storage (Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to. (Default: 0)
        """
        if self.handle is None:
            raise AssertionError("Cannot save data to a file that is not registered.")
        torch._C._gds_save_storage(self.handle, storage, offset)


def save(
    obj: object,
    f: str | os.PathLike[str],
    *,
    pickle_protocol: int = torch.serialization.DEFAULT_PROTOCOL,
) -> None:
    """Saves an object to a file using GPUDirect Storage.

    Tensor data is written directly from GPU memory to storage via DMA,
    bypassing CPU and system memory. Metadata is serialized via
    :func:`torch.save` and the file is backward-compatible with
    :func:`torch.load`.

    This is equivalent to calling
    ``torch.save(obj, f, storage_io=StorageIO.GDS)``.

    Args:
        obj: Object to save. CUDA tensors are written via GDS; CPU tensors
            are serialized normally.
        f: Path to the file to save to. Should be on a GDS-compatible
            filesystem (e.g. ext4/XFS on NVMe) for optimal performance.
        pickle_protocol: Protocol version for pickling metadata.
    """
    from torch.serialization import _serialization_tls
    from torch.utils._pytree import tree_flatten

    f = os.fspath(f)
    leaves, _ = tree_flatten(obj)
    devices: set[torch.device] = set()
    for v in leaves:
        if isinstance(v, torch.Tensor) and v.is_cuda:
            devices.add(v.device)
    for d in devices:
        torch.cuda.current_stream(d).synchronize()

    pending_writes: list[tuple[str, torch.UntypedStorage]] = []

    def _gds_save_hook(
        zip_file: torch._C.PyTorchFileWriter,
        name: str,
        storage: torch.UntypedStorage,
        num_bytes: int,
    ) -> bool:
        if storage.device.type == "cuda":
            zip_file.write_record_metadata(name, num_bytes)  # type: ignore[attr-defined]
            pending_writes.append((name, storage))
            return True
        return False

    _serialization_tls.storage_alignment = 4096
    _serialization_tls.storage_save_hook = _gds_save_hook
    try:
        torch.save(obj, f, pickle_protocol=pickle_protocol)
    finally:
        _serialization_tls.storage_alignment = None
        _serialization_tls.storage_save_hook = None

    if pending_writes:
        reader = torch._C.PyTorchFileReader(f)  # type: ignore[attr-defined]
        try:
            with GdsFile(f, os.O_RDWR) as gds_f:
                written_offsets: set[int] = set()
                for name, storage in pending_writes:
                    offset = reader.get_record_offset(name)  # type: ignore[attr-defined]
                    if offset not in written_offsets:
                        written_offsets.add(offset)
                        gds_f.save_storage(storage, offset)
        except (OSError, RuntimeError):
            os.unlink(f)
            raise


def load(
    f: str | os.PathLike[str],
    map_location: "torch.serialization.MAP_LOCATION" = None,
    *,
    weights_only: bool = True,
    **kwargs: Any,
) -> object:
    """Loads an object from a file using GPUDirect Storage.

    Tensor data for CUDA storages is read directly from storage into GPU
    memory via DMA, bypassing CPU and system memory. Non-CUDA storages
    are loaded normally. The file should have been saved with
    :func:`torch.cuda.gds.save` or with :func:`torch.save` using
    4096-byte storage alignment.

    Args:
        f: Path to the file to load from. Should be on a GDS-compatible
            filesystem (e.g. ext4/XFS on NVMe) for optimal performance.
        map_location: Passed through to :func:`torch.load`. GDS reads
            into the current CUDA device; ``restore_location`` handles
            any remapping afterward.
        weights_only: Passed through to :func:`torch.load`. Defaults to
            ``True`` for safety.
        **kwargs: Additional keyword arguments forwarded to :func:`torch.load`.

    Returns:
        The loaded object with tensors on the specified device.
    """
    from torch.serialization import _serialization_tls

    f = os.fspath(f)
    gds_device = torch.device("cuda", torch.cuda.current_device())

    with GdsFile(f, os.O_RDONLY) as gds_reader:

        def _gds_load_hook(
            zip_file: torch._C.PyTorchFileReader,
            name: str,
            nbytes: int,
            location: str,
        ) -> torch.UntypedStorage | None:
            if not location.startswith("cuda"):
                return None
            offset = zip_file.get_record_offset(name)  # type: ignore[attr-defined]
            storage = torch.UntypedStorage(nbytes, device=gds_device)
            gds_reader.load_storage(storage, offset)  # noqa: F821
            return storage

        kwargs.pop("storage_io", None)
        _serialization_tls.storage_load_hook = _gds_load_hook
        try:
            return torch.load(
                f,
                map_location=map_location,
                weights_only=weights_only,
                **kwargs,
            )
        finally:
            _serialization_tls.storage_load_hook = None
