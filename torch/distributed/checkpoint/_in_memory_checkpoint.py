import logging
import pickle
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Callable, cast, Optional, TypeVar, Union

import torch
from torch.distributed import ProcessGroup, Work
from torch.distributed.tensor import _DTensorSpec, DTensor
from torch.utils._pytree import (
    KeyPath,
    tree_flatten_with_path,
    tree_unflatten,
    TreeSpec,
)


logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _TensorMeta:
    """
    This is the metadata for a tensor that is used to transfer checkpoints.
    It contains the shape, the dtype, the storage offset and the stride of the
    tensor.

    This must be pickleable so that it can be sent over the wire.
    """

    shape: torch.Size
    dtype: torch.dtype
    storage_offset: int
    stride: tuple[int, ...]
    nbytes: int


@dataclass
class _DTensorMeta:
    """
    This is the metadata for a DTensor that is used to transfer checkpoints.
    It contains the metadata for the local tensor and the spec of the DTensor.

    This must be pickleable so that it can be sent over the wire.
    """

    local: _TensorMeta
    spec: _DTensorSpec


@dataclass
class _StateDictMeta:
    """
    This is the metadata for a state dict that is used to transfer checkpoints.
    It contains the step, the pytree spec of the state dict and the metadata for
    each tensor in the state dict.

    This must be pickleable so that it can be sent over the wire.

    Args:
        step: the step of the checkpoint to verify consistency
        treespec: the pytree spec of the state dict
        paths: the path of each leaf in the state dict
        non_tensor_leaves: the metadata for each tensor in the state dict and any
            non-tensor leaves in the state dict
    """

    step: int
    treespec: TreeSpec
    paths: list[KeyPath]
    non_tensor_leaves: list[Union[object, _TensorMeta, _DTensorMeta]]


@contextmanager
def _timeit(name: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    yield
    dur = time.perf_counter() - start
    logger.info(f"{name} took {dur}s")


def _prepare_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, _TensorMeta]:
    return (
        _cast_tensor(tensor, torch.uint8),
        _TensorMeta(
            shape=tensor.shape,
            dtype=tensor.dtype,
            storage_offset=cast(int, tensor.storage_offset()),
            stride=tensor.stride(),
            nbytes=tensor.untyped_storage().nbytes(),
        ),
    )


def _prepare_state_dict(
    state_dict: object,
    step: int,
    device: torch.device,
) -> tuple[_StateDictMeta, list[torch.Tensor]]:
    leaves: list[tuple[KeyPath, object]]
    leaves, treespec = tree_flatten_with_path(state_dict)

    paths: list[KeyPath] = []
    non_tensor_leaves: list[Union[object, _TensorMeta, _DTensorMeta]] = []
    tensors: list[torch.Tensor] = []
    for key_path, v in leaves:
        paths.append(key_path)

        if isinstance(v, DTensor):
            tensor, tensor_meta = _prepare_tensor(v._local_tensor)

            tensors.append(tensor)

            non_tensor_leaves.append(
                _DTensorMeta(
                    local=tensor_meta,
                    spec=v._spec,
                )
            )
        elif isinstance(v, torch.Tensor):
            tensor, tensor_meta = _prepare_tensor(v)
            tensors.append(tensor)
            non_tensor_leaves.append(tensor_meta)
        else:
            non_tensor_leaves.append(v)

    return (
        _StateDictMeta(
            step=step,
            treespec=treespec,
            paths=paths,
            non_tensor_leaves=non_tensor_leaves,
        ),
        tensors,
    )


def _cast_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Casts the underlying storage to a tensor of the given dtype.

    The returned tensor will be of size ``storage.nbytes``.

    This works for all datatypes and supports strided/offset tensors with the
    caveat that the cast tensor may be larger than the original tensor due to
    the differences in striding.
    """
    assert type(tensor) is torch.Tensor, (
        f"can only cast standard tensors not {type(tensor)}"
    )
    storage = tensor.untyped_storage()
    ret = torch.tensor(storage, dtype=dtype, device=tensor.device)
    assert ret.untyped_storage() is storage, "storage should be the same"
    return ret


class InMemoryStorage:
    """
    In memory storage for checkpointing. This is a simple in memory storage which
    uses shared memory to store the checkpoint. 1 process keeps this alive and other
    processes can access it. TODO: can also save the memory to temporary files and load it via mmap
    """

    def __init__(self):
        self._storage = {}

    def save(self, key: str, data: object) -> None:
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)
        shm = shared_memory.SharedMemory(create=True, size=data_size)
        shm.buf[:data_size] = serialized_data
        # Store the shared memory name and size
        self._storage[key] = (shm.name, data_size)

    def load(self, key: str) -> object:
        if key not in self._storage:
            raise KeyError(f"No data found for key: {key}")
        # Retrieve shared memory name and size
        shm_name, data_size = self._storage[key]
        # Access the shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        serialized_data = bytes(existing_shm.buf[:data_size])
        data = pickle.loads(serialized_data)
        # Close the shared memory
        existing_shm.close()
        return data

    def clear(self) -> None:
        # Unlink all shared memory objects
        for shm_name, _ in self._storage.values():
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            existing_shm.unlink()
        self._storage.clear()


class PGTransport:
    """
    This is a checkpoint transport that uses the process group to transfer checkpoints.
    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.

    Args:
        pg: the process group to use for communication
        timeout: the timeout for communication
        device: the device to use for tensors
        state_dict: if specified this function will be called to do an inplace
            receive into the returned state_dict. This is much faster than
            having to allocate new tensors and transferring them to the CPU.
    """

    def __init__(
        self,
        pg: ProcessGroup,
        timeout: timedelta,
        device: torch.device,
        state_dict: Optional[Callable[[], object]] = None,
    ) -> None:
        self._work: list[Work] = []
        self._pg = pg
        self._timeout = timeout
        self._device = device
        self._state_dict = state_dict

    def metadata(self) -> str:
        return "<n/a>"

    def disallow_checkpoint(self) -> None:
        pass

    def send_checkpoint(
        self, dst_ranks: list[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        with _timeit("preparing state_dict"):
            meta, tensors = _prepare_state_dict(state_dict, step, device=self._device)

        work = []

        with _timeit("send pickle"):
            buf = pickle.dumps(meta)
            len_t = torch.tensor([len(buf)], dtype=torch.int64, device=self._device)
            buf_t = torch.frombuffer(buf, dtype=torch.uint8).to(self._device)
            for dst_rank in dst_ranks:
                work.append(self._pg.send([len_t], dst_rank, tag=1))
                work.append(self._pg.send([buf_t], dst_rank, tag=2))

        with _timeit("send tensors"):
            for i, t in enumerate(tensors):
                original_device = t.device
                t = t.to(self._device)
                for dst_rank in dst_ranks:
                    work.append(self._pg.send([t], dst_rank, tag=3 + i))

                # if we did a copy we should wait for the work to complete so we
                # can free the memory to avoid OOMs
                if original_device == torch.device("cpu"):
                    for w in work:
                        w.wait(timeout)
                    work = []

            for w in work:
                w.wait(timeout)

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        state_dict = self._state_dict() if self._state_dict else {}
        state_dict_leaves, _ = tree_flatten_with_path(state_dict)

        dst_tensors: dict[KeyPath, object] = dict(state_dict_leaves)

        len_t = torch.zeros(1, dtype=torch.int64, device=self._device)
        self._pg.recv([len_t], src_rank, tag=1).wait(timeout)
        length = cast(int, len_t.item())

        assert length > 0, f"invalid metadata length {length=}"

        buf = torch.empty(length, dtype=torch.uint8, device=self._device)
        self._pg.recv([buf], src_rank, tag=2).wait(timeout)

        meta: _StateDictMeta = pickle.loads(buf.cpu().numpy().tobytes())
        assert meta.step == step

        i: int = 0
        works: list[Work] = []

        def recv(path: KeyPath, v: _TensorMeta) -> torch.Tensor:
            nonlocal i

            inplace = dst_tensors.get(path)
            if (
                isinstance(inplace, torch.Tensor)
                and inplace.device.type == self._device.type
            ):
                if isinstance(inplace, DTensor):
                    inplace = inplace._local_tensor
                t = _cast_tensor(inplace, torch.uint8)
                assert t.nbytes == v.nbytes, (
                    "inplace tensor storage must be the same size"
                )
            else:
                t = torch.empty(v.nbytes, dtype=torch.uint8, device=self._device)

            work = self._pg.recv([t], src_rank, tag=3 + i)
            i += 1

            if inplace is None:
                # if not inplace we need to copy it to CPU to avoid OOMing
                work.wait(timeout)
                t = t.cpu()
            else:
                works.append(work)

            return torch.as_strided(
                t.view(v.dtype),
                size=v.shape,
                stride=v.stride,
                storage_offset=v.storage_offset,
            )

        values = []
        for path, v in zip(meta.paths, meta.non_tensor_leaves):
            if isinstance(v, _TensorMeta):
                values.append(recv(path, v))
            elif isinstance(v, _DTensorMeta):
                tensor = recv(path, v.local)
                # pyre-fixme[29]: DTensor is not a function
                values.append(DTensor(tensor, v.spec, requires_grad=False))
            else:
                values.append(v)

        for work in works:
            work.wait(timeout)

        return tree_unflatten(values, meta.treespec)
