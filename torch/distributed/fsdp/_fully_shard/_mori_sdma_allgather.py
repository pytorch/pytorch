import importlib
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from ._fsdp_api import AllGather


def is_mori_fsdp_sdma_enabled() -> bool:
    raw = os.environ.get("MORI_FSDP_ENABLE_SDMA", "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


class _MoriSdmaAllGatherWork:
    def __init__(self, collective: Any, stream: torch.Stream) -> None:
        self._collective = collective
        self._stream = stream
        self._waited = False

    def wait(self) -> bool:
        if not self._waited:
            self._collective.wait_async(stream=self._stream)
            self._waited = True
        return True


class MoriSdmaAllGather(AllGather):
    def __init__(self) -> None:
        self._collective: Any | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
        self._input_buffer_size = 0
        self._output_buffer_size = 0
        self._output_buffer: torch.Tensor | None = None
        self._output_buffer_nbytes = 0
        self._registered_output_ptr: int | None = None

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        numel = _numel(size)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if (
            self._output_buffer is not None
            and self._output_buffer.dtype == dtype
            and self._output_buffer.device == device
            and self._output_buffer.numel() >= numel
        ):
            return self._output_buffer.narrow(0, 0, numel)
        self._deregister_output_buffer_if_needed()
        self._output_buffer = torch.empty(*size, dtype=dtype, device=device)
        self._output_buffer_nbytes = nbytes
        self._registered_output_ptr = None
        return self._output_buffer

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> Any | None:
        self._validate_tensors(output_tensor, input_tensor, group)
        collective = self._get_collective(group)
        stream = torch.cuda.current_stream(input_tensor.device)
        count = input_tensor.numel()
        self._ensure_output_registered(collective, output_tensor)
        if async_op:
            collective.start_async(input_tensor, output_tensor, count, stream=stream)
            return _MoriSdmaAllGatherWork(collective, stream)
        collective.enqueue(input_tensor, output_tensor, count, stream=stream)
        return None

    def _validate_tensors(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> None:
        if not input_tensor.is_cuda or not output_tensor.is_cuda:
            raise RuntimeError("MORI FSDP SDMA allgather requires CUDA tensors")
        if input_tensor.device != output_tensor.device:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires input and output on the same device"
            )
        if input_tensor.dtype != output_tensor.dtype:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires input and output dtypes to match"
            )
        expected_numel = input_tensor.numel() * group.size()
        if output_tensor.numel() != expected_numel:
            raise RuntimeError(
                "MORI FSDP SDMA allgather expected output numel "
                f"{expected_numel}, got {output_tensor.numel()}"
            )
        input_nbytes = _tensor_nbytes(input_tensor)
        output_nbytes = _tensor_nbytes(output_tensor)
        if input_nbytes % 4 != 0 or output_nbytes % 4 != 0:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires input/output byte sizes "
                "to be 4-byte aligned"
            )

    def _get_collective(self, group: dist.ProcessGroup) -> Any:
        rank, world_size = group.rank(), group.size()
        if (
            self._collective is not None
            and self._rank == rank
            and self._world_size == world_size
        ):
            return self._collective

        shmem = importlib.import_module("mori.shmem")
        AllgatherSdma = importlib.import_module("mori.ccl").AllgatherSdma

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        if my_pe != rank or npes != world_size:
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires the FSDP process group to "
                f"match SHMEM PEs, got rank/world_size={rank}/{world_size} and "
                f"my_pe/npes={my_pe}/{npes}"
            )

        self._collective = AllgatherSdma(
            my_pe,
            npes,
            input_buffer_size=4,
            output_buffer_size=4,
            copy_output_to_user=True,
        )
        self._rank = rank
        self._world_size = world_size
        self._input_buffer_size = 4
        self._output_buffer_size = 4
        self._registered_output_ptr = None
        return self._collective

    def _ensure_output_registered(self, collective: Any, output_tensor: torch.Tensor) -> None:
        ptr = output_tensor.data_ptr()
        nbytes = _tensor_nbytes(output_tensor)
        if self._registered_output_ptr == ptr and self._output_buffer_nbytes >= nbytes:
            return
        if collective.is_output_registered(output_tensor):
            self._registered_output_ptr = ptr
            return
        collective.register_output_buffer(output_tensor)
        self._registered_output_ptr = ptr

    def _deregister_output_buffer_if_needed(self) -> None:
        if self._collective is None or self._output_buffer is None:
            return
        if self._registered_output_ptr != self._output_buffer.data_ptr():
            return
        self._collective.deregister_output_buffer(self._output_buffer)
        self._registered_output_ptr = None


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _numel(size: Sequence[int | torch.SymInt]) -> int:
    numel = 1
    for dim in size:
        numel *= int(dim)
    return numel
