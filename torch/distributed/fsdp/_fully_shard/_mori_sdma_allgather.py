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


def is_mori_fsdp_zero_copy_output_enabled() -> bool:
    raw = os.environ.get("MORI_FSDP_ZERO_COPY_OUTPUT", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


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
        self._zero_copy_output = is_mori_fsdp_zero_copy_output_enabled()
        self.supports_no_copy = self._zero_copy_output
        self.supports_param_contiguous_output = self._zero_copy_output
        self._collective: Any | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
        self._input_buffer_size = 0
        self._output_buffer_size = 0
        self._output_buffer: torch.Tensor | None = None
        self._output_buffer_nbytes = 0
        self._registered_output_ptr: int | None = None
        self._param_contiguous_split_sizes: torch.Tensor | None = None
        self._param_contiguous_split_offsets: torch.Tensor | None = None

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
        if self._can_call_param_contiguous(input_tensor):
            split_sizes = self._param_contiguous_split_sizes
            split_offsets = self._param_contiguous_split_offsets
            if split_sizes is None or split_offsets is None:
                raise RuntimeError(
                    "MORI param-contiguous allgather metadata is not initialized"
                )
            if async_op:
                collective.start_async_param_contiguous(
                    input_tensor,
                    output_tensor,
                    count,
                    split_sizes,
                    split_offsets,
                    stream=stream,
                )
                return _MoriSdmaAllGatherWork(collective, stream)
            collective.enqueue_param_contiguous(
                input_tensor,
                output_tensor,
                count,
                split_sizes,
                split_offsets,
                stream=stream,
            )
            return None
        if async_op:
            collective.start_async(input_tensor, output_tensor, count, stream=stream)
            return _MoriSdmaAllGatherWork(collective, stream)
        collective.enqueue(input_tensor, output_tensor, count, stream=stream)
        return None

    def prepare_param_contiguous_output(
        self,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> object | None:
        self.clear_param_contiguous_output()
        if not self._zero_copy_output:
            return None
        if not all_gather_input_split_sizes:
            raise RuntimeError("MORI zero-copy allgather requires non-empty splits")
        if sum(all_gather_input_split_sizes) != all_gather_input_numel:
            raise RuntimeError(
                "MORI zero-copy allgather split sizes do not match input numel"
            )
        element_size = torch.empty((), dtype=dtype).element_size()
        split_sizes_u32: list[int] = []
        split_offsets_u32: list[int] = []
        offset = 0
        for split_size in all_gather_input_split_sizes:
            split_nbytes = int(split_size) * element_size
            if split_nbytes % 4 != 0:
                raise RuntimeError(
                    "MORI zero-copy allgather requires every split to be "
                    "4-byte aligned"
                )
            split_u32 = split_nbytes // 4
            split_offsets_u32.append(offset)
            split_sizes_u32.append(split_u32)
            offset += split_u32
        if offset * 4 != all_gather_input_numel * element_size:
            raise RuntimeError("MORI zero-copy allgather byte size mismatch")
        self._param_contiguous_split_sizes = torch.tensor(
            split_sizes_u32, dtype=torch.int64, device=device
        )
        self._param_contiguous_split_offsets = torch.tensor(
            split_offsets_u32, dtype=torch.int64, device=device
        )
        return self.param_contiguous_metadata()

    def clear_param_contiguous_output(self) -> None:
        self._param_contiguous_split_sizes = None
        self._param_contiguous_split_offsets = None

    def param_contiguous_metadata(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            self._param_contiguous_split_sizes is None
            or self._param_contiguous_split_offsets is None
        ):
            return None
        return self._param_contiguous_split_sizes, self._param_contiguous_split_offsets

    def _can_call_param_contiguous(self, input_tensor: torch.Tensor) -> bool:
        if (
            self._param_contiguous_split_sizes is None
            or self._param_contiguous_split_offsets is None
        ):
            return False
        split_nbytes = int(self._param_contiguous_split_sizes.sum().item()) * 4
        if split_nbytes != _tensor_nbytes(input_tensor):
            self.clear_param_contiguous_output()
            return False
        return True

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
            copy_output_to_user=not self._zero_copy_output,
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
        if self._zero_copy_output and not collective.is_output_registered(output_tensor):
            raise RuntimeError(
                "MORI FSDP SDMA allgather requires registered output buffers "
                "when zero-copy output is enabled"
            )
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
