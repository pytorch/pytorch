# mypy: allow-untyped-defs
"""MORI SDMA all-gather backend for FSDP2 on ROCm.

This is an opt-in :class:`AllGather` backend backed by the ROCm `MORI
<https://github.com/ROCm/mori>`_ SDMA collectives. Enable it on an FSDP module
with::

    from torch.distributed.fsdp._fully_shard._mori_sdma_allgather import (
        MoriSdmaAllGather,
    )

    model.set_custom_all_gather(MoriSdmaAllGather(zero_copy_output=True))

When ``zero_copy_output`` is set the backend produces a parameter-contiguous
output that FSDP can use in place, avoiding the rank-major copy-out. The
``mori`` package is imported lazily so importing this module does not require
ROCm/MORI to be installed.
"""

import importlib
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist

from ._fsdp_api import AllGather


if TYPE_CHECKING:
    from ._fsdp_collectives import AllGatherResult
    from ._fsdp_param import FSDPParam


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
    """All-gather backend using MORI SDMA collectives (ROCm).

    Args:
        zero_copy_output (bool): produce a parameter-contiguous output that FSDP
            uses in place, skipping the rank-major copy-out wherever the
            parameter group is eligible. Defaults to ``True``.
    """

    def __init__(self, zero_copy_output: bool = True) -> None:
        self._zero_copy_output = zero_copy_output
        self._collective: Any | None = None
        self._rank: int | None = None
        self._world_size: int | None = None
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
        if (
            self._output_buffer is not None
            and self._output_buffer.dtype == dtype
            and self._output_buffer.device == device
            and self._output_buffer.numel() >= numel
        ):
            return self._output_buffer.narrow(0, 0, numel)
        self._deregister_output_buffer_if_needed()
        self._output_buffer = torch.empty(*size, dtype=dtype, device=device)
        self._output_buffer_nbytes = _tensor_nbytes(self._output_buffer)
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

    def prepare_output(
        self,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
        fsdp_params: list["FSDPParam"],
        param_all_gather_input_dtypes: list[list[torch.dtype]],
        param_all_gather_input_numels: list[list[int]],
    ) -> object | None:
        if not self._zero_copy_output:
            return None

        if not self.can_use_param_contiguous_output(
            fsdp_params,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ):
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
                    "MORI zero-copy allgather requires every split to be 4-byte aligned"
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
        return (
            self._param_contiguous_split_sizes,
            self._param_contiguous_split_offsets,
        )

    def copy_in(
        self,
        all_gather_inputs: list[torch.Tensor],
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        rank: int,
        output_metadata: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_metadata is None:
            return super().copy_in(
                all_gather_inputs,
                all_gather_output,
                all_gather_input_split_sizes,
                all_gather_input_numel,
                rank,
                output_metadata,
            )
        all_gather_input = torch.empty(
            (all_gather_input_numel,),
            dtype=all_gather_output.dtype,
            device=all_gather_output.device,
        )
        torch._foreach_copy_(
            torch.split(all_gather_input, all_gather_input_split_sizes),
            all_gather_inputs,
        )
        return all_gather_input, all_gather_output

    def finalize_outputs(
        self,
        all_gather_result: "AllGatherResult",
        fsdp_params: list["FSDPParam"],
        group: dist.ProcessGroup,
        default_finalize: Callable[[], None],
    ) -> None:
        if all_gather_result.output_metadata is None:
            default_finalize()
            return

        self.init_param_contiguous_outputs(
            all_gather_result.all_gather_output,
            fsdp_params,
            all_gather_result.param_all_gather_input_numels,
            group.size(),
        )

    def clear_output(self) -> None:
        self._param_contiguous_split_sizes = None
        self._param_contiguous_split_offsets = None

    def _can_call_param_contiguous(self, input_tensor: torch.Tensor) -> bool:
        if (
            self._param_contiguous_split_sizes is None
            or self._param_contiguous_split_offsets is None
        ):
            return False
        split_nbytes = int(self._param_contiguous_split_sizes.sum().item()) * 4
        if split_nbytes != _tensor_nbytes(input_tensor):
            self.clear_output()
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

        try:
            shmem = importlib.import_module("mori.shmem")
            AllgatherSdma = importlib.import_module("mori.ccl").AllgatherSdma
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] == "mori":
                raise RuntimeError(
                    "MoriSdmaAllGather requires the optional ROCm MORI Python "
                    "package providing `mori.shmem` and `mori.ccl`. Install or "
                    "load MORI before using this backend."
                ) from exc
            raise

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
        self._registered_output_ptr = None
        return self._collective

    def _ensure_output_registered(
        self, collective: Any, output_tensor: torch.Tensor
    ) -> None:
        ptr = output_tensor.data_ptr()
        nbytes = _tensor_nbytes(output_tensor)
        if self._registered_output_ptr == ptr and self._output_buffer_nbytes >= nbytes:
            return
        if collective.is_output_registered(output_tensor):
            self._registered_output_ptr = ptr
            return
        collective.register_output_buffer(output_tensor)
        if self._zero_copy_output and not collective.is_output_registered(
            output_tensor
        ):
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
