from typing import List, NamedTuple, Optional

import torch
import torch.distributed as dist
from torch.utils._contextlib import _DecoratorContextManager

from ._fsdp_param import FSDPParam


class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.cuda.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    all_gather_input_numels: List[int]


class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.cuda.Event  # copy-out


class AllGatherStateHolder:
    def __init__(self):
        self._state: Optional[AllGatherState] = None

    def put(self, state: AllGatherState) -> None:
        assert self._state is None, "Expects to hold only one all-gather state"
        self._state = state

    def pop(self) -> Optional[AllGatherState]:
        state = self._state
        self._state = None
        return state


@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
    async_op: bool,
    all_gather_copy_in_stream: torch.cuda.Stream,
    all_gather_stream: torch.cuda.Stream,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[AllGatherResult]:
    world_size, rank = (group.size(), group.rank())
    # - Copy in
    with torch.cuda.stream(all_gather_copy_in_stream):
        param_all_gather_inputs = [
            fsdp_param.all_gather_input for fsdp_param in fsdp_params
        ]
        inp_split_sizes = [inp.numel() for inp in param_all_gather_inputs]
        all_gather_input_numel = sum(inp_split_sizes)
        all_gather_output = torch.empty(
            (all_gather_input_numel * world_size,), dtype=dtype, device=device
        )
        all_gather_input = all_gather_output.narrow(
            0, all_gather_input_numel * rank, all_gather_input_numel
        )
        foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
        torch._foreach_copy_(foreach_copy_dsts, param_all_gather_inputs)
        del param_all_gather_inputs
        all_gather_copy_in_event = torch.cuda.Event()
        all_gather_copy_in_event.record()
    all_gather_stream.wait_event(all_gather_copy_in_event)
    with torch.cuda.stream(all_gather_stream):
        # - All-gather
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        all_gather_event = torch.cuda.Event()
        all_gather_event.record()
        return AllGatherResult(
            all_gather_output, all_gather_event, all_gather_work, inp_split_sizes
        )


def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,
    fsdp_params: List[FSDPParam],
    group: dist.ProcessGroup,
) -> None:
    all_gather_output, _, _, all_gather_input_numels = all_gather_result
    if (event := all_gather_result.all_gather_event) is not None:  # sync op
        torch.cuda.current_stream().wait_event(event)
    if (work := all_gather_result.all_gather_work) is not None:  # async op
        work.wait()
    world_size = group.size()
    dtype, device = all_gather_output.dtype, all_gather_output.device
    for all_gather_input_numel, fsdp_param in zip(all_gather_input_numels, fsdp_params):
        fsdp_param.init_all_gather_output(
            all_gather_input_numel, world_size, dtype, device
        )  # no-op after 1st call
        fsdp_param.alloc_all_gather_output()
    all_gather_output = all_gather_output.view(world_size, -1)
    out = [
        fsdp_param.all_gather_output.view(world_size, -1) for fsdp_param in fsdp_params
    ]
    # TODO: Use `torch.split_with_sizes_copy` fast path once it lands.
    splits = torch.split(all_gather_output, all_gather_input_numels, dim=1)
    with _unsafe_preserve_version_counters(out):
        torch._foreach_copy_(out, splits)  # one `copy_` per parameter


# We need this context for the backward all-gather, which would otherwise
# raise an error when writing to the all-gather output tensors in-place, e.g.:
# RuntimeError: one of the variables needed for gradient computation has been
# modified by an inplace operation: [torch.cuda.FloatTensor [15, 3]], which is
# output 0 of AsStridedBackward0, is at version 3; expected version 2 instead.
class _unsafe_preserve_version_counters(_DecoratorContextManager):
    # Same as `_unsafe_preserve_version_counter` but only entering/exiting the
    # context manager once for a list of tensors to reduce CPU overhead
    def __init__(self, tensors: List[torch.Tensor]) -> None:
        self.tensors = tensors
        self.prev_versions = [t._version for t in tensors]

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args) -> None:
        for tensor, prev_version in zip(self.tensors, self.prev_versions):
            torch._C._autograd._unsafe_set_version_counter(tensor, prev_version)
