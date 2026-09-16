from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import torch

from ._fsdp_param import FSDPParam, ShardedState


AllGatherCopyIn = Callable[
    [list[torch.Tensor], torch.Tensor, list[int], int, int],
    tuple[torch.Tensor, torch.Tensor],
]


class AllGatherLayout(ABC):
    """Optional input packing and output layout for an all-gather backend."""

    _owner: object | None = None

    def _bind_owner(self, owner: object) -> None:
        if self._owner is None:
            self._owner = owner
        elif self._owner is not owner:
            raise ValueError(
                "an all-gather layout instance cannot be shared across FSDP "
                "parameter groups"
            )

    def prepare(
        self,
        input_split_sizes: list[int],
        input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
        param_input_dtypes: list[list[torch.dtype]],
        param_input_numels: list[list[int]],
        can_use_param_contiguous_output: bool,
    ) -> tuple[AllGatherCopyIn, object | None]:
        """Select input packing and metadata before allocating the output."""
        metadata = self.prepare_output(
            input_split_sizes,
            input_numel,
            world_size,
            dtype,
            device,
            param_input_dtypes,
            param_input_numels,
            can_use_param_contiguous_output,
        )
        if metadata is None:
            return torch.ops.fsdp.all_gather_copy_in, None
        return partial(self.copy_in, output_metadata=metadata), metadata

    @abstractmethod
    def prepare_output(
        self,
        input_split_sizes: list[int],
        input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
        param_input_dtypes: list[list[torch.dtype]],
        param_input_numels: list[list[int]],
        can_use_param_contiguous_output: bool,
    ) -> object | None:
        """Return per-call metadata, or None to use rank-major input and output.

        The backend must produce the selected layout for this collective.
        Metadata must remain valid until its result is finalized.
        """
        ...

    def copy_in(
        self,
        all_gather_inputs: list[torch.Tensor],
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        rank: int,
        output_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack inputs for a collective using this layout."""
        return torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            all_gather_input_split_sizes,
            all_gather_input_numel,
            rank,
        )

    @abstractmethod
    def finalize_outputs(
        self,
        all_gather_output: torch.Tensor,
        param_input_numels: list[list[int]],
        world_size: int,
        output_metadata: object,
    ) -> list[list[torch.Tensor]]:
        """Return the per-parameter outputs after collective completion."""
        ...

    def param_contiguous_output_views(
        self,
        all_gather_output: torch.Tensor,
        param_input_numels: list[list[int]],
        world_size: int,
    ) -> list[list[torch.Tensor]]:
        """Carve a parameter-contiguous output into per-parameter views."""
        output_offset = 0
        outputs: list[list[torch.Tensor]] = []
        for input_numels in param_input_numels:
            output_numel = input_numels[0] * world_size
            param_output = all_gather_output.narrow(0, output_offset, output_numel)
            outputs.append([param_output])
            output_offset += output_numel
        if output_offset != all_gather_output.numel():
            raise AssertionError(
                "parameter-contiguous all-gather output covered "
                f"{output_offset} of {all_gather_output.numel()} elements"
            )
        return outputs


def _can_use_param_contiguous_output(
    fsdp_params: list[FSDPParam],
    param_input_dtypes: list[list[torch.dtype]],
    param_input_numels: list[list[int]],
    output_dtype: torch.dtype,
) -> bool:
    if _compile_active():
        return False
    if not (len(fsdp_params) == len(param_input_dtypes) == len(param_input_numels)):
        return False
    for fsdp_param, input_dtypes, input_numels in zip(
        fsdp_params, param_input_dtypes, param_input_numels
    ):
        if (
            len(input_dtypes) != 1
            or len(input_numels) != 1
            or input_dtypes[0] != output_dtype
            or fsdp_param.fsdp_placement.dim != 0
            or fsdp_param.is_dtensor
            or hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
            or hasattr(fsdp_param._sharded_local_tensor, "fsdp_post_all_gather")
            or fsdp_param.sharded_state == ShardedState.SHARDED_POST_FORWARD
        ):
            return False
    return True


def _compile_active() -> bool:
    if torch.compiler.is_compiling():
        return True
    from torch._dynamo.compiled_autograd import compiled_autograd_enabled

    return compiled_autograd_enabled


def _init_layout_outputs(
    fsdp_params: list[FSDPParam],
    param_outputs: list[list[torch.Tensor]],
) -> None:
    if len(fsdp_params) != len(param_outputs):
        raise AssertionError(
            f"all-gather layout returned {len(param_outputs)} parameter outputs "
            f"for {len(fsdp_params)} parameters"
        )
    for fsdp_param, outputs in zip(fsdp_params, param_outputs):
        if not outputs:
            raise AssertionError("all-gather layout returned no output for a parameter")
        if (
            hasattr(fsdp_param, "_unsharded_param")
            and fsdp_param._unsharded_param.data_ptr() != outputs[0].data_ptr()
        ):
            del fsdp_param._unsharded_param
        fsdp_param.all_gather_outputs = outputs
        fsdp_param._keep_all_gather_output_storage = True
