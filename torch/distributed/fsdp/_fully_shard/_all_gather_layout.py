"""Experimental, private contract for FSDP all-gather backend authors.

Preparation selects a layout before output allocation. Returning None selects
the native rank-major input packing and output finalizer. Custom layouts pack
into independent input storage by default; the internal copy-in callable keeps
the native operator's signature, including its rank argument.

Finalization receives tensor-only parameter metadata on the compute stream,
after collective completion. Existing parameter objects and saved aliases keep
their storage when layouts change. Backend-owned storage stays allocated across
reshard, but its lease may be shared after AllGather.release_output(). The
backend, not FSDP, must order reuse after all local and remote consumers and
restore each parameter's original storage region before its next use.

These authoring types are not exported from torch.distributed.fsdp. Out-of-tree
backends must target a matching revision of this private interface.
``DefaultAllGatherLayout`` is exported from ``torch.distributed.fsdp.experimental``
so that its ``output_fn`` can be replaced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ._fsdp_param import FSDPParam


AllGatherCopyIn = Callable[
    [list[torch.Tensor], torch.Tensor, list[int], int, int],
    tuple[torch.Tensor, torch.Tensor],
]
# Called as fn(all_gather_output, outputs, split_sizes, outer_sizes, world_size)
# under no_grad on the current stream after the all-gather, with the outputs'
# version counters preserved. all_gather_output is the flat rank-major
# collective buffer, and outputs are the preallocated all-gather outputs, viewed
# as uint8 if the buffer is uint8, in which case split_sizes count bytes.
# outputs[i] receives each rank's split_sizes[i] elements concatenated across
# ranks along the dim whose leading dims multiply to outer_sizes[i]. A Tensor
# returned by fsdp_pre_all_gather may be smaller than its cached output, in
# which case it fills the leading split_sizes[i] * world_size elements of the
# rank-major buffer that is reassembled into outputs[i], with zeros after it.
# The callback must only write to outputs and must not keep references to its
# arguments. It is not called when the all-gather buffer is empty, or when the
# all-gather group has one rank, since FSDP then copies the inputs directly.
AllGatherOutputFn = Callable[
    [torch.Tensor, list[torch.Tensor], list[int], list[int], int], None
]


def _default_all_gather_output_fn(
    all_gather_output: torch.Tensor,
    outputs: list[torch.Tensor],
    split_sizes: list[int],
    outer_sizes: list[int],
    world_size: int,
) -> None:
    r"""Copy gathered payloads through intermediate buffers when needed.

    Nonempty payloads with more than one outer slice copy through intermediate
    buffers, then concatenate into their final layout. Other payloads copy
    directly.
    """
    copy_outputs: list[torch.Tensor] = []
    for output, split_size, outer_size in zip(outputs, split_sizes, outer_sizes):
        if outer_size == 1 or not output.numel():
            copy_output = output
        elif output.numel() == split_size * world_size:
            copy_output = torch.empty_like(output)
        else:
            copy_output = output.new_zeros(output.numel())
        copy_outputs.append(copy_output)
    torch.ops.fsdp.split_with_sizes_copy(
        all_gather_output.view(world_size, -1),
        split_sizes,
        dim=1,
        out=[
            t.view(-1).narrow(0, 0, split_size * world_size).view(world_size, -1)
            for t, split_size in zip(copy_outputs, split_sizes)
        ],
    )
    for copy_output, output, outer_size in zip(copy_outputs, outputs, outer_sizes):
        if copy_output is not output:
            chunks = copy_output.view(world_size, outer_size, -1).unbind(0)
            torch.cat(chunks, dim=1, out=output.view(outer_size, -1))


@dataclass(frozen=True, kw_only=True)
class AllGatherInputMetadata:
    """Description of this call's flattened local input and output eligibility.

    ``input_outer_sizes`` gives, for each payload, the product of its dims
    before the dim that ranks are concatenated along (see ``AllGatherOutputFn``).
    """

    input_split_sizes: list[int]
    input_outer_sizes: list[int]
    input_numel: int
    world_size: int
    dtype: torch.dtype
    device: torch.device
    can_use_param_contiguous_output: bool


@dataclass
class AllGatherParamMetadata:
    """Tensor-only description of one parameter's collective outputs.

    ``outputs`` contains the existing destinations, if any. Their storage and
    tensor identities must survive while autograd can hold saved aliases.
    ``outer_sizes`` is ``AllGatherInputMetadata.input_outer_sizes`` for this
    parameter's payloads.
    """

    input_numels: list[int]
    input_dtypes: list[torch.dtype]
    outer_sizes: list[int]
    outputs: list[torch.Tensor]
    backend_owned: bool


@dataclass
class AllGatherOutputs:
    """Outputs and whether FSDP must leave their storage to the backend.

    Backend-owned storage must remain allocated, including across reshard,
    until the parameter group is destroyed. After ``AllGather.release_output``
    it may be shared with other groups, provided overwrites are ordered after
    consumers and the same parameter regions are restored before the next use.
    """

    tensors: list[list[torch.Tensor]]
    backend_owned: bool = False


@dataclass
class _DefaultAllGatherCopyPlan:
    input_split_sizes: list[int]
    outputs: list[list[torch.Tensor]]
    outer_sizes: list[int]
    clone_input: bool = False


class AllGatherLayout(ABC):
    """Input packing and output handling for an all-gather backend.

    FSDP orders collective completion before finalization. The backend owns
    communication stream lifetimes and any persistent registered storage.
    Create a separate stateful backend/layout instance per parameter group;
    share storage through the backend's pool, not by sharing the layout instance.
    The stateless default layout is exempt from this ownership restriction.
    """

    _owner: object | None = None

    def _bind_owner(self, owner: object) -> None:
        if self._owner is None:
            self._owner = owner
        elif self._owner is not owner:
            raise ValueError(
                "an all-gather layout instance cannot be shared across FSDP "
                "parameter groups; create a separate stateful backend/layout "
                "instance for each group (shared storage may use a backend pool)"
            )

    def prepare(
        self,
        input_metadata: AllGatherInputMetadata,
    ) -> tuple[AllGatherCopyIn, AllGatherLayout, object | None]:
        """Select input packing and metadata before allocating the output."""
        metadata = self.prepare_output(input_metadata)
        if metadata is None:
            return DEFAULT_ALL_GATHER_LAYOUT.prepare(input_metadata)
        return self.copy_in, self, metadata

    @abstractmethod
    def prepare_output(
        self,
        input_metadata: AllGatherInputMetadata,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack into independent input storage without modifying the output.

        The signature matches the native rank-major copy-in operator. Layouts
        that opt into rank-local packing may use rank; this default does not.
        """
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

    @abstractmethod
    def finalize_outputs(
        self,
        all_gather_output: torch.Tensor,
        param_metadata: list[AllGatherParamMetadata],
        world_size: int,
        output_metadata: object | None,
    ) -> AllGatherOutputs:
        """Copy or view outputs after collective completion on the compute stream.

        Existing parameter storage is preserved by FSDP when returned views
        differ from it. Such calls use copy-out instead of replacing aliases.
        """
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
            # Parameters share storage, but must have independent version counters.
            param_output = all_gather_output.narrow(0, output_offset, output_numel).data
            outputs.append([param_output])
            output_offset += output_numel
        if output_offset != all_gather_output.numel():
            raise AssertionError(
                "parameter-contiguous all-gather output covered "
                f"{output_offset} of {all_gather_output.numel()} elements"
            )
        return outputs


class DefaultAllGatherLayout(AllGatherLayout):
    r"""Rank-major collective output copied into stable parameter storage.

    .. warning::
        This API is experimental and may change without backward compatibility.

    Inputs are packed into each rank's slot of the collective output, and
    ``output_fn`` copies the gathered output into the parameters' all-gather
    outputs (see ``AllGatherOutputFn``). Instances are stateless and can be
    shared across modules. Install one with
    :meth:`torch.distributed.fsdp.FSDPModule.set_all_gather_layout`.

    Args:
        output_fn (Callable): Copy-out function. Defaults to FSDP's copy
            through intermediate buffers; see
            :func:`torch.distributed.fsdp.experimental.all_gather_output_fn_with_native_copy`
            for a native implementation.
    """

    def __init__(self, output_fn: AllGatherOutputFn = _default_all_gather_output_fn):
        self.output_fn = output_fn

    def _bind_owner(self, owner: object) -> None:
        # Only the built-in default layout is known to be stateless.
        if type(self) is not DefaultAllGatherLayout:
            super()._bind_owner(owner)

    def prepare(
        self, input_metadata: AllGatherInputMetadata
    ) -> tuple[AllGatherCopyIn, AllGatherLayout, object | None]:
        return torch.ops.fsdp.all_gather_copy_in, self, input_metadata.input_split_sizes

    def prepare_output(self, input_metadata: AllGatherInputMetadata) -> None:
        return None

    def copy_in(
        self,
        all_gather_inputs: list[torch.Tensor],
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            all_gather_input_split_sizes,
            all_gather_input_numel,
            rank,
        )

    def finalize_outputs(
        self,
        all_gather_output: torch.Tensor,
        param_metadata: list[AllGatherParamMetadata],
        world_size: int,
        output_metadata: object | None,
    ) -> AllGatherOutputs:
        from ._fsdp_param import alloc_storage

        if isinstance(output_metadata, _DefaultAllGatherCopyPlan):
            plan = output_metadata
        else:
            # Custom layouts may delegate rank-major output to this finalizer.
            plan = _DefaultAllGatherCopyPlan(
                [] if output_metadata is None else cast(list[int], output_metadata),
                [],
                [],
            )
            for param in param_metadata:
                param_outputs = param.outputs or [
                    torch.empty(
                        numel * world_size, dtype=dtype, device=all_gather_output.device
                    )
                    for numel, dtype in zip(param.input_numels, param.input_dtypes)
                ]
                if param.backend_owned:
                    plan.clone_input = plan.clone_input or any(
                        tensor.untyped_storage().data_ptr()
                        == all_gather_output.untyped_storage().data_ptr()
                        for tensor in param.outputs
                    )
                else:
                    for tensor in param_outputs:
                        alloc_storage(tensor)
                plan.outputs.append(param_outputs)
                if output_metadata is None:
                    plan.input_split_sizes.extend(
                        numel
                        * tensor.element_size()
                        // all_gather_output.element_size()
                        for numel, tensor in zip(param.input_numels, param_outputs)
                    )
                plan.outer_sizes.extend(param.outer_sizes)

        if all_gather_output.numel() == 0:
            return AllGatherOutputs(plan.outputs)
        if plan.clone_input:
            # Fallback may gather into storage that existing parameters alias.
            all_gather_output = all_gather_output.clone()
        outputs = [output for param_outputs in plan.outputs for output in param_outputs]
        non_inference_outputs = tuple(t for t in outputs if not t.is_inference())
        if all_gather_output.dtype == torch.uint8:
            outputs = [t.view(torch.uint8) for t in outputs]
        # Views share their base's version counter
        with torch.autograd._unsafe_preserve_version_counter(non_inference_outputs):
            self.output_fn(
                all_gather_output,
                outputs,
                plan.input_split_sizes,
                plan.outer_sizes,
                world_size,
            )
        return AllGatherOutputs(plan.outputs)


DEFAULT_ALL_GATHER_LAYOUT = DefaultAllGatherLayout()


def _can_use_param_contiguous_output(
    fsdp_params: list[FSDPParam],
    param_input_dtypes: list[list[torch.dtype]],
    param_input_numels: list[list[int]],
    output_dtype: torch.dtype,
) -> bool:
    from torch._dynamo.compiled_autograd import (
        compiled_autograd_enabled,
        in_compiled_autograd_region,
    )

    from ._fsdp_param import ShardedState

    if compiled_autograd_enabled or in_compiled_autograd_region:
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


def _init_layout_outputs(
    fsdp_params: list[FSDPParam],
    result: AllGatherOutputs,
) -> None:
    from ._fsdp_param import alloc_storage

    if len(fsdp_params) != len(result.tensors):
        raise AssertionError(
            f"all-gather layout returned {len(result.tensors)} parameter outputs "
            f"for {len(fsdp_params)} parameters"
        )
    for fsdp_param, outputs in zip(fsdp_params, result.tensors):
        if not outputs:
            raise AssertionError("all-gather layout returned no output for a parameter")
        previous = fsdp_param.all_gather_outputs
        if not previous:
            fsdp_param.all_gather_outputs = outputs
            fsdp_param._keep_all_gather_output_storage = result.backend_owned
        elif outputs is not previous:
            if len(previous) != len(outputs):
                raise AssertionError("all-gather layout changed the number of outputs")
            # Saved tensors may alias this storage even after reshard. Preserve
            # it when switching layouts or receiving a new backend buffer.
            for target, source in zip(previous, outputs):
                if target.shape != source.shape or target.dtype != source.dtype:
                    raise AssertionError(
                        "all-gather layout changed output shape or dtype"
                    )
                if not fsdp_param._keep_all_gather_output_storage:
                    alloc_storage(target)
                if target.data_ptr() != source.data_ptr():
                    with torch.autograd._unsafe_preserve_version_counter(
                        () if target.is_inference() else (target,)
                    ):
                        target.copy_(source)
