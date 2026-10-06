"""Experimental contract for FSDP all-gather input packing and output handling.

A layout prepares each all-gather before its output is allocated, packs the
collective input, and finalizes the collective output into the parameters'
all-gather outputs after the collective completes.

FSDP owns the persistent buffers behind those outputs. On a parameter group's
first unshard, finalize allocates them in any layout it chooses (e.g. one buffer
viewed by every parameter) and returns them with outputs that view them. The
group frees their storage on reshard and re-allocates it to its recorded size
before each later finalize, which must refill the same outputs in place.
``DefaultAllGatherLayout`` gives each output its own buffer, which its
parameter allocates and frees, and refills it with an ``AllGatherOutputFn``.
Only a ``BackendOwnedAllGatherLayout`` may instead return views into storage the
backend owns, which FSDP keeps across reshard.

A layout that needs a specific collective holds it as ``comm``.
``FSDPModule.set_all_gather_layout`` is the only way to install a layout, and
installing a layout with a ``comm`` installs that comm too; FSDP then rejects
any other all-gather comm for the group until a different layout is installed.

``AllGatherLayout`` and its metadata types are private authoring interfaces;
out-of-tree backends must target a matching revision of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ._fsdp_api import AllGather
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

    ``outputs`` is empty on the group's first unshard and afterwards holds the
    outputs adopted then, which autograd may alias, so finalize refills them in
    place. ``outer_sizes`` is ``AllGatherInputMetadata.input_outer_sizes`` for
    this parameter's payloads. ``backend_owned`` marks outputs adopted from a
    ``BackendOwnedAllGatherLayout``.
    """

    input_numels: list[int]
    input_dtypes: list[torch.dtype]
    outer_sizes: list[int]
    outputs: list[torch.Tensor]
    backend_owned: bool


@dataclass
class AllGatherOutputs:
    """Per-parameter all-gather outputs and the persistent buffers they view.

    On the group's first unshard FSDP adopts ``tensors`` as the parameters'
    outputs and ``buffers`` as the FSDP-owned buffers behind them; every
    nonempty output must view one of the buffers. Only a
    ``BackendOwnedAllGatherLayout`` may return no buffers, for outputs that view
    storage the backend owns. Later finalizes return the outputs they refilled.
    """

    tensors: list[list[torch.Tensor]]
    buffers: list[torch.Tensor] = field(default_factory=list)


@dataclass
class AllGatherFinalizeMetadata:
    """Arguments of ``AllGatherLayout.finalize_outputs``, in one object so that
    new fields do not break out-of-tree layouts.

    ``all_gather_output`` is the completed collective output and
    ``output_metadata`` is what ``prepare`` returned for this call. ``buffers``
    is empty on the group's first unshard and afterwards holds the group's
    persistent buffers with storage re-allocated.
    """

    all_gather_output: torch.Tensor
    param_metadata: list[AllGatherParamMetadata]
    world_size: int
    output_metadata: object | None
    buffers: list[torch.Tensor]


@dataclass
class _PersistentBuffers:
    """FSDP-owned buffers that a layout chose on a group's first unshard and
    the storage size of each in bytes, which may exceed a view's own size."""

    tensors: list[torch.Tensor]
    nbytes: list[int]

    def alloc(self) -> None:
        for tensor, nbytes in zip(self.tensors, self.nbytes):
            if (storage := tensor.untyped_storage()).size() != nbytes:
                storage.resize_(nbytes)

    def free(self) -> None:
        for tensor in self.tensors:
            if (storage := tensor.untyped_storage()).size() != 0:
                storage.resize_(0)


@dataclass
class _DefaultAllGatherCopyPlan:
    input_split_sizes: list[int]
    outputs: list[list[torch.Tensor]]
    outer_sizes: list[int]
    clone_input: bool = False


class AllGatherLayout(ABC):
    """Input packing and output handling for an all-gather backend.

    FSDP orders collective completion before finalization and owns the
    persistent buffers that finalize returns (see the module documentation). A
    stateful layout instance belongs to one parameter group.
    ``DefaultAllGatherLayout`` is stateless and may be shared. ``comm`` is the
    collective this layout requires, if any, which FSDP installs with it.
    """

    comm: AllGather | None = None
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
    def finalize_outputs(self, metadata: AllGatherFinalizeMetadata) -> AllGatherOutputs:
        """Fill the parameters' outputs after collective completion on the current stream.

        On the group's first unshard ``metadata.buffers`` is empty: allocate the
        persistent buffers and return outputs that view them. Later calls pass
        those buffers, with storage re-allocated, and the adopted outputs in
        ``metadata.param_metadata``, which must be refilled in place and
        returned, under preserved version counters.
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


class BackendOwnedAllGatherLayout(AllGatherLayout):
    """Opt-in for outputs that view storage the all-gather backend owns.

    FSDP adopts the outputs of this layout's first finalize without owning
    them: it neither frees nor re-allocates their storage, preserves version
    counters around input packing and the collective, which may write storage
    that saved parameters alias, and calls ``release_output``. Later finalizes
    may return new views; FSDP copies them into the adopted outputs if their
    storage moved.
    """

    @abstractmethod
    def release_output(self) -> None:
        """Release the group's output lease on the current stream.

        FSDP calls this after reshard, after waiting for a discarded unused
        all-gather, and after a failed input preparation or collective setup,
        so it must be idempotent when no output is active. Adopted outputs must
        remain valid objects; a backend sharing their storage must restore the
        same regions on the next gather and order overwrites after all local
        and remote consumers, since FSDP does not synchronize that reuse.
        """
        ...


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

    def finalize_outputs(self, metadata: AllGatherFinalizeMetadata) -> AllGatherOutputs:
        all_gather_output, world_size = metadata.all_gather_output, metadata.world_size
        # FSDP passes a plan with allocated outputs for a custom layout's
        # fallback; custom layouts that delegate pass per-parameter metadata
        if isinstance(metadata.output_metadata, _DefaultAllGatherCopyPlan):
            plan, new_buffers = metadata.output_metadata, []
        else:
            plan, new_buffers = _plan_rank_major_outputs(
                all_gather_output, metadata.param_metadata, world_size
            )
        buffers = metadata.buffers or new_buffers
        if all_gather_output.numel() == 0:
            return AllGatherOutputs(plan.outputs, buffers)
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
        return AllGatherOutputs(plan.outputs, buffers)


DEFAULT_ALL_GATHER_LAYOUT = DefaultAllGatherLayout()


def _plan_rank_major_outputs(
    all_gather_output: torch.Tensor,
    param_metadata: list[AllGatherParamMetadata],
    world_size: int,
) -> tuple[_DefaultAllGatherCopyPlan, list[torch.Tensor]]:
    """Returns the copy plan and, on the first unshard, the new outputs, each
    its own persistent buffer. FSDP has re-allocated existing outputs."""
    plan = _DefaultAllGatherCopyPlan([], [], [])
    new_buffers: list[torch.Tensor] = []
    device = all_gather_output.device
    for param in param_metadata:
        outputs = param.outputs
        if not outputs:
            outputs = [
                torch.empty(numel * world_size, dtype=dtype, device=device)
                for numel, dtype in zip(param.input_numels, param.input_dtypes)
            ]
            new_buffers.extend(outputs)
        elif param.backend_owned:
            plan.clone_input = plan.clone_input or any(
                t.untyped_storage().data_ptr()
                == all_gather_output.untyped_storage().data_ptr()
                for t in outputs
            )
        plan.outputs.append(outputs)
        plan.input_split_sizes.extend(
            numel * output.element_size() // all_gather_output.element_size()
            for numel, output in zip(param.input_numels, outputs)
        )
        plan.outer_sizes.extend(param.outer_sizes)
    return plan, new_buffers


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


def _adopt_layout_outputs(
    fsdp_params: list[FSDPParam], result: AllGatherOutputs, backend_owned: bool
) -> _PersistentBuffers | None:
    """Adopts a group's first finalized outputs and returns its FSDP-owned
    buffers, or None if the backend owns the outputs' storage."""
    if len(fsdp_params) != len(result.tensors):
        raise AssertionError(
            f"all-gather layout returned {len(result.tensors)} parameter outputs "
            f"for {len(fsdp_params)} parameters"
        )
    if not result.buffers:
        if not backend_owned:
            raise AssertionError(
                "all-gather layout returned no persistent buffers; only a "
                "BackendOwnedAllGatherLayout may return outputs that FSDP does not own"
            )
        for fsdp_param, outputs in zip(fsdp_params, result.tensors):
            fsdp_param.all_gather_outputs = outputs
            fsdp_param._keep_all_gather_output_storage = True
        return None
    buffer_storages = {buffer.untyped_storage().data_ptr() for buffer in result.buffers}
    for fsdp_param, outputs in zip(fsdp_params, result.tensors):
        # Refills write into the buffers, so outputs must view them
        if any(
            t.numel() and t.untyped_storage().data_ptr() not in buffer_storages
            for t in outputs
        ):
            raise AssertionError(
                f"all-gather output of {fsdp_param._param_fqn} does not view the "
                "persistent buffers its layout returned, so refills would not reach it"
            )
        fsdp_param.all_gather_outputs = outputs
        # The group allocates and frees the buffers the outputs view
        fsdp_param._keep_all_gather_output_storage = True
    return _PersistentBuffers(
        list(result.buffers),
        [buffer.untyped_storage().size() for buffer in result.buffers],
    )


def _check_layout_refill(
    fsdp_params: list[FSDPParam], result: AllGatherOutputs, backend_owned: bool
) -> None:
    if len(fsdp_params) != len(result.tensors):
        raise AssertionError(
            f"all-gather layout returned {len(result.tensors)} parameter outputs "
            f"for {len(fsdp_params)} parameters"
        )
    for fsdp_param, outputs in zip(fsdp_params, result.tensors):
        previous = fsdp_param.all_gather_outputs
        if len(previous) != len(outputs):
            raise AssertionError("all-gather layout changed the number of outputs")
        if not backend_owned:
            if any(a is not b for a, b in zip(outputs, previous)):
                raise AssertionError(
                    "all-gather layout did not refill the outputs it was given in "
                    "place; only a BackendOwnedAllGatherLayout may return new outputs"
                )
            continue
        # Saved tensors alias the adopted outputs, so copy new views into them
        for target, source in zip(previous, outputs):
            if target.shape != source.shape or target.dtype != source.dtype:
                raise AssertionError("all-gather layout changed output shape or dtype")
            if target.data_ptr() != source.data_ptr():
                with torch.autograd._unsafe_preserve_version_counter(
                    () if target.is_inference() else (target,)
                ):
                    target.copy_(source)
