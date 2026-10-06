"""Experimental contract for FSDP all-gather input packing and output handling.

A layout prepares each all-gather before its output is allocated, packs the
collective input, and finalizes the collective output into the parameters'
all-gather outputs after the collective completes. A layout may fall back to
``DefaultAllGatherLayout`` for any single all-gather.

FSDP owns the persistent buffers behind those outputs. On a parameter group's
first unshard, finalize allocates them in any layout it chooses (e.g. one buffer
viewed by every parameter) and returns them with outputs that view them. The
group frees their storage on reshard, unless its parameters keep their
unsharded storage, and re-allocates it to its recorded size before each later
finalize, which must refill the same outputs in place.
``DefaultAllGatherLayout`` gives each output its own buffer, which its
parameter allocates and frees, and refills it with an ``AllGatherOutputFn``.

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
from dataclasses import dataclass
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
    """Description of this call's flattened local input.

    ``input_outer_sizes`` gives, for each payload, the product of its dims
    before the dim that ranks are concatenated along (see ``AllGatherOutputFn``).
    """

    input_split_sizes: list[int]
    input_outer_sizes: list[int]
    input_numel: int
    world_size: int
    dtype: torch.dtype
    device: torch.device


@dataclass
class AllGatherParamMetadata:
    """Tensor-only description of one parameter's collective outputs.

    ``outputs`` is empty on the group's first unshard and afterwards holds the
    outputs adopted then, which autograd may alias, so finalize refills them in
    place. ``outer_sizes`` is ``AllGatherInputMetadata.input_outer_sizes`` for
    this parameter's payloads.
    """

    input_numels: list[int]
    input_dtypes: list[torch.dtype]
    outer_sizes: list[int]
    outputs: list[torch.Tensor]


@dataclass
class AllGatherOutputs:
    """Per-parameter all-gather outputs and the persistent buffers they view.

    On the group's first unshard FSDP adopts ``tensors`` as the parameters'
    outputs and ``buffers`` as the FSDP-owned buffers behind them; every
    nonempty output must view one of the buffers and should have its own
    version counter, e.g. a ``.data`` view, like a separately allocated output.
    Later finalizes return the outputs they refilled and the buffers they were
    given.
    """

    tensors: list[list[torch.Tensor]]
    buffers: list[torch.Tensor]


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
    outputs: list[list[torch.Tensor]]
    split_sizes: list[int]
    outer_sizes: list[int]


class AllGatherLayout(ABC):
    """Input packing and output handling for an FSDP parameter group's all-gather.

    FSDP orders collective completion before finalization and owns the
    persistent buffers that finalize returns (see the module documentation).
    Parameter groups' all-gathers can overlap, so a layout that keeps per-call
    state, e.g. for its comm, must not be shared across groups.
    ``DefaultAllGatherLayout`` is stateless and may be shared. ``comm`` is the
    collective this layout requires, if any, which FSDP installs with it.
    """

    comm: AllGather | None = None

    def prepare(
        self, input_metadata: AllGatherInputMetadata
    ) -> tuple[AllGatherCopyIn, AllGatherLayout, object | None]:
        """Select input packing and per-call metadata before allocating the output."""
        metadata = self.prepare_output(input_metadata)
        if metadata is None:
            return DEFAULT_ALL_GATHER_LAYOUT.prepare(input_metadata)
        return self.copy_in, self, metadata

    @abstractmethod
    def prepare_output(self, input_metadata: AllGatherInputMetadata) -> object | None:
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

        The signature matches the native rank-major copy-in operator.
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


class DefaultAllGatherLayout(AllGatherLayout):
    r"""Rank-major all-gather copied into stable parameter storage.

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

    def prepare(
        self, input_metadata: AllGatherInputMetadata
    ) -> tuple[AllGatherCopyIn, AllGatherLayout, object | None]:
        return torch.ops.fsdp.all_gather_copy_in, self, None

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
        outputs = [output for param_outputs in plan.outputs for output in param_outputs]
        non_inference_outputs = tuple(t for t in outputs if not t.is_inference())
        if all_gather_output.dtype == torch.uint8:
            outputs = [t.view(torch.uint8) for t in outputs]
        # Views share their base's version counter
        with torch.autograd._unsafe_preserve_version_counter(non_inference_outputs):
            self.output_fn(
                all_gather_output,
                outputs,
                plan.split_sizes,
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
        plan.outputs.append(outputs)
        plan.split_sizes.extend(
            numel * output.element_size() // all_gather_output.element_size()
            for numel, output in zip(param.input_numels, outputs)
        )
        plan.outer_sizes.extend(param.outer_sizes)
    return plan, new_buffers


def _adopt_layout_outputs(
    fsdp_params: list[FSDPParam], result: AllGatherOutputs
) -> _PersistentBuffers:
    """Adopts a group's first finalized outputs and returns the FSDP-owned
    buffers behind them."""
    if len(fsdp_params) != len(result.tensors):
        raise AssertionError(
            f"all-gather layout returned {len(result.tensors)} parameter outputs "
            f"for {len(fsdp_params)} parameters"
        )
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
    fsdp_params: list[FSDPParam], result: AllGatherOutputs
) -> None:
    if len(fsdp_params) != len(result.tensors):
        raise AssertionError(
            f"all-gather layout returned {len(result.tensors)} parameter outputs "
            f"for {len(fsdp_params)} parameters"
        )
    for fsdp_param, outputs in zip(fsdp_params, result.tensors):
        previous = fsdp_param.all_gather_outputs
        if len(previous) != len(outputs) or any(
            a is not b for a, b in zip(outputs, previous)
        ):
            raise AssertionError(
                "all-gather layout did not refill the outputs it was given in place"
            )
