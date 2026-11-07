# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``Shard`` for buckets whose parameters define their own all-gather inputs.

The frontend's ``ParamRepresentation`` supplies each parameter's all-gather
inputs, e.g. from FSDP2's all-gather extensions (``fsdp_pre_all_gather``), and
builds its unsharded tensor from the gathered outputs
(``fsdp_post_all_gather``). The bucket packs the inputs with its
``AllGatherLayout``, which a custom layout may replace, and keeps the outputs
across unshards, like the persistent buffers of ``Shard``'s own path. The
gradient reduction is ``Shard``'s.
"""

from __future__ import annotations

import contextlib
import math
import operator
from dataclasses import dataclass
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from ..all_gather_layout import (
    _get_all_gather_input_metadatas,
    _PersistentBuffers,
    AllGatherInput,
    AllGatherInputMetadata,
    AllGatherLayout,
    AllGatherResult,
    DEFAULT_ALL_GATHER_LAYOUT,
    DefaultAllGatherLayout,
    foreach_all_gather_copy_out,
)
from ..placement_contract import PlacementPreparedUnshard, PlacementUnshardResult
from .shard import Shard


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..bucket_runtime import ParamRepresentation
    from ..bucket_storage import ParamInfo


class _AllGatherOutputLayout(NamedTuple):
    # View size of the output passed to fsdp_post_all_gather; may contain -1
    output_size: torch.Size
    # Rows of each rank's chunk that the copy-out concatenates: the product of the
    # dims before the gather dim (the padded sharded parameter's for Tensor
    # payloads), or 1 if a flat copy suffices
    outer_size: int


@dataclass
class ParamAllGatherInputs:
    """A parameter's all-gather inputs for one unshard, from its representation."""

    inputs: Sequence[torch.Tensor | AllGatherInput]
    # Whether the parameter defines them (an all-gather extension), so that
    # their output layouts follow them; otherwise one input holds the padded
    # sharded data
    from_extension: bool = False
    # Whether Tensor inputs must have the padded sharded size: only an extension
    # that receives the outer size can pad them (pytorch/pytorch#137005)
    check_padding: bool = False


def _alloc_storage(tensor: torch.Tensor) -> None:
    size = tensor.numel() * tensor.itemsize
    if (storage := tensor.untyped_storage()).size() != size:
        storage.resize_(size)


def _free_storage(tensor: torch.Tensor) -> None:
    if (storage := tensor.untyped_storage()).size() != 0:
        storage.resize_(0)


class ParamAllGather:
    """One parameter's all-gather outputs, which persist across unshards and
    back its unsharded param."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.all_gather_outputs: list[torch.Tensor] = []
        # Set when the outputs view a custom layout's buffers, which the bucket
        # allocates and frees
        self.outputs_view_layout_buffers = False
        # Output layouts of the inputs last flattened
        self.output_layouts: tuple[_AllGatherOutputLayout, ...] = ()
        self.from_extension = False
        # Extension layouts checked against the allocated outputs, which never
        # change, keyed by each input's size, dtype and, for records, integer dim
        self._layouts_by_key: dict[Any, tuple[_AllGatherOutputLayout, ...]] = {}
        # Tensors that hold the unsharded tensor's data besides the outputs, as
        # the representation's post_all_gather returned them
        self.inner_tensors: list[torch.Tensor] = []
        self.unsharded_param: nn.Parameter | None = None

    def init_all_gather_outputs(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[torch.dtype],
        world_size: int,
        device: torch.device,
    ) -> None:
        if self.all_gather_outputs:
            return  # already initialized
        self.all_gather_outputs = [
            torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def alloc_all_gather_outputs(self) -> None:
        if self.outputs_view_layout_buffers:
            return
        for tensor in self.all_gather_outputs:
            _alloc_storage(tensor)

    def free_all_gather_outputs(self) -> None:
        if self.outputs_view_layout_buffers:
            return
        for tensor in self.all_gather_outputs:
            _free_storage(tensor)

    def unflatten_all_gather_outputs(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.view(*layout.output_size)
            for tensor, layout in zip(self.all_gather_outputs, self.output_layouts)
        )

    def flatten_inputs(
        self,
        inputs: ParamAllGatherInputs,
        *,
        shard_world_size: int,
        shard_dim: int,
        padded_sharded_size: torch.Size,
        uneven: bool,
    ) -> list[torch.Tensor]:
        """Record the output layouts of ``inputs``, which must fit the
        parameter's sharding and any allocated outputs, and return them
        flattened."""
        self.from_extension = inputs.from_extension
        if not inputs.from_extension:
            self.output_layouts = (
                _get_all_gather_output_layout(
                    padded_sharded_size, shard_dim, shard_world_size
                ),
            )
            return [cast(torch.Tensor, inp).view(-1) for inp in inputs.inputs]
        # Invalid record dims key as None, which the layouts check rejects
        key = tuple(
            [
                (
                    dim if type(dim := inp.dim) is int else _record_dim(dim),
                    inp.tensor.size(),
                    inp.tensor.dtype,
                )
                if isinstance(inp, AllGatherInput)
                else (inp.size(), inp.dtype)
                for inp in inputs.inputs
            ]
        )
        layouts = self._layouts_by_key.get(key)
        if layouts is None:
            layouts = _get_all_gather_output_layouts(
                inputs.inputs,
                self.all_gather_outputs,
                shard_world_size=shard_world_size,
                shard_dim=shard_dim,
                padded_sharded_size=padded_sharded_size,
                require_padded_size=inputs.check_padding and uneven,
                param_name=self.name,
            )
            if self.all_gather_outputs:
                # Bounded for payloads whose sizes keep changing
                if len(self._layouts_by_key) == 4:
                    self._layouts_by_key.clear()
                self._layouts_by_key[key] = layouts
        self.output_layouts = layouts
        return [
            (inp.tensor if isinstance(inp, AllGatherInput) else inp).view(-1)
            for inp in inputs.inputs
        ]

    def check_post_forward_reshard(
        self, global_shape: torch.Size, world_size: int, requires_grad: bool
    ) -> None:
        """Raise if a reshard to ``world_size`` ranks after forward cannot
        chunk these outputs."""
        if (num_outputs := len(self.all_gather_outputs)) != 1:
            raise NotImplementedError(
                f"Resharding {self.name} to smaller mesh is not supported for "
                f"all-gather extensions with {num_outputs} inputs yet"
            )
        if not self.from_extension:
            return
        # The post-forward parameter views a chunk of the extension's output. The
        # largest post-forward shard is the same bound on every rank, so all ranks
        # raise together.
        output = self.all_gather_outputs[0]
        rows = -(-global_shape[0] // world_size)
        shard_numel = rows * math.prod(global_shape[1:])
        if output.numel() // world_size < shard_numel:
            raise NotImplementedError(
                f"Resharding {self.name} to smaller mesh needs post-forward shards "
                f"of up to {shard_numel} elements, but its all-gather extension "
                f"output has {output.numel() // world_size} per rank"
            )
        if requires_grad and not (output.is_floating_point() or output.is_complex()):
            raise NotImplementedError(
                f"Resharding {self.name} to smaller mesh needs a floating or complex "
                "all-gather extension output to require gradients, but its output "
                f"has dtype {output.dtype}"
            )


class ParamDrivenAllGather:
    """A ``ParamDrivenShard`` bucket's all-gather: its parameters' outputs, its
    layout, and the persistent buffers a custom layout chose."""

    def __init__(
        self,
        representation: ParamRepresentation,
        num_params: int,
        device: torch.device,
    ) -> None:
        self.representation = representation
        self.device = device
        self.params = [ParamAllGather() for _ in range(num_params)]
        self.layout: AllGatherLayout = DEFAULT_ALL_GATHER_LAYOUT
        self.layout_buffers: _PersistentBuffers | None = None
        # As BucketRuntime.keep_persistent_storage: a rejected refill keeps the
        # storage it re-allocated
        self.keep_storage = False

    def reset(self) -> None:
        """Drop the outputs and unsharded params, e.g. for another layout."""
        self.params = [ParamAllGather(param.name) for param in self.params]
        self.layout_buffers = None


@dataclass
class _ParamDrivenUnshardState:
    infos: list[ParamInfo]
    # On one rank, each parameter's inputs are copied into its outputs instead
    single_rank_inputs: list[list[torch.Tensor]] | None
    split_sizes: list[int]
    outer_sizes: list[int]
    param_numels: list[list[int]]
    param_dtypes: list[list[torch.dtype]]
    layout: AllGatherLayout
    output_metadata: object | None
    comm: Any
    group: dist.ProcessGroup | None


class ParamDrivenShard(Shard):
    """``Shard`` for a bucket whose parameters define their own all-gather
    inputs and unsharded tensors (see the module docstring)."""

    refills_persistent_buffers_in_finish = True

    def __init__(self, dim: int, all_gather: ParamDrivenAllGather) -> None:
        super().__init__(dim)
        self.all_gather = all_gather

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ParamDrivenShard)
            and other.all_gather is self.all_gather
            and other.dim == self.dim
        )

    def __hash__(self) -> int:
        return hash((type(self), id(self.all_gather), self.dim))

    def bucket_compatibility_key(self) -> object:
        return (type(self), id(self.all_gather))

    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: Any,
        debug_fqn: str | None,
        *,
        comm: Any = None,
        device: torch.device | None = None,
    ) -> PlacementPreparedUnshard:
        all_gather = self.all_gather
        world_size, rank = mesh.size(), mesh.get_local_rank()
        param_inputs = []
        for index, (param, info) in enumerate(
            zip(all_gather.params, infos, strict=True)
        ):
            param.name = info.fqn
            shard_dim = cast(Shard, info.placements[0]).dim
            padded_sharded_size = list(info.global_shape)
            uneven = False
            if padded_sharded_size:
                uneven = padded_sharded_size[shard_dim] % world_size != 0
                padded_sharded_size[shard_dim] = -(
                    -padded_sharded_size[shard_dim] // world_size
                )
            param_inputs.append(
                param.flatten_inputs(
                    all_gather.representation.all_gather_inputs(index),
                    shard_world_size=world_size,
                    shard_dim=shard_dim,
                    padded_sharded_size=torch.Size(padded_sharded_size),
                    uneven=uneven,
                )
            )
        state = _ParamDrivenUnshardState(
            infos=list(infos),
            single_rank_inputs=param_inputs,
            split_sizes=[],
            outer_sizes=[],
            param_numels=[],
            param_dtypes=[],
            layout=all_gather.layout,
            output_metadata=None,
            comm=comm,
            group=mesh.get_group(),
        )
        if world_size == 1:
            return PlacementPreparedUnshard(
                placement=self, buffers=[], placement_state=state
            )
        state.single_rank_inputs = None
        state.outer_sizes = [
            layout.outer_size
            for param in all_gather.params
            for layout in param.output_layouts
        ]
        state.param_dtypes, state.param_numels, dtype = _get_all_gather_input_metadatas(
            param_inputs
        )
        inputs = [
            t.view(torch.uint8) if dtype == torch.uint8 else t
            for ts in param_inputs
            for t in ts
        ]
        state.split_sizes = [t.numel() for t in inputs]
        numel = sum(state.split_sizes)
        layout = all_gather.layout
        if isinstance(layout, DefaultAllGatherLayout):
            copy_in = layout.copy_in
        else:
            copy_in, state.layout, state.output_metadata = layout.prepare(
                AllGatherInputMetadata(
                    input_split_sizes=list(state.split_sizes),
                    input_outer_sizes=list(state.outer_sizes),
                    input_numel=numel,
                    world_size=world_size,
                    dtype=dtype,
                    device=all_gather.device,
                )
            )
        size = (numel * world_size,)
        output = (
            torch.empty(size, dtype=dtype, device=all_gather.device)
            if comm is None
            else comm.allocate(size, dtype=dtype, device=all_gather.device)
        )
        all_gather_input, output = copy_in(
            inputs, output, state.split_sizes, numel, rank
        )
        return PlacementPreparedUnshard(
            placement=self, buffers=[all_gather_input, output], placement_state=state
        )

    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        state = cast(_ParamDrivenUnshardState, prepared.placement_state)
        if state.single_rank_inputs is not None:
            return None
        all_gather_input, output = prepared.buffers
        if state.comm is None:
            return dist.all_gather_single(
                output, all_gather_input, group=state.group, async_op=async_op
            )
        kwargs = {}
        if state.layout.comm_type is not None:
            kwargs["output_metadata"] = state.output_metadata
        return state.comm(
            output_tensor=output,
            input_tensor=all_gather_input,
            group=state.group,
            async_op=async_op,
            **kwargs,
        )

    def finish_prepared_unshard(
        self, prepared: PlacementPreparedUnshard
    ) -> PlacementUnshardResult:
        state = cast(_ParamDrivenUnshardState, prepared.placement_state)
        all_gather = self.all_gather
        params = all_gather.params
        if state.single_rank_inputs is not None:
            self._copy_single_rank_inputs(state.single_rank_inputs)
            return self._finish_unsharded_params(prepared, state.infos)
        result = AllGatherResult(
            prepared.buffers[1],
            None,
            None,
            state.param_dtypes,
            state.param_numels,
            state.split_sizes,
            state.outer_sizes,
            state.layout,
            state.output_metadata,
            prepared.buffers[0],
        )
        try:
            all_gather.layout_buffers = foreach_all_gather_copy_out(
                result, params, state.group, all_gather.layout_buffers
            )
        except ValueError:
            # Free what a rejected refill re-allocated, as FSDP2 does
            if not all_gather.keep_storage:
                if all_gather.layout_buffers is not None:
                    all_gather.layout_buffers.free()
                for param in params:
                    param.free_all_gather_outputs()
            raise
        return self._finish_unsharded_params(prepared, state.infos)

    def _copy_single_rank_inputs(self, param_inputs: list[list[torch.Tensor]]) -> None:
        """As FSDP2 on one rank: copy the all-gather inputs into the outputs."""
        all_gather = self.all_gather
        for param, inputs in zip(all_gather.params, param_inputs, strict=True):
            if not param.all_gather_outputs:
                param.init_all_gather_outputs(
                    [t.numel() for t in inputs],
                    [t.dtype for t in inputs],
                    1,
                    all_gather.device,
                )
            for tensor, inp in zip(param.all_gather_outputs, inputs, strict=True):
                _alloc_storage(tensor)
                with (
                    torch.autograd._unsafe_preserve_version_counter(tensor)
                    if not tensor.is_inference()
                    else contextlib.nullcontext()
                ):
                    if tensor.dtype == inp.dtype and tensor.numel() != inp.numel():
                        tensor = tensor[: inp.numel()]
                    elif inp.dtype == torch.uint8:
                        tensor = tensor.view(torch.uint8)
                    tensor.copy_(inp)

    def _finish_unsharded_params(
        self, prepared: PlacementPreparedUnshard, infos: list[ParamInfo]
    ) -> PlacementUnshardResult:
        """Build each unsharded param from its outputs on the first unshard, as
        FSDP2 does, and let its representation rebuild it in place on later
        ones."""
        all_gather = self.all_gather
        representation = all_gather.representation
        for index, (param, info) in enumerate(
            zip(all_gather.params, infos, strict=True)
        ):
            outputs = param.unflatten_all_gather_outputs()
            if param.unsharded_param is not None:
                for tensor in param.inner_tensors:
                    if tensor.untyped_storage().size() == 0:
                        _alloc_storage(tensor)
                representation.post_all_gather(index, outputs, param.unsharded_param)
            else:
                built = representation.post_all_gather(index, outputs, None)
                if built is None:
                    # The output gives the unsharded data directly
                    if len(param.all_gather_outputs) != 1:
                        raise AssertionError(
                            "Expected 1 all_gather_output, got "
                            f"{len(param.all_gather_outputs)}"
                        )
                    tensor, inner_tensors = param.all_gather_outputs[0], []
                else:
                    tensor, inner_tensors = built[0], list(built[1])
                    if param.outputs_view_layout_buffers:
                        # Inner tensors that view the layout buffers are the
                        # bucket's to free
                        ptrs = {
                            output.untyped_storage().data_ptr()
                            for output in param.all_gather_outputs
                        }
                        inner_tensors = [
                            t
                            for t in inner_tensors
                            if t.untyped_storage().data_ptr() not in ptrs
                        ]
                param.inner_tensors = inner_tensors
                tensor = torch.as_strided(
                    tensor,
                    info.global_shape,
                    info.global_stride,
                    storage_offset=tensor.storage_offset(),
                )
                unsharded_param = nn.Parameter(
                    representation.unsharded_param(index, tensor),
                    requires_grad=info.requires_grad,
                )
                for name, value in info.param_attrs.items():
                    setattr(unsharded_param, name, value)
                unsharded_param.grad_dtype = info.unsharded_grad_dtype
                param.unsharded_param = unsharded_param
            if representation.releases_all_gather_outputs(index):
                param.free_all_gather_outputs()
        if prepared.persistent_buffers is not None:
            return PlacementUnshardResult(
                full_params=[], persistent_buffers=prepared.persistent_buffers
            )
        persistent: dict[int, torch.Tensor] = {}
        for param in all_gather.params:
            for tensor in (*param.all_gather_outputs, *param.inner_tensors):
                persistent.setdefault(id(tensor), tensor)
        if all_gather.layout_buffers is not None:
            for tensor in all_gather.layout_buffers.tensors:
                persistent.setdefault(id(tensor), tensor)
        return PlacementUnshardResult(
            full_params=[
                cast(nn.Parameter, param.unsharded_param) for param in all_gather.params
            ],
            persistent_buffers=list(persistent.values()) if prepared.persistent else [],
        )


def _record_dim(dim: Any) -> int | None:
    # Tensor dims are invalid, since converting them would sync in every lookup
    if isinstance(dim, torch.Tensor):
        return None
    try:
        return operator.index(dim)
    except TypeError:
        return None


def _get_all_gather_output_layout(
    input_size: torch.Size, dim: int, world_size: int
) -> _AllGatherOutputLayout:
    input_size = torch.Size(input_size or (1,))
    dim %= len(input_size)
    gathered_size = list(input_size)
    gathered_size[dim] *= world_size
    flat = world_size == 1 or input_size.numel() == 0
    outer_size = 1 if flat else math.prod(input_size[:dim])
    return _AllGatherOutputLayout(torch.Size(gathered_size), outer_size)


def _get_all_gather_output_layouts(
    inputs: Sequence[torch.Tensor | AllGatherInput],
    all_gather_outputs: Sequence[torch.Tensor],
    *,
    shard_world_size: int,
    shard_dim: int,
    padded_sharded_size: torch.Size,
    require_padded_size: bool,
    param_name: str,
) -> tuple[_AllGatherOutputLayout, ...]:
    """
    Returns the output layouts of the ``fsdp_pre_all_gather`` ``inputs``, which
    must fit the parameter's sharding and any allocated outputs.
    """
    if all_gather_outputs and len(all_gather_outputs) != len(inputs):
        raise ValueError(
            f"fsdp_pre_all_gather for {param_name} returned {len(inputs)} inputs, "
            f"but {len(all_gather_outputs)} on its first call"
        )
    # The copy-out reassembles the rows of Tensor inputs along a nonzero shard
    # dim, reading them as the padded sharded parameter's rows. A single rank
    # copies them flat, so there several rows only forbid shrinking.
    leading_dims = padded_sharded_size[:shard_dim]
    shard_outer_size = math.prod(leading_dims)
    reassembled = shard_world_size > 1 and shard_outer_size > 1
    layouts: list[_AllGatherOutputLayout] = []
    for i, inp in enumerate(inputs):
        is_record = isinstance(inp, AllGatherInput)
        tensor = inp.tensor if is_record else inp
        size = tensor.size()
        output = all_gather_outputs[i] if all_gather_outputs else None
        # Tensor inputs may be bytes of a typed output, which keep its size and layout
        byte_view = output is not None and tensor.dtype == torch.uint8 != output.dtype
        if is_record:
            ndim = max(len(size), 1)
            dim = _record_dim(inp.dim)
            if dim is None or not -ndim <= dim < ndim:
                raise ValueError(
                    f"AllGatherInput {i} of {param_name} has dim {inp.dim!r}, which "
                    f"is invalid for size {tuple(size)}"
                )
            layout = _get_all_gather_output_layout(size, dim, shard_world_size)
        else:
            if require_padded_size and size != padded_sharded_size:
                # NOTE: Inputs with the unpadded local shard's size pass on ranks
                # whose shard needs no padding, which then hang in the collective
                raise ValueError(
                    f"{param_name} is unevenly sharded, so its Tensor all-gather "
                    "inputs must have the padded sharded size "
                    f"{tuple(padded_sharded_size)}, but input {i} has size "
                    f"{tuple(size)}. Return other payloads as AllGatherInput records."
                )
            if (
                reassembled
                and size.numel()
                and not byte_view
                and size.numel() != padded_sharded_size.numel()
                and size[:shard_dim] != leading_dims
            ):
                raise ValueError(
                    f"Tensor all-gather input {i} of {param_name} is gathered along "
                    f"its Shard({shard_dim}) dim, so it must have the "
                    f"{padded_sharded_size.numel()} elements of the padded sharded "
                    f"size {tuple(padded_sharded_size)} or its dims "
                    f"{tuple(leading_dims)} before that dim, but has size "
                    f"{tuple(size)}. Return an AllGatherInput to gather it along "
                    "another dim."
                )
            # -1 cannot be inferred with a zero trailing dim
            leading_size = size[0] * shard_world_size if 0 in size[1:] else -1
            outer_size = shard_outer_size if reassembled and size.numel() else 1
            layout = _AllGatherOutputLayout(
                torch.Size((leading_size, *size[1:])), outer_size
            )
        layouts.append(layout)
        if output is None:
            continue
        if tensor.dtype != output.dtype and (is_record or not byte_view):
            raise ValueError(
                f"All-gather input {i} of {param_name} must keep the dtype "
                f"{output.dtype} of its first call, but has dtype {tensor.dtype}"
            )
        # Tensor inputs may shrink unless they are byte views, have several rows, or
        # have a zero trailing dim that fixes their output size
        exact = is_record or byte_view or shard_outer_size > 1 or 0 in size[1:]
        nbytes = tensor.nbytes * shard_world_size
        fits = nbytes == output.nbytes if exact else nbytes <= output.nbytes
        if not fits:
            raise ValueError(
                f"All-gather input {i} of {param_name} must "
                f"{'match' if exact else 'fit in'} the size of its first call, "
                f"{output.nbytes // shard_world_size} bytes per rank, but has "
                f"{tensor.nbytes} bytes of {tensor.dtype}"
            )
        # fsdp_post_all_gather receives the whole output viewed as (-1, *size[1:])
        trailing_numel = math.prod(size[1:])
        if not is_record and trailing_numel and output.numel() % trailing_numel:
            raise ValueError(
                f"All-gather input {i} of {param_name} has trailing dims "
                f"{tuple(size[1:])}, which do not divide the {output.numel()} "
                "elements of its output"
            )
    return tuple(layouts)
