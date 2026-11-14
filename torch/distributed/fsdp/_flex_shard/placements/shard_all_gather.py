# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The all-gather state of ``Shard`` buckets.

Each parameter of a ``Shard`` bucket contributes all-gather inputs: by default
its padded local shard, read where the bucket stores it, or, for a parameter
whose frontend defines them (FSDP2's all-gather extensions,
``fsdp_pre_all_gather``), the inputs that its ``ParamRepresentation`` returns.
Each parameter keeps its all-gather outputs (``ParamAllGather``), which back
its unsharded tensor: the output itself, or the tensor that the representation
builds from the outputs (``fsdp_post_all_gather``). A frontend that keeps a
bucket's ``ShardAllGather`` across unshards, as ``fully_shard`` does, gets its
outputs, layout buffers and output layouts kept; otherwise each unshard
starts one, whose outputs are the bucket's persistent buffers.
"""

from __future__ import annotations

import contextlib
import math
import operator
from dataclasses import dataclass
from typing import Any, cast, NamedTuple, TYPE_CHECKING

import torch

from ..all_gather_layout import (
    _PersistentBuffers,
    AllGatherInput,
    AllGatherLayout,
    DEFAULT_ALL_GATHER_LAYOUT,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..bucket_runtime import ParamRepresentation
    from ..param_homing import ParamHoming


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
        # Whether the unsharded param drops the padding inside the rows of an
        # unevenly sharded Shard(i), i > 0, parameter without an all-gather
        # extension, so that it has its unpadded numel. The default layout's
        # output drops it (see init_all_gather_outputs); a custom layout's
        # keeps it, as for other params, and copy_out_unpadded copies the data
        # into unpadded_output, which then backs the unsharded param.
        self.drops_row_padding = False
        self.outputs_keep_row_padding = False
        self.unpadded_output: torch.Tensor | None = None
        self.shard_dim = 0
        self.unpadded_numel = 0
        # Extension layouts checked against the allocated outputs, which never
        # change, keyed by each input's size, dtype and, for records, integer dim
        self._layouts_by_key: dict[Any, tuple[_AllGatherOutputLayout, ...]] = {}
        # Tensors that hold the unsharded tensor's data besides the outputs, as
        # the representation's post_all_gather returned them
        self.inner_tensors: list[torch.Tensor] = []
        # The (world size, custom layout) for which use_local_shard recorded
        # the local shard's output layout, None after flatten_inputs
        self._local_shard_key: tuple[int, bool] | None = None

    def init_all_gather_outputs(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[torch.dtype],
        world_size: int,
        device: torch.device,
    ) -> None:
        if self.all_gather_outputs:
            return  # already initialized
        output_numels = [numel * world_size for numel in all_gather_input_numels]
        if self.drops_row_padding and not self.outputs_keep_row_padding:
            # The copy-out drops the padding (see _default_all_gather_output_fn),
            # while extensions receive it, as for Shard(0)
            output_numels = [self.unpadded_numel]
        self.all_gather_outputs = [
            torch.empty(torch.Size([numel]), dtype=dtype, device=device)
            for numel, dtype in zip(output_numels, all_gather_input_dtypes)
        ]

    def alloc_all_gather_outputs(self) -> None:
        if self.outputs_view_layout_buffers:
            return
        for tensor in self.all_gather_outputs:
            _alloc_storage(tensor)

    def free_all_gather_outputs(self) -> None:
        if self.unpadded_output is not None:
            _free_storage(self.unpadded_output)
        if self.outputs_view_layout_buffers:
            return
        for tensor in self.all_gather_outputs:
            _free_storage(tensor)

    @property
    def unsharded_outputs(self) -> tuple[torch.Tensor, ...]:
        """The outputs that the unsharded param is built from: an extension's
        in the layouts of its inputs, else the one output with the data."""
        if self.from_extension:
            return self.unflatten_all_gather_outputs()
        if self.outputs_keep_row_padding:
            return (cast(torch.Tensor, self.unpadded_output),)
        return tuple(self.all_gather_outputs)

    @torch.no_grad()
    def copy_out_unpadded(self) -> None:
        """Copy the data of the output that keeps the row padding into
        ``unpadded_output``: each row of the gathered param holds its data
        first, before the padding of the short chunk and the empty ranks'
        chunks (see _default_all_gather_output_fn)."""
        (output,) = self.all_gather_outputs
        if self.unpadded_output is None:
            self.unpadded_output = output.new_empty(self.unpadded_numel)
        unpadded = self.unpadded_output
        _alloc_storage(unpadded)
        rows = unpadded.view(self.output_layouts[0].outer_size, -1)
        # Refills write into the storage behind the unsharded param
        with (
            torch.autograd._unsafe_preserve_version_counter(unpadded)
            if not unpadded.is_inference()
            else contextlib.nullcontext()
        ):
            rows.copy_(output.view(rows.size(0), -1)[:, : rows.size(1)])

    def unflatten_all_gather_outputs(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.view(*layout.output_size)
            for tensor, layout in zip(self.all_gather_outputs, self.output_layouts)
        )

    def _record_sharding(
        self,
        *,
        from_extension: bool,
        shard_dim: int,
        global_shape: torch.Size,
        uneven: bool,
        custom_layout: bool,
    ) -> None:
        self.from_extension = from_extension
        self.drops_row_padding = shard_dim != 0 and uneven and not from_extension
        self.outputs_keep_row_padding = self.drops_row_padding and custom_layout
        self.shard_dim = shard_dim
        self.unpadded_numel = global_shape.numel()

    def use_local_shard(
        self,
        *,
        shard_world_size: int,
        shard_dim: int,
        global_shape: torch.Size,
        custom_layout: bool,
    ) -> None:
        """Record the output layout of the parameter's padded local shard,
        which the unshard all-gathers when its representation gives no
        inputs."""
        key = (shard_world_size, custom_layout)
        if self._local_shard_key == key:
            return
        padded_sharded_size, uneven = _padded_sharded_size(
            global_shape, shard_dim, shard_world_size
        )
        self._record_sharding(
            from_extension=False,
            shard_dim=shard_dim,
            global_shape=global_shape,
            uneven=uneven,
            custom_layout=custom_layout,
        )
        self.output_layouts = (
            _get_all_gather_output_layout(
                padded_sharded_size, shard_dim, shard_world_size
            ),
        )
        self._local_shard_key = key

    def flatten_inputs(
        self,
        inputs: ParamAllGatherInputs,
        *,
        shard_world_size: int,
        shard_dim: int,
        global_shape: torch.Size,
        custom_layout: bool,
    ) -> list[torch.Tensor]:
        """Record the output layouts of ``inputs``, which must fit the
        parameter's sharding and any allocated outputs, and return them
        flattened."""
        padded_sharded_size, uneven = _padded_sharded_size(
            global_shape, shard_dim, shard_world_size
        )
        self._record_sharding(
            from_extension=inputs.from_extension,
            shard_dim=shard_dim,
            global_shape=global_shape,
            uneven=uneven,
            custom_layout=custom_layout,
        )
        self._local_shard_key = None
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


class ShardAllGather:
    """A ``Shard`` bucket's all-gather: its parameters' outputs, its
    ``AllGatherLayout``, and the persistent buffers that a custom layout
    chose. ``homing``, the bucket's parameter storage, may let parameters
    define their inputs and unsharded tensors (all-gather extensions, tensor
    subclasses); ``representation`` exposes the unsharded tensors."""

    def __init__(
        self,
        representation: ParamRepresentation,
        homing: ParamHoming | None,
        num_params: int,
    ) -> None:
        self.representation = representation
        self.homing = homing
        self.params = [ParamAllGather() for _ in range(num_params)]
        self.layout: AllGatherLayout = DEFAULT_ALL_GATHER_LAYOUT
        self.layout_buffers: _PersistentBuffers | None = None
        # As BucketRuntime.keep_persistent_storage: a rejected refill keeps the
        # storage it re-allocated
        self.keep_storage = False
        # The (world size, custom layout) for which every param last recorded
        # its padded local shard (ParamAllGather.use_local_shard)
        self.local_shards_key: tuple[int, bool] | None = None

    def adopt_outputs(self, persistent_buffers: list[torch.Tensor]) -> None:
        """Take the persistent buffers of a bucket whose frontend keeps no
        all-gather as the outputs that a refill fills, one per parameter."""
        for param, buffer in zip(self.params, persistent_buffers, strict=True):
            param.all_gather_outputs = [buffer]

    def reset(self) -> None:
        """Drop the outputs, e.g. for another layout."""
        self.params = [ParamAllGather(param.name) for param in self.params]
        self.layout_buffers = None
        self.local_shards_key = None


def _padded_sharded_size(
    global_shape: torch.Size, shard_dim: int, world_size: int
) -> tuple[torch.Size, bool]:
    """The size of the first (largest) chunk, which every rank's slot holds,
    and whether the chunks are uneven."""
    size = list(global_shape)
    if not size:
        return global_shape, False
    uneven = size[shard_dim] % world_size != 0
    size[shard_dim] = -(-size[shard_dim] // world_size)
    return torch.Size(size), uneven


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
