import operator
from functools import reduce
from typing import cast, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard


class NonLinearReductionsBase:
    def __init__(self):
        self._input_dtensor: Optional[dtensor.DTensor] = None
        self._local_tensor: Optional[torch.Tensor] = None
        self._dim: Optional[int] = None
        self._device_mesh: Optional[dtensor.DeviceMesh] = None
        self._placements: Optional[tuple[dtensor.Placement, ...]] = None
        self._keepdim: bool = False
        self._expected_shape: torch.Size = torch.Size([])
        self._shard_mesh_dims: list[int] = []

    def __call__(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ):
        raise NotImplementedError

    def _compute_local_reduction(self, op_call: torch._ops.OpOverload):
        raise NotImplementedError

    def _gather_tensors(
        self, gather_dim: int, gathered_idxs: torch.Tensor, local_redux: torch.Tensor
    ):
        assert self._device_mesh is not None
        gathered_redux = local_redux
        for mesh_dim in self._shard_mesh_dims:
            gathered_redux = funcol.all_gather_tensor(
                gathered_redux,
                gather_dim=gather_dim,
                group=(self._device_mesh, mesh_dim),
            )
            gathered_idxs = funcol.all_gather_tensor(
                gathered_idxs,
                gather_dim=gather_dim,
                group=(self._device_mesh, mesh_dim),
            )
        return gathered_redux, gathered_idxs

    def _convert_to_global_idxs(self, local_idx: torch.Tensor):
        assert self._input_dtensor is not None
        assert self._device_mesh is not None
        assert self._local_tensor is not None
        assert self._placements is not None

        global_shape = self._input_dtensor.shape
        _, global_offset = compute_local_shape_and_global_offset(
            global_shape, self._device_mesh, self._placements
        )

        if self._dim is None:
            local_coord = torch.unravel_index(local_idx, self._local_tensor.shape)
            global_coord = torch.stack(local_coord)
            gather_dim = 0
            for i, offset in enumerate(global_offset):
                global_coord[i] += offset
            # compute with proper striding
            gathered_idxs = torch.tensor(
                0, device=self._local_tensor.device, dtype=torch.long
            )
            for i, coord in enumerate(global_coord):
                gathered_idxs += coord * reduce(operator.mul, global_shape[i + 1 :], 1)
        else:
            gather_dim = self._dim
            gathered_idxs = local_idx + global_offset[self._dim]
        return gather_dim, gathered_idxs

    def _set_arguments(self, args: tuple[object, ...]):
        self._input_dtensor = cast(dtensor.DTensor, args[0])
        self._local_tensor = self._input_dtensor.to_local()
        if not isinstance(self._input_dtensor, dtensor.DTensor):
            raise NotImplementedError
        if len(args) > 1:
            self._dim = cast(int, args[1])
        if len(args) > 2:
            self._keepdim = cast(bool, args[2])
        self._device_mesh = self._input_dtensor.device_mesh
        self._placements = self._input_dtensor.placements
        return

    def _check_placements(self):
        assert self._input_dtensor is not None
        assert self._device_mesh is not None
        assert self._placements is not None
        # check for partial placements and handle it as a replicate.
        if any(isinstance(p, Partial) for p in self._placements):
            target_placements = [
                Replicate() if isinstance(p, Partial) else p for p in self._placements
            ]
            self._input_dtensor = self._input_dtensor.redistribute(
                device_mesh=self._device_mesh, placements=target_placements
            )
            self._placements = self._input_dtensor.placements
        return

    def _get_expected_shape(self):
        assert self._local_tensor is not None
        input_shape = list(self._local_tensor.shape)
        if self._dim is None:
            self._expected_shape = (
                torch.Size([1] * len(input_shape)) if self._keepdim else torch.Size([])
            )
        elif self._keepdim:
            if input_shape:
                input_shape[self._dim] = 1
            self._expected_shape = torch.Size(input_shape)
        else:
            if input_shape:
                input_shape.pop(self._dim)
            self._expected_shape = torch.Size(input_shape)
        return

    def _collect_shard_mesh_dims(self):
        assert self._placements is not None
        assert self._local_tensor is not None
        for mesh_dim, p in enumerate(self._placements):
            if isinstance(p, Shard):
                if self._dim is None or p.dim == (
                    self._dim if self._dim >= 0 else self._local_tensor.ndim + self._dim
                ):
                    self._shard_mesh_dims.append(mesh_dim)
        return


class ArgMinMaxHandler(NonLinearReductionsBase):
    _REDUCTION_OPS = {
        torch.ops.aten.argmax.default: torch.max,
        torch.ops.aten.argmin.default: torch.min,
    }

    def __call__(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple["dtensor.DTensor", int] | tuple["dtensor.DTensor", int, bool],
        kwargs: dict[str, object],
    ):
        self._set_arguments(args)
        self._check_placements()
        op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(
            op_call, args, kwargs
        )
        dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
        output_sharding = op_info.output_sharding
        assert output_sharding is not None, "output sharding should not be None"
        if op_call not in self._REDUCTION_OPS:
            raise NotImplementedError(f"Unsupported reduction op: {op_call}")

        self._get_expected_shape()
        self._collect_shard_mesh_dims()
        local_redux, local_idx = self._compute_local_reduction(op_call)

        if not self._shard_mesh_dims:
            return dtensor.DTensor._op_dispatcher.wrap(
                local_idx.reshape(self._expected_shape), output_sharding.output_spec
            )

        gather_dim, gathered_idxs = self._convert_to_global_idxs(local_idx)
        gathered_redux, gather_idxs = self._gather_tensors(
            gather_dim, gathered_idxs, local_redux
        )
        # op_call here is argmin/argmax which returns indices only
        rank_winner = op_call(gathered_redux, self._dim, True)
        final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

        return dtensor.DTensor._op_dispatcher.wrap(
            final_idx.reshape(self._expected_shape), output_sharding.output_spec
        )

    def _compute_local_reduction(self, op_call: torch._ops.OpOverload):
        assert self._local_tensor is not None
        assert self._dim is not None
        if self._dim is None:
            local_idx = op_call(self._local_tensor)
            local_redux = self._local_tensor.flatten()[local_idx]
        else:
            val_op = self._REDUCTION_OPS[op_call]
            local_redux, local_idx = val_op(
                self._local_tensor, dim=self._dim, keepdim=True
            )
        return local_redux, local_idx


class MinMaxDimHandler(NonLinearReductionsBase):
    """Handler for aten.min.dim and aten.max.dim ops."""

    def __call__(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple["dtensor.DTensor", int] | tuple["dtensor.DTensor", int, bool],
        kwargs: dict[str, object],
    ):
        self._set_arguments(args)
        self._check_placements()
        op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(
            op_call, args, kwargs
        )
        dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
        output_sharding = op_info.output_sharding
        assert output_sharding is not None, "output sharding should not be None"

        self._get_expected_shape()
        self._collect_shard_mesh_dims()
        local_redux, local_idx = self._compute_local_reduction(op_call)

        if not self._shard_mesh_dims:
            return dtensor.DTensor._op_dispatcher.wrap(
                    (local_redux.reshape(self._expected_shape), local_idx.reshape(self._expected_shape)),
                    output_sharding.output_spec,
                )

        gather_dim, gathered_idxs = self._convert_to_global_idxs(local_idx)
        gathered_redux, gather_idxs = self._gather_tensors(
            gather_dim, gathered_idxs, local_redux
        )
        # The op_call here is min/max with dim which returns (values, indices)
        final_redux, rank_winner = op_call(gathered_redux, self._dim, True)
        final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

        return dtensor.DTensor._op_dispatcher.wrap(
                (final_redux.reshape(self._expected_shape), final_idx.reshape(self._expected_shape)),
                output_sharding.output_spec
            )
    
    def _compute_local_reduction(self, op_call: torch._ops.OpOverload):
        assert self._local_tensor is not None
        assert self._dim is not None
        # Will always have a dim for min and max operators for this case.
        return op_call(self._local_tensor, dim=self._dim, keepdim=True)
