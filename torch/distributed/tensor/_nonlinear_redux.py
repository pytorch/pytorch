import operator
from abc import ABC, abstractmethod
from functools import reduce
from typing import cast, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._op_schema import OutputSharding
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


def register_argminmax_handler():
    def impl(
        op_call: torch._ops.OpOverload,
        args: tuple["dtensor.DTensor", int] | tuple["dtensor.DTensor", int, bool],
        kwargs: dict[str, object],
    ):
        # create a new handler every time this is called
        handler = ArgMinMaxHandler()
        return handler(op_call, args, kwargs)

    return impl


def register_minmax_handler():
    def impl(
        op_call: torch._ops.OpOverload,
        args: tuple["dtensor.DTensor", int] | tuple["dtensor.DTensor", int, bool],
        kwargs: dict[str, object],
    ):
        # create a new handler
        handler = MinMaxDimHandler()
        return handler(op_call, args, kwargs)

    return impl


class NonLinearReductionsBase(ABC):
    def __init__(self):
        self._dim: Optional[int] = None
        self._keepdim: bool = False
        self._shard_mesh_dims: list[int] = []
        self._op_call_repr: str = ""

    @abstractmethod
    def __call__(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ):
        raise NotImplementedError

    @abstractmethod
    def _compute_local_reduction(
        self, op_call: torch._ops.OpOverload, local_tensor: torch.Tensor
    ):
        raise NotImplementedError

    def _gather_tensors(
        self,
        gather_dim: int,
        gathered_idxs: torch.Tensor,
        local_redux: torch.Tensor,
        device_mesh: torch.distributed.device_mesh.DeviceMesh,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This method gather the min or max of the tensors and their corresponding indices.

        :param self:
        :param gather_dim: The dim to stack the collected min/max tensors.
        :type gather_dim: int
        :param gathered_idxs: The local tensor holding the corresponding indices that will eventually be filled.
        :type gathered_idxs: torch.Tensor
        :param local_redux: The local tensor holding the operator's value i.e. min/max
        :type local_redux: torch.Tensor
        :param device_mesh: Device mesh of the DTensor.
        :type device_mesh: torch.distributed.device_mesh.DeviceMesh
        :return: All gathered tensors, gathered_redux and gathered_idxs, of the reducing operator
        :rtype: tuple[Tensor, Tensor]
        """
        gathered_redux = local_redux
        for mesh_dim in self._shard_mesh_dims:
            gathered_redux = funcol.all_gather_tensor(
                gathered_redux,
                gather_dim=gather_dim,
                group=(device_mesh, mesh_dim),
            )
            gathered_idxs = funcol.all_gather_tensor(
                gathered_idxs,
                gather_dim=gather_dim,
                group=(device_mesh, mesh_dim),
            )
        return gathered_redux, gathered_idxs

    def _convert_to_global_idxs(
        self,
        local_idx: torch.Tensor,
        global_shape: torch.Size,
        device_mesh: torch.distributed.device_mesh.DeviceMesh,
        placements: tuple[Placement, ...],
    ) -> tuple[int, torch.Tensor]:
        local_shape, global_offset = compute_local_shape_and_global_offset(
            global_shape, device_mesh, placements
        )

        if self._dim is None:
            local_coord = torch.unravel_index(local_idx, local_shape)
            global_coord = torch.stack(local_coord)
            gather_dim = 0
            for i, offset in enumerate(global_offset):
                global_coord[i] += offset
            # compute with proper striding
            gathered_idxs = torch.tensor(0, device=local_idx.device, dtype=torch.long)
            for i, coord in enumerate(global_coord):
                gathered_idxs += coord * reduce(operator.mul, global_shape[i + 1 :], 1)
        else:
            gather_dim = self._dim
            gathered_idxs = local_idx + global_offset[self._dim]
        return gather_dim, gathered_idxs

    def _prep_arguments(
        self,
        op_call_repr: str,
        args: tuple[object, ...],
        kwargs: dict[str, object] | None,
    ):
        input_dtensor = cast(dtensor.DTensor, args[0])
        self._dim = None
        self._keepdim = False
        self._shard_mesh_dims = []
        self._op_call_repr = op_call_repr
        if not isinstance(input_dtensor, dtensor.DTensor):
            raise NotImplementedError
        if len(args) > 1:
            self._dim = cast(int, args[1])
        if len(args) > 2:
            self._keepdim = cast(bool, args[2])
        if kwargs:
            if "dim" in kwargs:
                self._dim = cast(int, kwargs["dim"])
            if "keepdim" in kwargs:
                self._keepdim = cast(bool, kwargs["keepdim"])
        device_mesh = input_dtensor.device_mesh
        placements = input_dtensor.placements

        # check for partial placements and handle it as a replicate.
        if any(isinstance(p, Partial) for p in placements):
            target_placements = [
                Replicate() if isinstance(p, Partial) else p for p in placements
            ]
            input_dtensor = input_dtensor.redistribute(
                device_mesh=device_mesh, placements=target_placements
            )
            placements = input_dtensor.placements
        local_tensor = input_dtensor.to_local()
        global_shape = input_dtensor.shape

        return local_tensor, global_shape, device_mesh, placements

    def _get_expected_shape(self, local_tensor: torch.Tensor) -> torch.Size:
        input_shape = list(local_tensor.shape)
        if self._dim is None:
            expected_shape = (
                torch.Size([1] * len(input_shape)) if self._keepdim else torch.Size([])
            )
        elif self._keepdim:
            if input_shape:
                input_shape[self._dim] = 1
            expected_shape = torch.Size(input_shape)
        else:
            if input_shape:
                input_shape.pop(self._dim)
            expected_shape = torch.Size(input_shape)

        return expected_shape

    def _collect_shard_mesh_dims(
        self, local_tensor: torch.Tensor, placements: tuple[Placement, ...]
    ) -> None:
        for mesh_dim, p in enumerate(placements):
            if isinstance(p, Shard):
                if self._dim is None or p.dim == (
                    self._dim if self._dim >= 0 else local_tensor.ndim + self._dim
                ):
                    self._shard_mesh_dims.append(mesh_dim)
            elif isinstance(p, _StridedShard):
                raise NotImplementedError(
                    f"{self._op_call_repr} does not support _StridedShard!"
                )
        return

    @staticmethod
    def _get_output_sharding(
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> OutputSharding:
        op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(
            op_call, args, kwargs
        )
        dtensor.DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
        output_sharding = op_info.output_sharding
        assert output_sharding is not None, "output sharding should not be None"
        return output_sharding


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
        if op_call not in self._REDUCTION_OPS:
            raise NotImplementedError(f"Unsupported reduction op: {op_call}")

        local_tensor, global_shape, device_mesh, placements = self._prep_arguments(
            str(op_call), args, kwargs
        )
        output_sharding = self._get_output_sharding(op_call, args, kwargs)

        expected_shape = self._get_expected_shape(local_tensor)
        self._collect_shard_mesh_dims(local_tensor, placements)
        local_redux, local_idx = self._compute_local_reduction(op_call, local_tensor)

        if not self._shard_mesh_dims:
            return dtensor.DTensor._op_dispatcher.wrap(
                local_idx.reshape(expected_shape), output_sharding.output_spec
            )

        gather_dim, gathered_idxs = self._convert_to_global_idxs(
            local_idx, global_shape, device_mesh, placements
        )
        gathered_redux, gather_idxs = self._gather_tensors(
            gather_dim, gathered_idxs, local_redux, device_mesh
        )
        # op_call here is argmin/argmax which returns indices only
        rank_winner = op_call(gathered_redux, self._dim, True)
        final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

        return dtensor.DTensor._op_dispatcher.wrap(
            final_idx.reshape(expected_shape), output_sharding.output_spec
        )

    def _compute_local_reduction(
        self, op_call: torch._ops.OpOverload, local_tensor: torch.Tensor
    ):
        if self._dim is None:
            local_idx = op_call(local_tensor)
            local_redux = local_tensor.flatten()[local_idx]
        else:
            val_op = self._REDUCTION_OPS[op_call]
            local_redux, local_idx = val_op(local_tensor, dim=self._dim, keepdim=True)
        return local_redux, local_idx


class MinMaxDimHandler(NonLinearReductionsBase):
    """Handler for aten.min.dim and aten.max.dim ops."""

    def __call__(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple["dtensor.DTensor", int] | tuple["dtensor.DTensor", int, bool],
        kwargs: dict[str, object],
    ):
        local_tensor, global_shape, device_mesh, placements = self._prep_arguments(
            str(op_call), args, kwargs
        )
        output_sharding = self._get_output_sharding(op_call, args, kwargs)

        expected_shape = self._get_expected_shape(local_tensor)
        self._collect_shard_mesh_dims(local_tensor, placements)
        local_redux, local_idx = self._compute_local_reduction(op_call, local_tensor)

        if not self._shard_mesh_dims:
            return dtensor.DTensor._op_dispatcher.wrap(
                (
                    local_redux.reshape(expected_shape),
                    local_idx.reshape(expected_shape),
                ),
                output_sharding.output_spec,
            )

        gather_dim, gathered_idxs = self._convert_to_global_idxs(
            local_idx, global_shape, device_mesh, placements
        )

        gathered_redux, gather_idxs = self._gather_tensors(
            gather_dim, gathered_idxs, local_redux, device_mesh
        )
        # The op_call here is min/max with dim which returns (values, indices)
        final_redux, rank_winner = op_call(gathered_redux, self._dim, True)
        final_idx = torch.gather(gather_idxs, dim=gather_dim, index=rank_winner)

        return dtensor.DTensor._op_dispatcher.wrap(
            (
                final_redux.reshape(expected_shape),
                final_idx.reshape(expected_shape),
            ),
            output_sharding.output_spec,
        )

    def _compute_local_reduction(
        self, op_call: torch._ops.OpOverload, local_tensor: torch.Tensor
    ):
        assert self._dim is not None
        # Will always have a dim for min and max operators for this case.
        return op_call(local_tensor, dim=self._dim, keepdim=True)
