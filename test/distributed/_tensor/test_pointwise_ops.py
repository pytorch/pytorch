# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from typing import Any, Callable, Dict, Optional, Sequence
from unittest import skip

import torch

import torch.utils._pytree as pytree
from torch import Tensor

from torch.distributed._tensor import DeviceMesh, distribute_tensor, DTensor
from torch.distributed._tensor.placement_types import (
    _Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.distributed.distributed_c10d import ReduceOp
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorOpTestBase,
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)
from torch.distributed._tensor.random import (
    _set_offset,
    get_rng_state,
    manual_seed_all,
)

def no_op():
    return None


def deepcopy_convert_to_dtensor(
    val: Any,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Any:
    """
    Recursively convert (over Sequence and Dict types) Tensors into DTensors.

    :param device_mesh: the DeviceMesh to use.
    :param placements: the Placement list to use.
    :return: the transformed structure.
    """

    def f(x):
        if isinstance(x, Tensor) and not isinstance(x, DTensor):
            return distribute_tensor(
                x,
                device_mesh=device_mesh,
                placements=placements,
            )
        return x

    return pytree.tree_map(f, [val])[0]


def deepcopy_convert_from_dtensor(val: Any) -> Any:
    """
    Recursive convert any DTensor to local Tensor.

    :param val: the structure to coerce.
    :return: the coerced structure.
    """

    def f(x):
        if isinstance(x, DTensor):
            return x.redistribute(
                device_mesh=x.device_mesh,
                placements=[Replicate()] * x.device_mesh.ndim,
            ).to_local()
        return x

    return pytree.tree_map(f, [val])[0]


class DistElementwiseOpsTest(DTensorOpTestBase):
    def _compare_pairwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        op: Callable,
        pre_op_fn: Optional[Callable] = None,
        args: Sequence[Any] = tuple(),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        if not kwargs:
            kwargs = {}

        dargs = deepcopy_convert_to_dtensor(
            args,
            device_mesh=device_mesh,
            placements=placements,
        )
        dkwargs = deepcopy_convert_to_dtensor(
            kwargs,
            device_mesh=device_mesh,
            placements=placements,
        )

        pre_op_fn()

        # run the reference first, in case the call is broken;
        # it's better to debug an incorrect call at this point.
        reference_result = op(*args, **kwargs)

        pre_op_fn()

        dist_result = op(*dargs, **dkwargs)

        collected_result = deepcopy_convert_from_dtensor(dist_result)

        self.assertEqualOnRank(reference_result, collected_result)

    # TODO: We need to add CPU tests for ops in the future.
    def _run_sharded_elementwise_ops(
        self,
        *,
        device_mesh: DeviceMesh,
        placements: Sequence[Placement],
        pre_op_fn: Optional[Callable] = None,
        input_size: Sequence[int],
        op: Callable,
        **kwargs,
    ):
        if pre_op_fn is None:
            pre_op_fn = no_op

        input_tensor = torch.randn(
            *input_size,
            device=self.device_type,
            requires_grad=True,
        )

        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            pre_op_fn=pre_op_fn,
            op=op,
            args=(input_tensor,),
            kwargs=kwargs,
        )

    def test_activations(self):
        device_mesh = self.build_device_mesh()
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.gelu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 12),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.nn.functional.relu,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.sigmoid,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Replicate()],
            input_size=(8, 5),
            op=torch.sigmoid,
        )

    @skip("testing RNG based ops is broken: https://github.com/pytorch/tau/issues/494")
    def test_dropout(self):
        device_mesh = self.build_device_mesh()

        def _reset_random_seed():
            torch.manual_seed(self.rank + 4)

        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(0)],
            input_size=(8, 5),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.4,
            training=False,
        )
        self._run_sharded_elementwise_ops(
            device_mesh=device_mesh,
            placements=[Shard(1)],
            input_size=(3, 14),
            op=torch.nn.functional.dropout,
            pre_op_fn=_reset_random_seed,
            p=0.5,
            training=True,
        )

    @skip_unless_torch_gpu
    def test_dropout_backward(self):
        device_mesh = self.build_device_mesh()
        placements = [Shard(0)]

        input_size = (8, 5)

        grad_output = torch.rand(
            input_size,
            device=self.device_type,
            requires_grad=True,
        )
        mask = (
            torch.rand(
                input_size,
                device=self.device_type,
                requires_grad=False,
            )
            < 0.8
        )

        self._compare_pairwise_ops(
            device_mesh=device_mesh,
            placements=placements,
            op=torch.ops.aten.native_dropout_backward,
            kwargs=dict(
                grad_output=grad_output,
                mask=mask,
                scale=0.3,
            ),
        )

    def test_dropout_errors(self):
        device_mesh = self.build_device_mesh()
        with self.assertRaisesRegex(RuntimeError, "supported"):
            self._run_sharded_elementwise_ops(
                device_mesh=device_mesh,
                placements=[_Partial(ReduceOp.SUM)],
                input_size=(8, 5),
                op=torch.nn.functional.dropout,
            )

    def test_mul_out(self):
        device_mesh = self.build_device_mesh()
        torch.manual_seed(self.rank)
        shard_spec = [Shard(0)]
        input_size = (8, 4)
        input_tensor = torch.randn(*input_size, device=self.device_type)
        dtensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)

        other_tensor = torch.randn(*input_size, device=self.device_type)
        other_dtensor = DTensor.from_local(other_tensor, device_mesh, shard_spec)

        output_tensor = torch.randn(*input_size, device=self.device_type)
        output_dtensor = DTensor.from_local(output_tensor, device_mesh, shard_spec)
        dt = torch.mul(dtensor, other_dtensor, out=output_dtensor)
        expected = torch.mul(input_tensor, other_tensor, out=output_tensor)
        self.assertEqual(input_tensor, dtensor.to_local())
        self.assertEqual(expected, dt.to_local())


class DistTensorRandomOpTest(DTensorTestBase):
    def check_rng_state(self, seed: int, offset: int, device_mesh: DeviceMesh) -> None:
        state = get_rng_state(device_mesh)
        seed_int64 = state[-16:-8].view(torch.int64)
        offset_int64 = state[-8:].view(torch.int64)
        self.assertEqual(seed_int64, torch.tensor([seed]))
        self.assertEqual(offset_int64, torch.tensor([offset]))

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_uniform_1d(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [4, 1]

        # initialize rng state
        manual_seed_all(1234, device_mesh)
        self.check_rng_state(1234, 0, device_mesh)
        _tensor = torch.empty(*size, device='cuda')
        dtensor = DTensor.from_local(_tensor, device_mesh, [Shard(1)])

        # get rng offset for checking correctness
        global_size = dtensor.numel()
        state = get_rng_state(device_mesh)
        offset = state[-8:].view(torch.int64)[0].item()
        offset_after_op = offset + global_size

        # random op call
        dtensor.uniform_(0, 1)

        # check rng offset is correctly synchroized after perform op 
        self.check_rng_state(1234, offset_after_op, device_mesh)

        dtensor = dtensor.redistribute(device_mesh, [Replicate()])
        local_tensor = dtensor.to_local()

        for shard_num in range(self.world_size):
            if self.rank == shard_num:
                self.assertEqual(local_tensor[:,shard_num], local_tensor[:,self.rank])
            else:
                self.assertNotEqual(local_tensor[:,shard_num], local_tensor[:,self.rank])

        dtensor.uniform_(0, 1)
        local_tensor = dtensor.to_local()
        tensor_list = [torch.empty_like(local_tensor) for i in range(self.world_size)]
        device_mesh.all_gather(tensor_list, local_tensor)
        # check if every rank generate the same random numbers
        for t in tensor_list:
            self.assertEqual(local_tensor, t)

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_uniform_nd(self):
        mesh = torch.arange(self.world_size).reshape(2, 2, -1)
        device_mesh = DeviceMesh(self.device_type, mesh)
        dtensor_size = [4 for l in mesh.size()]  # DTensor shape replace with self.world_size
        # initialize rng state
        manual_seed_all(1234, device_mesh)
        self.check_rng_state(1234, 0, device_mesh)

        placements_list = [  # this list of placements should be enough to cover
            [Shard(0), Shard(1), Shard(2)],
            [Shard(2), Shard(1), Shard(0)],
            [Shard(1), Replicate(), Shard(0)],
            [Replicate(), Replicate(), Replicate()],
        ]

        dim_map_list = [
            [0, 1, 2],
            [2, 1, 0],
            [2, 0, -1],
            [-1, -1, -1],
        ]

        coord = device_mesh.get_coordinate()
        assert coord is not None

        for (placements, dim_map) in zip(placements_list, dim_map_list):
            # shard shape:
            shard_shape = [
                mesh.size()[dim] if dim >= 0 else 1
                for dim in dim_map
            ]
            # shard coord:
            shard_coord = [
                coord[dim] if dim >= 0 else 0
                for dim in dim_map
            ]
            strides = [1]
            for l in shard_shape[:0:-1]:
                strides.append(strides[-1] * l)
            strides = strides[::-1]
            # shard idx:
            shard_idx = sum([x * y for x, y in zip(shard_coord, strides)])
            # compute local size
            local_tensor_size = [size // n_shards for size, n_shards in zip(dtensor_size, shard_shape)]
            _tensor = torch.empty(*local_tensor_size, device='cuda')
            dtensor = DTensor.from_local(_tensor, device_mesh, placements)
            self.assertEqual(dtensor._spec.dim_map, dim_map)

            # get rng offset for checking correctness
            global_size = dtensor.numel()
            state = get_rng_state(device_mesh)
            offset = state[-8:].view(torch.int64)[0].item()
            offset_after_op = offset + global_size

            # random op call
            dtensor.uniform_(0, 1)

            # check rng offset is correctly synchroized after perform op 
            self.check_rng_state(1234, offset_after_op, device_mesh)

            local_tensor = dtensor.to_local()
            dtensor = dtensor.redistribute(device_mesh, [Replicate(), Replicate(), Replicate()])
            local_tensor_gathered = dtensor.to_local()
            # generate shard's range on each dim
            shard_range_on_dim = [list(range(0, l+1, l // n)) for l, n in zip(dtensor_size, shard_shape)]
            shard_range_on_dim = [
                [
                    (dim_range[i],dim_range[i+1])
                    for i in range(len(dim_range)-1)
                ]
                for dim_range in shard_range_on_dim
            ]
            from itertools import product
            shard_range_comb = list(product(*shard_range_on_dim))
            shard_range_comb = [
                [
                    slice(*t) for t in shard_range
                ]
                for shard_range in shard_range_comb
            ]

            for idx in range(len(shard_range_comb)):
                slice_idx = shard_range_comb[idx]
                if idx == shard_idx:
                    self.assertEqual(local_tensor_gathered[slice_idx], local_tensor)
                else:
                    self.assertNotEqual(local_tensor_gathered[slice_idx], local_tensor)


if __name__ == "__main__":
    run_tests()
