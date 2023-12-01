# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn

import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Replicate, Shard, init_device_mesh
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
    NUM_DEVICES,
)


c10d_functional = torch.ops.c10d_functional

class TensorParallelStyleTest(DTensorTestBase):
    @property
    def world_size(self):
        return NUM_DEVICES

    def _1d_input_func_check(
        self,
        input_local_tensor,
        expected_local_tensor,
        func,
        error_msgs="device_mesh is not passed nor can be inferred",
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, error_msgs):
            dtensor = func(input_local_tensor)

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        # test 1: replicate local tensor
        dtensor = func(input_local_tensor, device_mesh)
        result = dtensor[0] if isinstance(dtensor, tuple) else dtensor
        self.assertEqual(expected_local_tensor, result.to_local())
        # test 2: replicate DTensor
        dtensor = func(dtensor)
        result = dtensor[0] if isinstance(dtensor, tuple) else dtensor
        self.assertEqual(expected_local_tensor, result.to_local())
        # test 3: replicate DTensor with DeviceMesh passed
        dtensor = func(dtensor, device_mesh)
        result = dtensor[0] if isinstance(dtensor, tuple) else dtensor
        self.assertEqual(expected_local_tensor, result.to_local())

    @with_comms
    def test_colwise_parallel_style(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        comm_mode = CommDebugMode()
        tensor = torch.rand(8, 16, device=self.device_type, requires_grad=True)
        model = nn.Linear(16, 16, device=self.device_type)

        default_col_parallel = ColwiseParallel()
        with comm_mode:
            colwise_mod = parallelize_module(deepcopy(model), mesh, default_col_parallel)
            out = colwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (8, 16 // self.world_size))
            # ensure no communication happened in fwd
            self.assertEqual(comm_mode.get_total_counts(), 0)

            out.sum().backward()
            # allreduce in bwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)

        sharded_col_parallel = ColwiseParallel(input_layouts=Shard(0))
        with comm_mode:
            colwise_mod = parallelize_module(deepcopy(model), mesh, sharded_col_parallel)
            out = colwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (8 * self.world_size, 16 // self.world_size))
            # allgather in fwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1)

            out.sum().backward()
            # reduce_scatter in bwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1)

    @with_comms
    def test_rowwise_parallel_style(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        comm_mode = CommDebugMode()
        tensor = torch.rand(8, 16, device=self.device_type, requires_grad=True)
        model = nn.Linear(16, 16, device=self.device_type)

        default_row_parallel = RowwiseParallel()
        with comm_mode:
            rowwise_mod = parallelize_module(deepcopy(model), mesh, default_row_parallel)
            out = rowwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (8, 16 // self.world_size))
            # ensure no communication happened in fwd
            self.assertEqual(comm_mode.get_total_counts(), 0)

            out.sum().backward()
            # allreduce in bwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)

        sharded_col_parallel = ColwiseParallel(input_layouts=Shard(0))
        with comm_mode:
            colwise_mod = parallelize_module(deepcopy(model), mesh, sharded_col_parallel)
            out = colwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (8 * self.world_size, 16 // self.world_size))
            # allgather in fwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1)

            out.sum().backward()
            # reduce_scatter in bwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1)

    @with_comms
    def test_prepare_module_input(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        gathered_tensors = [
            torch.empty_like(tensor) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_tensors, tensor)
        gathered_tensors = torch.cat(gathered_tensors, dim=0).contiguous()
        prepare_hook = PrepareModuleInput(input_layouts=[Shard(0)], output_layouts=[Replicate()])
        self._1d_input_func_check(
            [tensor],
            gathered_tensors,
            prepare_hook._prepare_input,
            error_msgs="No device mesh is currently active",
        )

    @with_comms
    def test_prepare_module_input_multiple_inputs(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x, y):
                return self.linear(x) + y

        test_mod = TestModule().to(self.device_type)
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        parallelize_module(
            test_mod,
            mesh,
            PrepareModuleInput(input_layouts=(Shard(0), None), output_layouts=(Replicate(), None))
        )
        output = test_mod(
            torch.randn(2, 8, device=self.device_type),
            torch.ones(self.world_size * 2, 8 // self.world_size, device=self.device_type)
        )
        self.assertEqual(output.shape, (self.world_size * 2, 8 // self.world_size))

    @with_comms
    def test_prepare_module_output(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        prepare_hook = PrepareModuleOutput(input_layouts=[Replicate()], output_layouts=[Shard(0)])
        output, dtensor, device_mesh = self._test_prepare_output(
            prepare_hook._prepare_output, [Replicate()]
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local())


if __name__ == "__main__":
    run_tests()
