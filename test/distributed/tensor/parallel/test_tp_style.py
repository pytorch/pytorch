# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn

from torch.distributed._tensor import Replicate, Shard, init_device_mesh
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
            self.assertEqual(comm_mode.get_total_counts(), 1)

        sharded_col_parallel = ColwiseParallel(input_layouts=Shard(0))
        with comm_mode:
            colwise_mod = parallelize_module(deepcopy(model), mesh, sharded_col_parallel)
            out = colwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (8 * self.world_size, 16 // self.world_size))
            # allgather in fwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1)
            self.assertEqual(comm_mode.get_total_counts(), 1)

            out.sum().backward()
            # reduce_scatter in bwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1)
            self.assertEqual(comm_mode.get_total_counts(), 2)

    @with_comms
    def test_colwise_parallel_embedding(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        comm_mode = CommDebugMode()
        tensor = torch.arange(8, device=self.device_type).reshape(4, 2)
        model = nn.Embedding(16, 16, device=self.device_type)

        default_col_parallel = ColwiseParallel()
        with comm_mode:
            colwise_mod = parallelize_module(deepcopy(model), mesh, default_col_parallel)
            out = colwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (4, 2, 16 // self.world_size))
            # ensure no communication happened in fwd
            self.assertEqual(comm_mode.get_total_counts(), 0)

            out.sum().backward()
            # no comm in bwd
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_rowwise_parallel_style(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        comm_mode = CommDebugMode()
        tensor = torch.rand(8, 16 // self.world_size, device=self.device_type, requires_grad=True)
        model = nn.Linear(16, 16, device=self.device_type)

        default_row_parallel = RowwiseParallel()
        with comm_mode:
            rowwise_mod = parallelize_module(deepcopy(model), mesh, default_row_parallel)
            out = rowwise_mod(tensor)
            # ensure output replicated
            self.assertEqual(out.shape, (8, 16))
            # allreduce in fwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)
            self.assertEqual(comm_mode.get_total_counts(), 1)

            out.sum().backward()
            # no op in bwd
            self.assertEqual(comm_mode.get_total_counts(), 1)

        sharded_row_parallel = RowwiseParallel(output_layouts=Shard(0))
        with comm_mode:
            rowwise_mod = parallelize_module(deepcopy(model), mesh, sharded_row_parallel)
            out = rowwise_mod(tensor)
            # ensure output replicated
            self.assertEqual(out.shape, (8 // self.world_size, 16))
            # reduce_scatter in fwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1)
            self.assertEqual(comm_mode.get_total_counts(), 1)

            out.sum().backward()
            # allgather in bwd
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1)
            self.assertEqual(comm_mode.get_total_counts(), 2)

    @with_comms
    def test_prepare_module_input(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensor = torch.ones(2, 16, device=self.device_type)
        expected_tensor = torch.ones(2 * self.world_size, 16, device=self.device_type)
        prepare_inp_style = PrepareModuleInput(input_layouts=Shard(0), desired_input_layouts=Replicate())

        model = nn.Identity()
        allgather_mod = parallelize_module(model, mesh, prepare_inp_style)
        output = allgather_mod(tensor).full_tensor()
        self.assertEqual(output, expected_tensor)


    @with_comms
    def test_prepare_module_input_multiple_inputs(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x, y):
                return self.linear(x) + y

        # Raise assertion error if input_layouts and desired_input_layouts do not have same length.
        test_mod = TestModule().to(self.device_type)
        with self.assertRaisesRegex(AssertionError, "input_layouts and desired_input_layouts should have same length!"):
            prepare_inps_dimension_mismatch = PrepareModuleInput(input_layouts=Shard(0), desired_input_layouts=(Replicate(), None))
        # Raise assertion error if module inputs and input_layouts do not have same length.
        prepare_inps_short_dimension = PrepareModuleInput(input_layouts=Shard(0), desired_input_layouts=Replicate())
        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        parallelize_module(test_mod, mesh, prepare_inps_short_dimension)
        with self.assertRaisesRegex(ValueError, "module inputs and input_layouts should have same length!"):
            output = test_mod(
                torch.randn(2, 8, device=self.device_type),
                torch.ones(self.world_size * 2, 8 // self.world_size, device=self.device_type)
            )

        test_mod = TestModule().to(self.device_type)
        prepare_inps = PrepareModuleInput(input_layouts=(Shard(0), None), desired_input_layouts=(Replicate(), None))

        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        parallelize_module(test_mod, mesh, prepare_inps)
        output = test_mod(
            torch.randn(2, 8, device=self.device_type),
            torch.ones(self.world_size * 2, 8 // self.world_size, device=self.device_type)
        )
        self.assertEqual(output.shape, (self.world_size * 2, 8 // self.world_size))

    @with_comms
    def test_prepare_module_output(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensor = torch.ones(8, 16, device=self.device_type)
        expected_tensor = torch.ones(8 // self.world_size, 16, device=self.device_type)
        prepare_out_style = PrepareModuleOutput(output_layouts=Replicate(), desired_output_layouts=Shard(0))

        model = nn.Identity()
        chunk_mod = parallelize_module(model, mesh, prepare_out_style)
        output = chunk_mod(tensor)
        self.assertEqual(output, expected_tensor)


if __name__ == "__main__":
    run_tests()
