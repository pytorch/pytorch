# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed._tensor import (
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import _Partial
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    NUM_DEVICES,
    RMSNormPython,
    with_comms,
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
        colwise_mod = parallelize_module(deepcopy(model), mesh, default_col_parallel)
        with comm_mode:
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
        colwise_mod = parallelize_module(deepcopy(model), mesh, sharded_col_parallel)
        with comm_mode:
            out = colwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (8 * self.world_size, 16 // self.world_size))
            # allgather in fwd
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 1)

            out.sum().backward()
            # reduce_scatter in bwd
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 2)

    @with_comms
    def test_colwise_parallel_embedding(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        comm_mode = CommDebugMode()
        tensor = torch.arange(8, device=self.device_type).reshape(4, 2)
        model = nn.Embedding(16, 16, device=self.device_type)

        default_col_parallel = ColwiseParallel()
        colwise_mod = parallelize_module(deepcopy(model), mesh, default_col_parallel)
        with comm_mode:
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
        tensor = torch.rand(
            8, 16 // self.world_size, device=self.device_type, requires_grad=True
        )
        model = nn.Linear(16, 16, device=self.device_type)

        default_row_parallel = RowwiseParallel()
        rowwise_mod = parallelize_module(deepcopy(model), mesh, default_row_parallel)
        with comm_mode:
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
        rowwise_mod = parallelize_module(deepcopy(model), mesh, sharded_row_parallel)
        with comm_mode:
            out = rowwise_mod(tensor)
            # ensure output replicated
            self.assertEqual(out.shape, (8 // self.world_size, 16))
            # reduce_scatter in fwd
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 1)

            out.sum().backward()
            # allgather in bwd
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )
            self.assertEqual(comm_mode.get_total_counts(), 2)

    @with_comms
    def test_rowwise_parallel_embedding(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        comm_mode = CommDebugMode()
        tensor = torch.arange(8, device=self.device_type).reshape(4, 2)
        model = nn.Embedding(16, 16, device=self.device_type)

        rowwise_mod = parallelize_module(
            deepcopy(model), mesh, RowwiseParallel(input_layouts=Replicate())
        )
        with comm_mode:
            out = rowwise_mod(tensor)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (4, 2, 16))
            # ensure allreduce communication happened in fwd
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 1)

            out.sum().backward()
            # no comm in bwd
            self.assertEqual(comm_mode.get_total_counts(), 1)

        sharded_row_parallel = RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        )

        rowwise_mod = parallelize_module(deepcopy(model), mesh, sharded_row_parallel)

        inp_indices = torch.arange(8, device=self.device_type)
        with comm_mode:
            out = rowwise_mod(inp_indices)
            # ensure output shard on the last dim
            self.assertEqual(out.shape, (8, 16 // self.world_size))
            # reduce scatter in fwd
            self.assertEqual(comm_mode.get_total_counts(), 1)
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.reduce_scatter_tensor], 1
            )
            out.sum().backward()
            # allgather comm in bwd
            self.assertEqual(comm_mode.get_total_counts(), 2)
            self.assertEqual(
                comm_mode.get_comm_counts()[c10d_functional.all_gather_into_tensor], 1
            )

    @with_comms
    def test_prepare_module_input(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensor = torch.ones(2, 16, device=self.device_type)
        expected_tensor = torch.ones(2 * self.world_size, 16, device=self.device_type)
        prepare_inp_style = PrepareModuleInput(
            input_layouts=Shard(0), desired_input_layouts=Replicate()
        )

        model = nn.Identity()
        allgather_mod = parallelize_module(model, mesh, prepare_inp_style)
        output = allgather_mod(tensor).full_tensor()
        self.assertEqual(output, expected_tensor)

    @with_comms
    def test_prepare_module_input_multiple_inputs(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x, y):
                return self.linear(x) + y

        # Raise assertion error if input_layouts and desired_input_layouts do not have same length.
        test_mod = TestModule().to(self.device_type)
        with self.assertRaisesRegex(
            AssertionError,
            "input_layouts and desired_input_layouts should have same length!",
        ):
            prepare_inps_dimension_mismatch = PrepareModuleInput(
                input_layouts=Shard(0), desired_input_layouts=(Replicate(), None)
            )
        # Raise assertion error if module inputs and input_layouts do not have same length.
        prepare_inps_short_dimension = PrepareModuleInput(
            input_layouts=Shard(0), desired_input_layouts=Replicate()
        )
        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        parallelize_module(test_mod, mesh, prepare_inps_short_dimension)
        with self.assertRaisesRegex(
            ValueError, "module inputs and input_layouts should have same length!"
        ):
            output = test_mod(
                torch.randn(2, 8, device=self.device_type),
                torch.ones(
                    self.world_size * 2, 8 // self.world_size, device=self.device_type
                ),
            )

        test_mod = TestModule().to(self.device_type)
        prepare_inps = PrepareModuleInput(
            input_layouts=(Shard(0), None), desired_input_layouts=(Replicate(), None)
        )

        parallelize_module(test_mod.linear, mesh, ColwiseParallel())
        parallelize_module(test_mod, mesh, prepare_inps)
        output = test_mod(
            torch.randn(2, 8, device=self.device_type),
            torch.ones(
                self.world_size * 2, 8 // self.world_size, device=self.device_type
            ),
        )
        self.assertEqual(output.shape, (self.world_size * 2, 8 // self.world_size))

    @with_comms
    def test_prepare_module_kwargs_input(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        class TestKwargModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, x, *, y, z=2):
                return self.linear(x) + y + z

        test_mod = TestKwargModule().to(self.device_type)
        prepare_inps_simple = PrepareModuleInput(
            input_kwarg_layouts={"y": Shard(0)},
            desired_input_kwarg_layouts={"y": Replicate()},
        )
        parallelize_module(
            test_mod.linear, mesh, ColwiseParallel(use_local_output=False)
        )
        parallelize_module(test_mod, mesh, prepare_inps_simple)

        comm_mode = CommDebugMode()
        with comm_mode:
            output = test_mod(
                torch.randn(1 * self.world_size, 8, device=self.device_type),
                y=torch.ones(1, 8, device=self.device_type),
            )

        self.assertEqual(comm_mode.get_total_counts(), 1)
        self.assertEqual(output.shape, (1 * self.world_size, 8))

        class TestKwargOnlyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            def forward(self, *, x, y=2, z=None):
                return self.linear(x) + y + z

        test_kwonly_mod = TestKwargOnlyModule().to(self.device_type)
        prepare_inps_simple = PrepareModuleInput(
            input_kwarg_layouts={"x": Shard(0), "z": Shard(0)},
            desired_input_kwarg_layouts={"x": Replicate(), "z": Replicate()},
        )
        parallelize_module(
            test_kwonly_mod.linear, mesh, ColwiseParallel(use_local_output=False)
        )
        parallelize_module(test_kwonly_mod, mesh, prepare_inps_simple)

        with comm_mode:
            output = test_kwonly_mod(
                x=torch.randn(1, 8, device=self.device_type),
                z=torch.ones(1, 8, device=self.device_type),
            )

        self.assertEqual(comm_mode.get_total_counts(), 2)
        self.assertEqual(output.shape, (1 * self.world_size, 8))

        # test the case where x is a DTensor
        x_dt = DTensor.from_local(
            torch.randn(1, 8, device=self.device_type), mesh, [Shard(0)]
        )
        with comm_mode:
            output = test_kwonly_mod(
                x=x_dt, z=torch.ones(1, 8, device=self.device_type)
            )

        self.assertEqual(comm_mode.get_total_counts(), 2)
        self.assertEqual(output.shape, (1 * self.world_size, 8))

    @with_comms
    def test_prepare_module_output(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        tensor = torch.ones(8, 16, device=self.device_type)
        expected_tensor = torch.ones(8 // self.world_size, 16, device=self.device_type)
        prepare_out_style = PrepareModuleOutput(
            output_layouts=Replicate(), desired_output_layouts=Shard(0)
        )

        model = nn.Identity()
        chunk_mod = parallelize_module(model, mesh, prepare_out_style)
        output = chunk_mod(tensor)
        self.assertEqual(output, expected_tensor)

    @with_comms
    def test_sequence_parallel_style(self):
        mesh = init_device_mesh(self.device_type, (self.world_size,))

        comm_mode = CommDebugMode()
        batch, N, embedding_dim = 20, 8, 12

        global_input = torch.rand(
            batch,
            N * self.world_size,
            embedding_dim,
            device=self.device_type,
            requires_grad=True,
        )
        sharded_input = distribute_tensor(global_input, mesh, [Shard(1)])

        # test LayerNorm
        for elementwise_affine in [True, False]:
            norm = nn.LayerNorm(
                embedding_dim,
                elementwise_affine=elementwise_affine,
                device=self.device_type,
            )
            sp_norm = parallelize_module(deepcopy(norm), mesh, SequenceParallel())

            output = norm(global_input)
            output.sum().backward()

            with comm_mode:
                sharded_out = sp_norm(sharded_input)
                grad_out = torch.ones_like(sharded_out)
                sharded_out.backward(grad_out)
                self.assertIsInstance(sharded_out, DTensor)
                self.assertEqual(sharded_out.placements, (Shard(1),))
                self.assertEqual(comm_mode.get_total_counts(), 0)
                self.assertEqual(
                    comm_mode.get_comm_counts()[c10d_functional.all_reduce], 0
                )
                if elementwise_affine:
                    self.assertEqual(sp_norm.weight.grad.placements, (_Partial(),))
                    self.assertEqual(sp_norm.bias.grad.placements, (_Partial(),))

                self.assertEqual(sharded_out.full_tensor(), output)

        # test RMSNorm
        rmsnorm = RMSNormPython(embedding_dim).to(self.device_type)
        sp_rmsnorm = parallelize_module(deepcopy(rmsnorm), mesh, SequenceParallel())

        output = rmsnorm(global_input)
        output.sum().backward()

        with comm_mode:
            sharded_out = sp_rmsnorm(sharded_input)
            grad_out = torch.ones_like(sharded_out)
            sharded_out.backward(grad_out)
            self.assertIsInstance(sharded_out, DTensor)
            self.assertEqual(sharded_out.placements, (Shard(1),))
            self.assertEqual(sp_rmsnorm.weight.grad.placements, (_Partial(),))
            self.assertEqual(comm_mode.get_total_counts(), 0)
            self.assertEqual(comm_mode.get_comm_counts()[c10d_functional.all_reduce], 0)

            self.assertEqual(sharded_out.full_tensor(), output)

        # test dropout
        dropout = nn.Dropout(0.5).to(self.device_type)
        sp_dropout = parallelize_module(deepcopy(dropout), mesh, SequenceParallel())

        output = dropout(global_input)
        output.sum().backward()
        with comm_mode:
            sharded_out = sp_dropout(sharded_input)
            grad_out = torch.ones_like(sharded_out)
            sharded_out.backward(grad_out)
            self.assertIsInstance(sharded_out, DTensor)
            self.assertEqual(sharded_out.placements, (Shard(1),))
            self.assertEqual(comm_mode.get_total_counts(), 0)

        # test sharded on non-sequence dim input
        sharded_batch_input = distribute_tensor(global_input, mesh, [Shard(0)])
        rmsnorm = RMSNormPython(embedding_dim).to(self.device_type)
        sp_rmsnorm = parallelize_module(deepcopy(rmsnorm), mesh, SequenceParallel())

        with comm_mode:
            sharded_out = sp_rmsnorm(sharded_batch_input)
            grad_out = torch.ones_like(sharded_out)
            sharded_out.backward(grad_out)
            self.assertIsInstance(sharded_out, DTensor)
            # output still sharded on sequence dimension
            self.assertEqual(sharded_out.placements, (Shard(1),))
            self.assertEqual(sp_rmsnorm.weight.grad.placements, (_Partial(),))
            # communication happens in both fwd/bwd to redistribute input
            self.assertEqual(comm_mode.get_total_counts(), 2)


if __name__ == "__main__":
    run_tests()
