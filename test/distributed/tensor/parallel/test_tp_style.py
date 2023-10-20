# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed.tensor.parallel.style import (
    ColwiseParallel,
    make_input_replicate_1d,
    make_input_reshard_replicate,
    make_input_shard_1d,
    make_output_replicate_1d,
    make_output_reshard_tensor,
    make_output_shard_1d,
    make_output_tensor,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class TensorParallelStyleTest(DTensorTestBase):
    @property
    def world_size(self):
        gpu_num = torch.cuda.device_count()
        return gpu_num if gpu_num % 2 == 0 and gpu_num > 4 else 4

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
    def test_make_input_replicate_1d(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        self._1d_input_func_check(tensor, tensor, make_input_replicate_1d)

    @with_comms
    def test_make_input_shard_1d(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        self._1d_input_func_check(tensor, tensor, make_input_shard_1d)

    @with_comms
    def test_make_input_reshard_replicate(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        gathered_tensor = [
            torch.empty(8, 16, device=self.device_type) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor)
        self._1d_input_func_check(tensor, gathered_tensor, make_input_reshard_replicate)

    # Common logic for testing prepare output funcs
    def _test_prepare_output(self, func, spec, dim=None, device_mesh_input_none=False):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        tensor = torch.rand(8, 16, device=self.device_type)
        dtensor = distribute_tensor(tensor, device_mesh, spec)
        device_mesh_input = None if device_mesh_input_none else device_mesh
        if dim is not None:
            output = func(dtensor, device_mesh_input, dim)
        else:
            output = func(dtensor, device_mesh_input)
        return output, dtensor, device_mesh

    @with_comms
    def test_make_output_shard_1d(self):
        # test when output is sharded.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_shard_1d, [Shard(0)], 1
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(1)]))
        #  test when output is replicated.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_shard_1d, [Replicate()], 0
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]))
        # test when input device_mesh is None.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_shard_1d, [Shard(0)], 1, True
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(1)]))

    @with_comms
    def test_make_output_replicate_1d(self):
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_replicate_1d, [Shard(0)]
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]))
        # test when input device_mesh is None.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_replicate_1d, [Shard(0)], None, True
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Replicate()]))

    @with_comms
    def test_make_output_tensor(self):
        # test when output is sharded.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_tensor, [Shard(0)]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()]).to_local()
        )
        #  test when output is replicated.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_tensor, [Replicate()]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()]).to_local()
        )
        # test when input device_mesh is None.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_tensor, [Shard(0)], None, True
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()]).to_local()
        )

    @with_comms
    def test_make_output_reshard_tensor(self):
        # test when output is sharded.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_reshard_tensor, [Shard(0)]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local()
        )
        #  test when output is replicated.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_reshard_tensor, [Replicate()]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local()
        )
        # test when input device_mesh is None.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_reshard_tensor, [Shard(0)], None, True
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local()
        )

    # Common logic for testing prepare output funcs errors.
    def _test_prepare_output_error(self, func):
        tensor = torch.rand(8, 16, device=self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        output = [dtensor]
        with self.assertRaisesRegex(
            AssertionError,
            "Expect output of Tensor Parallel to be a DTensor, but found"
            f" {type(output)}.",
        ):
            func(output, device_mesh)
        device_mesh = DeviceMesh(
            self.device_type,
            torch.arange(self.world_size).reshape(self.world_size // 2, 2),
        )
        with self.assertRaisesRegex(
            AssertionError,
            "device_mesh has dims 2 but expected to be 1 for output.",
        ):
            func(dtensor, device_mesh)

    def _test_prepare_output_error_new(self, func):
        tensor = torch.rand(8, 16, device=self.device_type)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        output = [dtensor]
        with self.assertRaisesRegex(
            RuntimeError,
            "Tensor parallel module expects DTensor or tensor"
            f" when layout specified but received {type(output)}!",
        ):
            func(output, device_mesh)

    @with_comms
    def test_prepare_output_error(self):
        self._test_prepare_output_error(make_output_shard_1d)
        self._test_prepare_output_error(make_output_replicate_1d)
        self._test_prepare_output_error(make_output_tensor)

    @with_comms
    def test_rowwise_parallel_style(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        rs = RowwiseParallel()
        self._1d_input_func_check(
            [tensor],
            tensor,
            rs._prepare_input,
            error_msgs="No device mesh is currently active",
        )
        # TODO: change output test
        output, dtensor, device_mesh = self._test_prepare_output(
            rs._prepare_output, [Shard(0)]
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()]).to_local()
        )
        # test when input device_mesh is None.
        output, dtensor, device_mesh = self._test_prepare_output(
            rs._prepare_output, [Shard(0)], None, True
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()]).to_local()
        )
        self._test_prepare_output_error_new(rs._prepare_output)

    @with_comms
    def test_colwise_parallel_style(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        cs = ColwiseParallel()
        self._1d_input_func_check(
            [tensor],
            tensor,
            cs._prepare_input,
            error_msgs="No device mesh is currently active",
        )
        output, dtensor, device_mesh = self._test_prepare_output(
            cs._prepare_output, [Shard(-1)]
        )
        self.assertEqual(output, dtensor.to_local())

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
    def test_prepare_module_output(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        prepare_hook = PrepareModuleOutput(input_layouts=[Replicate()], output_layouts=[Shard(0)])
        output, dtensor, device_mesh = self._test_prepare_output(
            prepare_hook._prepare_output, [Replicate()]
        )
        self.assertEqual(output, dtensor.redistribute(device_mesh, [Shard(0)]).to_local())


if __name__ == "__main__":
    run_tests()
