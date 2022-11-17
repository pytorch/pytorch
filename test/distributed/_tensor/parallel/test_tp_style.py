# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
    NUM_DEVICES,
)
from torch.distributed._tensor import (
    distribute_tensor,
    DeviceMesh,
    Shard,
    Replicate,
)
from torch.distributed._tensor.parallel.style import (
    make_input_shard_1d,
    make_input_replicate_1d,
    make_output_shard_1d,
    make_output_replicate_1d,
    make_output_tensor,
)


class TensorParallelStyleTest(DTensorTestBase):
    @with_comms
    def test_make_input_replicate_1d(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        with self.assertRaisesRegex(
            RuntimeError, "device_mesh is not passed nor can be inferred"
        ):
            dtensor = make_input_replicate_1d(tensor)
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        with self.assertRaisesRegex(
            RuntimeError,
            "device_mesh has dims [0-9]+ but expcted to be 1 for input.",
        ):
            dtensor = make_input_replicate_1d(tensor, device_mesh)

        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        # test 1: replicate local tensor
        dtensor = make_input_replicate_1d(tensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())
        # test 2: replicate DTensor
        dtensor = make_input_replicate_1d(dtensor)
        self.assertEqual(tensor, dtensor.to_local())
        # test 3: replicate DTensor with DeviceMesh passed
        dtensor = make_input_replicate_1d(dtensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())

    @with_comms
    def test_make_input_shard_1d(self):
        tensor = torch.rand(8, 16, device=self.device_type)
        with self.assertRaisesRegex(
            RuntimeError, "device_mesh is not passed nor can be inferred"
        ):
            dtensor = make_input_shard_1d(tensor)
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        with self.assertRaisesRegex(
            RuntimeError,
            "device_mesh has dims [0-9]+ but expcted to be 1 for input.",
        ):
            dtensor = make_input_shard_1d(tensor, device_mesh)

        device_mesh = DeviceMesh(self.device_type, list(range(NUM_DEVICES)))
        # test 1: shard local Tensor on default dim
        dtensor = make_input_shard_1d(tensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())
        # test 2: reshard DTensor
        dtensor = make_input_shard_1d(dtensor)
        self.assertEqual(tensor, dtensor.to_local())
        # test 3: reshard DTensor with DeviceMesh passed
        dtensor = make_input_shard_1d(dtensor, device_mesh)
        self.assertEqual(tensor, dtensor.to_local())

    # Common logic for testing prepare output funcs
    def _test_prepare_output(
        self, func, spec, dim=None, device_mesh_input_none=False
    ):
        device_mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])
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
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()])
        )
        # test when input device_mesh is None.
        output, dtensor, device_mesh = self._test_prepare_output(
            make_output_replicate_1d, [Shard(0)], None, True
        )
        self.assertEqual(
            output, dtensor.redistribute(device_mesh, [Replicate()])
        )

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

    # Common logic for testing prepare output funcs errors.
    def _test_prepare_output_error(self, func):
        tensor = torch.rand(8, 16, device=self.device_type)
        device_mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        output = [dtensor]
        with self.assertRaisesRegex(
            AssertionError,
            f"Expect output of Tensor Parallel to be a DTensor, but found {type(output)}.",
        ):
            func(output, device_mesh)
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        with self.assertRaisesRegex(
            AssertionError,
            "device_mesh has dims 2 but expcted to be 1 for output.",
        ):
            func(dtensor, device_mesh)

    @with_comms
    def test_prepare_output_error(self):
        self._test_prepare_output_error(make_output_shard_1d)
        self._test_prepare_output_error(make_output_replicate_1d)
        self._test_prepare_output_error(make_output_tensor)


if __name__ == "__main__":
    run_tests()
