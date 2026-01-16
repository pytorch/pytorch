# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.distributed.tensor._ops._math_ops import _NormPartial
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


DEVICE = "cpu"

# Placement validation test cases: (name, local_map_kwargs, expected_error_or_None)
PLACEMENT_VALIDATION_CASES = [
    (
        "out_placements_partial_max_rejected",
        {"out_placements": [Partial(reduce_op="max")], "track_variant_dims": True},
        "does not support",
    ),
    (
        "out_placements_norm_partial_rejected",
        {"out_placements": [_NormPartial(norm_type=2)], "track_variant_dims": True},
        "does not support",
    ),
    (
        "in_placements_partial_max_rejected",
        {
            "out_placements": [Replicate()],
            "in_placements": ([Partial(reduce_op="max")],),
            "track_variant_dims": True,
        },
        "does not support",
    ),
    # Unsupported placements allowed without variance tracking
    (
        "partial_max_allowed_without_tracking",
        {"out_placements": [Partial(reduce_op="max")], "track_variant_dims": False},
        None,
    ),
    # Supported placements should work with variance tracking
    (
        "shard_accepted",
        {"out_placements": [Shard(0)], "track_variant_dims": True},
        None,
    ),
    (
        "replicate_accepted",
        {"out_placements": [Replicate()], "track_variant_dims": True},
        None,
    ),
    (
        "partial_sum_accepted",
        {"out_placements": [Partial(reduce_op="sum")], "track_variant_dims": True},
        None,
    ),
]

# Input DTensor validation test cases: (name, placement, expected_error_or_None)
INPUT_DTENSOR_CASES = [
    ("input_partial_max_rejected", Partial(reduce_op="max"), "does not support"),
    ("input_shard_accepted", Shard(0), None),
    ("input_replicate_accepted", Replicate(), None),
    ("input_partial_sum_accepted", Partial(reduce_op="sum"), None),
]


@instantiate_parametrized_tests
class TestVarianceTrackingLocalMapPlacements(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @parametrize("case", PLACEMENT_VALIDATION_CASES, name_fn=lambda case: case[0])
    def test_placement_validation(self, case):
        """Parameterized test for placement validation at runtime."""
        name, kwargs, expected_error = case
        mesh = DeviceMesh(DEVICE, torch.arange(self.world_size), mesh_dim_names=("dp",))

        def identity(x):
            return x

        wrapped = local_map(identity, **kwargs)

        # Create a DTensor with valid placement to trigger runtime validation
        local_tensor = torch.randn(4, 4, device=DEVICE)
        dt = DTensor.from_local(local_tensor, mesh, [Replicate()])

        if expected_error is not None:
            with self.assertRaisesRegex(ValueError, expected_error):
                wrapped(dt)
        else:
            result = wrapped(dt)
            self.assertIsInstance(result, DTensor)

    @parametrize("case", INPUT_DTENSOR_CASES, name_fn=lambda case: case[0])
    def test_input_dtensor_placement_validation(self, case):
        """Parameterized test for input DTensor placement validation."""
        name, placement, expected_error = case
        mesh = DeviceMesh(DEVICE, torch.arange(self.world_size), mesh_dim_names=("dp",))

        def identity(x):
            return x

        wrapped = local_map(
            identity,
            out_placements=[Replicate()],
            track_variant_dims=True,
        )

        local_tensor = torch.randn(4, 4, device=DEVICE)
        dt = DTensor.from_local(local_tensor, mesh, [placement])

        if expected_error is not None:
            with self.assertRaisesRegex(ValueError, expected_error):
                wrapped(dt)
        else:
            result = wrapped(dt)
            self.assertIsInstance(result, DTensor)


if __name__ == "__main__":
    run_tests()
