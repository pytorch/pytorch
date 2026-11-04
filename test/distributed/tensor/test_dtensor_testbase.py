# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import sys
import unittest
from unittest.mock import patch

import numpy as np

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import _random as random
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    MultiThreadedTestCase,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed._tensor import common_dtensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorContinuousTestBase,
    DTensorOpTestBase,
    DTensorTestBase,
    LocalDTensorContinuousTestBase,
    LocalDTensorOpTestBase,
    LocalDTensorTestBase,
    requires_accelerator_ranks,
    with_comms,
)


class DTensorContinuousTestBaseTest(TestCase):
    def test_device_selection_matches_instance_property(self):
        class Fixture(DTensorContinuousTestBase):
            world_size = 4

        with (
            patch.object(common_dtensor, "TEST_CUDA", True),
            patch.object(common_dtensor, "DEVICE_TYPE", "cuda"),
        ):
            for count in (0, 1, 4, 8):
                with (
                    self.subTest(device_count=count),
                    patch.object(common_dtensor, "DEVICE_COUNT", count, create=True),
                ):
                    expected = "cpu" if count < Fixture.world_size else "cuda"
                    self.assertEqual(Fixture().device_type, expected)
                    self.assertEqual(
                        Fixture.backend_str(), "gloo" if expected == "cpu" else "nccl"
                    )

    def test_cpu_fallback_backend(self):
        class FallbackTest(DTensorContinuousTestBase):
            world_size = 4

        with patch.object(common_dtensor, "DEVICE_COUNT", 1, create=True):
            self.assertEqual(FallbackTest.backend_str(), "gloo")

    def test_cpu_fallback_does_not_bind_accelerator(self):
        class FallbackTest(DTensorContinuousTestBase):
            world_size = 4

        with (
            patch.object(common_dtensor, "DEVICE_COUNT", 1, create=True),
            patch.object(torch.accelerator, "is_available", return_value=True),
            patch.object(torch.accelerator, "device_count", return_value=1),
            patch.object(torch.accelerator, "set_device_index") as set_device,
            patch.object(MultiProcContinuousTest, "_init_pg") as init_pg,
        ):
            FallbackTest._init_pg(3, 4, "unused")
            init_pg.assert_called_once_with(3, 4, "unused")
            set_device.assert_not_called()


@instantiate_parametrized_tests
class LocalDTensorHarnessTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "orig_base,local_base",
        [
            (DTensorTestBase, LocalDTensorTestBase),
            (DTensorOpTestBase, LocalDTensorOpTestBase),
            (DTensorContinuousTestBase, LocalDTensorContinuousTestBase),
        ],
    )
    def test_factory_inherits_only_single_process_harness(self, orig_base, local_base):
        class Original(orig_base):
            world_size = 4

            def helper(self):
                return "shared body helper"

            def test_body(self):
                pass

        local_cls = create_local_tensor_test_class(Original, base_class=local_base)
        self.assertEqual(local_cls.__bases__, (local_base,))
        self.assertTrue(issubclass(local_cls, TestCase))
        self.assertFalse(
            issubclass(
                local_cls,
                (MultiProcessTestCase, MultiProcContinuousTest, MultiThreadedTestCase),
            )
        )
        self.assertEqual(local_cls("test_body").helper(), "shared body helper")
        self.assertEqual(local_cls("test_body").world_size, 4)

    @parametrize("device_count", [0, 1, 4])
    def test_device_capacity_does_not_shrink_virtual_world(self, device_count):
        class Fixture(LocalDTensorTestBase):
            world_size = 8

        with (
            patch.object(common_dtensor, "TEST_CUDA", True),
            patch.object(common_dtensor, "DEVICE_TYPE", "cuda"),
            patch.object(common_dtensor, "DEVICE_COUNT", device_count, create=True),
        ):
            case = Fixture()
            self.assertEqual(case.device_type, "cuda" if device_count else "cpu")
            self.assertEqual(case.world_size, 8)

    def test_factory_virtual_world_override(self):
        class Original(DTensorTestBase):
            world_size = 0

            def test_body(self):
                pass

        local_cls = create_local_tensor_test_class(Original, world_size=8)
        self.assertEqual(local_cls("test_body").world_size, 8)
        self.assertEqual(Original("test_body").world_size, 0)

    @parametrize("base", [LocalDTensorTestBase, LocalDTensorOpTestBase])
    def test_non_comms_setup_does_not_initialize_pg(self, base):
        observed = []

        class Fixture(base):
            world_size = 0

            def test_body(self):
                observed.append(dist.is_initialized())

        result = unittest.TestResult()
        Fixture("test_body").run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(observed, [False])

    @parametrize(
        "base",
        [LocalDTensorTestBase, LocalDTensorOpTestBase, LocalDTensorContinuousTestBase],
    )
    @parametrize(
        "outcome", ["pass", "failure", "skip", "setup_failure", "teardown_failure"]
    )
    def test_per_test_cleanup_survives_failures(self, base, outcome):
        initialized = []
        previous_hook = sys.excepthook

        class Fixture(base):
            world_size = 2
            device_type = "cpu"

            def setUp(self):
                super().setUp()
                self.init_pg(False)
                if outcome == "setup_failure":
                    raise RuntimeError("setup failure")

            @with_comms
            def test_body(self):
                self.init_pg(False)
                initialized.append(dist.get_world_size())
                random._rng_tracker = object()
                if outcome == "failure":
                    self.fail("body failure")
                if outcome == "skip":
                    self.skipTest("body skip")

            def tearDown(self):
                super().tearDown()
                if outcome == "teardown_failure":
                    raise RuntimeError("teardown failure")

        case = Fixture("test_body")
        result = unittest.TestResult()
        with (
            patch.object(
                dist, "init_process_group", wraps=dist.init_process_group
            ) as initialize,
            patch.object(
                torch.autograd,
                "_enable_record_function",
                wraps=torch.autograd._enable_record_function,
            ) as record,
            patch(
                "torch.distributed.tensor.debug._clear_sharding_prop_cache",
                wraps=torch.distributed.tensor.debug._clear_sharding_prop_cache,
            ) as clear_cache,
        ):
            case.run(result)
        self.assertEqual(initialize.call_count, 1)
        self.assertEqual(clear_cache.call_count, 1)
        self.assertEqual(
            [call.args[0] for call in record.call_args_list], [False, True]
        )
        self.assertEqual(len(result.failures), int(outcome == "failure"))
        self.assertEqual(
            len(result.errors), int(outcome in ("setup_failure", "teardown_failure"))
        )
        self.assertEqual(len(result.skipped), int(outcome == "skip"))
        self.assertEqual(initialized, [] if outcome == "setup_failure" else [2])
        self.assertFalse(dist.is_initialized())
        self.assertEqual(dist.distributed_c10d._world.pg_map, {})
        self.assertIsNone(case._pg)
        self.assertIsNone(random._rng_tracker)
        self.assertIs(sys.excepthook, previous_hook)

    def test_continuous_fixture_uses_fresh_pg_without_comms_decorator(self):
        groups = []

        class Fixture(LocalDTensorContinuousTestBase):
            world_size = 2
            device_type = "cpu"

            def test_body(self):
                groups.append(dist.distributed_c10d._get_default_group())

        result = unittest.TestResult()
        unittest.TestSuite([Fixture("test_body"), Fixture("test_body")]).run(result)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(len(groups), 2)
        self.assertIsNot(groups[0], groups[1])
        self.assertFalse(dist.is_initialized())

    def test_unowned_pg_is_rejected_and_not_destroyed(self):
        previous_hook = sys.excepthook
        previous_tracker = random._rng_tracker
        dist.init_process_group("fake", rank=0, world_size=3)
        group = dist.distributed_c10d._get_default_group()
        tracker = object()
        random._rng_tracker = tracker
        try:

            class Fixture(LocalDTensorContinuousTestBase):
                world_size = 2

                def test_body(self):
                    pass

            result = unittest.TestResult()
            Fixture("test_body").run(result)
            self.assertEqual(len(result.errors), 1)
            self.assertIn("unowned process group", result.errors[0][1])
            self.assertIs(dist.distributed_c10d._get_default_group(), group)
            self.assertEqual(dist.get_world_size(), 3)
            self.assertIs(random._rng_tracker, tracker)
        finally:
            dist.destroy_process_group(group)
            random._rng_tracker = previous_tracker
            sys.excepthook = previous_hook

    @parametrize(
        "world_size,device_type,passes",
        [(4, "cuda", True), (2, "cuda", False), (4, "cpu", False)],
    )
    def test_accelerator_rank_requirement_uses_harness_contract(
        self, world_size, device_type, passes
    ):
        called = []
        with patch(
            "torch.testing._internal.common_distributed.at_least_x_gpu",
            return_value=True,
        ):

            class Fixture(TestCase):
                @requires_accelerator_ranks(4)
                def test_body(self):
                    called.append("body")

            case = Fixture("test_body")
            case.world_size = world_size
            case.device_type = device_type
            if passes:
                case.test_body()
            else:
                with self.assertRaisesRegex(unittest.SkipTest, "at least 4 ranks"):
                    case.test_body()
            self.assertEqual(called, ["body"] if passes else [])
            self.assertEqual(Fixture.test_body._min_gpus_required, 4)

    @parametrize("required_ranks", [4, 8])
    @parametrize("physical_gpus", [0, 1])
    def test_rank_guard_needs_one_physical_accelerator(
        self, required_ranks, physical_gpus
    ):
        called = []
        with patch(
            "torch.testing._internal.common_distributed.at_least_x_gpu",
            side_effect=lambda count: physical_gpus >= count,
        ) as capacity:

            class Fixture(TestCase):
                world_size = required_ranks
                device_type = "cuda"

                @requires_accelerator_ranks(required_ranks)
                def test_body(self):
                    called.append("body")

            capacity.assert_called_once_with(1)
            case = Fixture("test_body")
            if physical_gpus:
                case.test_body()
            else:
                with self.assertRaisesRegex(unittest.SkipTest, "accelerator device"):
                    case.test_body()
            self.assertEqual(called, ["body"] if physical_gpus else [])

    @parametrize(
        "base,parent",
        [
            (DTensorTestBase, MultiProcessTestCase),
            (DTensorContinuousTestBase, MultiProcContinuousTest),
        ],
    )
    def test_real_harness_skips_unavailable_capacity_before_setup(self, base, parent):
        with patch(
            "torch.testing._internal.common_distributed.at_least_x_gpu",
            return_value=True,
        ):

            class Fixture(base):
                world_size = 8

                @requires_accelerator_ranks(8)
                def test_body(self):
                    pass

        with (
            patch.object(common_dtensor, "at_least_x_gpu", return_value=False),
            patch.object(parent, "setUp") as parent_setup,
            self.assertRaisesRegex(unittest.SkipTest, "8 accelerator devices"),
        ):
            Fixture("test_body").setUp()
        parent_setup.assert_not_called()

    def test_op_assertion_helper_checks_results(self):
        case = LocalDTensorOpTestBase()
        case.assertEqualOnRank(1, 1)
        with self.assertRaisesRegex(AssertionError, "Scalars are not equal"):
            case.assertEqualOnRank(1, 2)


class DTensorTestBaseUtilCPUTest(DTensorTestBase):
    """
    This class tests if the basic functionalities of DTensorTestBase are
    working as expected on CPU, regardless of the presence of CUDA devices.
    """

    @property
    def backend(self):
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self):
        return np.prod(list(self.mesh_dim_sizes.values())).item()

    @property
    def mesh_dim_sizes(self) -> dict[str, int]:
        """Mapping from mesh dimension names to sizes."""
        return {"data": 2, "fsdp": 3, "tensor": 5}

    def build_device_mesh(self) -> DeviceMesh:
        return init_device_mesh(
            self.device_type,
            mesh_shape=tuple(self.mesh_dim_sizes.values()),
            mesh_dim_names=tuple(self.mesh_dim_sizes.keys()),
        )

    @with_comms
    def test_dtensor_testbase_destroy_pg(self):
        # This tests destroy_pg() correctly finishes.
        device_mesh = self.build_device_mesh()  # noqa: F841


if __name__ == "__main__":
    run_tests()
