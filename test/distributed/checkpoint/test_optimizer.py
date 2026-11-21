# Owner(s): ["oncall: distributed checkpointing"]

from unittest import mock

import torch
import torch.distributed.checkpoint.optimizer as dcp_optimizer
from torch.distributed.checkpoint.metadata import TensorProperties
from torch.distributed.checkpoint.optimizer import _alloc_tensor, _gen_rank_device
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestOptimizerDeviceDefaults(TestCase):
    hw_classification = HardwareClassification.GENERIC

    # The default-follows-accelerator behavior is exercised against real
    # hardware by TestOptimizerDeviceAccelerator below. These GENERIC tests
    # pin the explicit-device and accelerator-less fallback paths, which are
    # deterministic on any machine.

    def test_gen_rank_device_explicit_cpu(self):
        self.assertEqual(_gen_rank_device(3, "cpu"), "cpu")

    def test_infer_default_device_type_falls_back_to_cpu(self):
        with mock.patch.object(
            torch.accelerator, "current_accelerator", return_value=None
        ):
            self.assertEqual(dcp_optimizer._infer_default_device_type(), "cpu")
            self.assertEqual(_gen_rank_device(5), "cpu")

    def test_gen_rank_device_falls_back_to_cpu_when_device_unavailable(self):
        # An explicit device type that is not available at runtime must fall
        # back to cpu (e.g. "cuda" on a CPU-only machine).
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(_gen_rank_device(5, "cuda"), "cpu")

    def test_alloc_tensor_default_falls_back_to_cpu(self):
        props = TensorProperties(dtype=torch.float32)
        with mock.patch.object(
            torch.accelerator, "current_accelerator", return_value=None
        ):
            t = _alloc_tensor(props, (2, 3))
        self.assertEqual(t.device.type, "cpu")
        self.assertEqual(t.shape, torch.Size((2, 3)))

    def test_alloc_tensor_explicit_device_unchanged(self):
        props = TensorProperties(dtype=torch.float32)
        t = _alloc_tensor(props, (2, 3), "cpu")
        self.assertEqual(t.device.type, "cpu")


class TestOptimizerDeviceAccelerator(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_infer_default_device_type_uses_accelerator(self, device):
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        assert accelerator is not None  # noqa: S101
        self.assertEqual(torch.device(device).type, accelerator.type)
        self.assertEqual(dcp_optimizer._infer_default_device_type(), accelerator.type)

    def test_gen_rank_device_default_uses_accelerator(self, device):
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        assert accelerator is not None  # noqa: S101
        self.assertEqual(torch.device(device).type, accelerator.type)
        self.assertNotEqual(_gen_rank_device(0), "cpu")
        self.assertTrue(_gen_rank_device(0).startswith(accelerator.type))

    def test_alloc_tensor_default_uses_accelerator(self, device):
        accelerator = torch.accelerator.current_accelerator(check_available=True)
        assert accelerator is not None  # noqa: S101
        props = TensorProperties(dtype=torch.float32)
        t = _alloc_tensor(props, (2, 3))
        self.assertEqual(t.device.type, accelerator.type)
        self.assertEqual(t.shape, torch.Size((2, 3)))


instantiate_device_type_tests(
    TestOptimizerDeviceAccelerator, globals(), except_for="cpu"
)


if __name__ == "__main__":
    run_tests()
