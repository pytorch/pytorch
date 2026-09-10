# Owner(s): ["oncall: distributed checkpoint"]

from unittest import mock

import torch
import torch.distributed.checkpoint.optimizer as dcp_optimizer
from torch.distributed.checkpoint.metadata import TensorProperties
from torch.distributed.checkpoint.optimizer import _alloc_tensor, _gen_rank_device
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestOptimizerDeviceDefaults(TestCase):
    hw_classification = HardwareClassification.GENERIC

    # All default-path tests patch torch.accelerator.current_accelerator:
    # on a GPU CI machine it would resolve to cuda and break determinism.

    def test_gen_rank_device_explicit_cpu(self):
        self.assertEqual(_gen_rank_device(3, "cpu"), "cpu")

    def test_infer_default_device_type_falls_back_to_cpu(self):
        with mock.patch.object(
            torch.accelerator, "current_accelerator", return_value=None
        ):
            self.assertEqual(dcp_optimizer._infer_default_device_type(), "cpu")
            self.assertEqual(_gen_rank_device(5), "cpu")

    def test_gen_rank_device_default_follows_accelerator(self):
        with mock.patch.object(
            torch.accelerator,
            "current_accelerator",
            return_value=torch.device("xpu"),
        ):
            with mock.patch.object(torch.xpu, "is_available", return_value=True):
                with mock.patch.object(torch.xpu, "device_count", return_value=2):
                    self.assertEqual(
                        dcp_optimizer._infer_default_device_type(), "xpu"
                    )
                    # rank 3 with 2 devices -> "xpu:1", same modulo rule as before
                    self.assertEqual(_gen_rank_device(3), "xpu:1")

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


if __name__ == "__main__":
    run_tests()
