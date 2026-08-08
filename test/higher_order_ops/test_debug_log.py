# Owner(s): ["module: higher order operators"]
"""Tests for torch.utils.debug_log.debug_grad_log."""

import logging

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)
from torch.utils.debug_log import debug_grad_log


class TestDebugGradLog(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        self._log_records: list[str] = []

        class _Handler(logging.Handler):
            def __init__(self, dest):
                super().__init__()
                self.dest = dest

            def emit(self, record):
                self.dest.append(self.format(record))

        self._handler = _Handler(self._log_records)
        logger = logging.getLogger("torch.utils.debug_log")
        logger.addHandler(self._handler)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, self._handler)

    @property
    def bwd_logs(self) -> list[str]:
        return [r for r in self._log_records if "[bwd]" in r]

    def test_single_tensor(self, device):
        x = torch.randn(4, requires_grad=True, device=device)
        y = x * 2
        debug_grad_log(y)
        y.sum().backward()

        self.assertEqual(len(self.bwd_logs), 1)
        self.assertIn("t0_grad_norm=", self.bwd_logs[0])

    def test_multi_tensor(self, device):
        x = torch.randn(4, requires_grad=True, device=device)
        y = torch.randn(4, requires_grad=True, device=device)
        debug_grad_log(x, y)
        (x * 2 + y * 3).sum().backward()

        self.assertEqual(len(self.bwd_logs), 1)
        self.assertIn("t0_grad_norm=", self.bwd_logs[0])
        self.assertIn("t1_grad_norm=", self.bwd_logs[0])

    def test_gradient_values(self, device):
        x = torch.tensor([1.0], requires_grad=True, device=device)
        y = torch.tensor([1.0], requires_grad=True, device=device)
        debug_grad_log(x, y)
        (x * 2 + y * 3).sum().backward()

        self.assertEqual(len(self.bwd_logs), 1)
        self.assertIn("t0_grad_norm=2.0000", self.bwd_logs[0])
        self.assertIn("t1_grad_norm=3.0000", self.bwd_logs[0])

    def test_no_requires_grad_no_log(self, device):
        x = torch.randn(3, requires_grad=False, device=device)
        debug_grad_log(x)
        self.assertEqual(len(self.bwd_logs), 0)

    def test_forward_is_noop(self, device):
        x = torch.randn(3, requires_grad=True, device=device)
        debug_grad_log(x)
        self.assertEqual(len(self._log_records), 0)


instantiate_device_type_tests(TestDebugGradLog, globals())


if __name__ == "__main__":
    run_tests()
