# Owner(s): ["module: higher order operators"]
"""Tests for torch.utils.debug_log.debug_grad_log."""

import logging

import torch
from functorch.compile import aot_function, make_boxed_func
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils.debug_log import debug_grad_log


def nop(fx_g, _):
    return make_boxed_func(fx_g)


class _LogCapture(logging.Handler):
    """Logging handler that captures formatted log records."""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(self.format(record))


def _run_eager(f, *inputs):
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    out = f(*cloned)
    out.sum().backward()


def _run_aot(f, *inputs):
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    aot_f = aot_function(f, fw_compiler=nop, bw_compiler=nop)
    out = aot_f(*cloned)
    out.sum().backward()


def _run_compile(f, *inputs):
    torch._dynamo.reset()
    cloned = [x.clone().detach().requires_grad_(x.requires_grad) for x in inputs]
    compiled_f = torch.compile(f, backend="aot_eager", fullgraph=True)
    out = compiled_f(*cloned)
    out.sum().backward()


_RUNNERS = {"eager": _run_eager, "aot": _run_aot, "compile": _run_compile}


@skipIfTorchDynamo("debug_grad_log tests manage their own compilation")
class TestDebugGradLog(TestCase):
    def _add_log_capture(self):
        capture = _LogCapture()
        logger = logging.getLogger("torch.utils.debug_log")
        logger.addHandler(capture)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, capture)
        return capture

    @parametrize("backend", ["eager", "aot", "compile"])
    def test_single_tensor(self, backend):
        """Backward gradient norm is logged for a single tensor."""
        capture = self._add_log_capture()

        def f(x):
            y = x * 2
            debug_grad_log("single", y)
            return y

        _RUNNERS[backend](f, torch.randn(4, requires_grad=True))

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("[single][bwd]", bwd[0])
        self.assertIn("t0_grad_norm=", bwd[0])

    @parametrize("backend", ["eager", "aot", "compile"])
    def test_multi_tensor(self, backend):
        """Backward gradient norms logged for multiple tensors, fires once."""
        capture = self._add_log_capture()

        def f(x, y):
            z = x * 2 + y * 3
            debug_grad_log("multi", x, y)
            return z

        _RUNNERS[backend](
            f,
            torch.randn(4, requires_grad=True),
            torch.randn(4, requires_grad=True),
        )

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("[multi][bwd]", bwd[0])
        self.assertIn("t0_grad_norm=", bwd[0])
        self.assertIn("t1_grad_norm=", bwd[0])

    @parametrize("backend", ["eager", "aot", "compile"])
    def test_gradient_values(self, backend):
        """Verify logged gradient norms match expected values."""
        capture = self._add_log_capture()

        def f(x, y):
            debug_grad_log("values", x, y)
            return x * 2 + y * 3

        _RUNNERS[backend](
            f,
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([1.0], requires_grad=True),
        )

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        # d(x*2+y*3)/dx = 2, d(x*2+y*3)/dy = 3
        self.assertIn("t0_grad_norm=2.0000", bwd[0])
        self.assertIn("t1_grad_norm=3.0000", bwd[0])

    def test_no_requires_grad_no_log(self):
        """No backward log when no tensor requires grad."""
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=False)
        debug_grad_log("noop", x)

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 0)

    def test_forward_is_noop(self):
        """debug_grad_log does nothing in the forward pass."""
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=True)
        debug_grad_log("fwd_check", x)

        self.assertEqual(len(capture.records), 0)


instantiate_parametrized_tests(TestDebugGradLog)


if __name__ == "__main__":
    run_tests()
