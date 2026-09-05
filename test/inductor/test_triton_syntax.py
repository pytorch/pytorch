# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.exc import TritonUnavailableError
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import HardwareClassification
from torch.testing._internal.inductor_utils import requires_triton
from torch.utils._triton import has_triton


def _require_device_triton(device):
    try:
        device_interface = get_interface_for_device(torch.device(device).type)
    except NotImplementedError as exc:
        raise unittest.SkipTest(f"requires Triton support for {device}") from exc
    if not device_interface.is_triton_capable(device):
        raise unittest.SkipTest(f"requires Triton support for {device}")
    try:
        device_interface.raise_if_triton_unavailable(device)
    except TritonUnavailableError as exc:
        raise unittest.SkipTest(str(exc)) from exc


def _test_triton_sqrt(device):
    _require_device_triton(device)
    # https://github.com/pytorch/pytorch/issues/142328
    import math

    import torch.nn as nn

    device_module = torch.get_device_module(device)
    is_bf16_supported = getattr(device_module, "is_bf16_supported", None)
    if is_bf16_supported is None:
        dtype = torch.float16
    else:
        try:
            supports_bf16 = is_bf16_supported(including_emulation=False)
        except TypeError:
            supports_bf16 = is_bf16_supported()
        dtype = torch.bfloat16 if supports_bf16 else torch.float16

    def newtonschulz5(G, steps: int, eps=1e-7):
        if len(G.shape) != 2:
            raise AssertionError(f"expected a matrix, got shape {G.shape}")
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.to(dtype)
        X /= X.norm() + eps  # ensure top singular value <= 1
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    @torch.compile(backend="inductor")
    def scaled_newton_schulz(G, steps: int):
        shape = G.shape
        dtype = G.dtype
        G = G.reshape(shape[0], -1)
        G = newtonschulz5(G, steps)
        G = G.reshape(shape).type(dtype)
        G = G * math.sqrt(max(1, shape[0] / G[0].numel()))
        return G

    model = nn.Sequential(
        nn.Linear(16, 16, bias=False),
        nn.Linear(16, 32, bias=False),
    ).to(device=device)

    loss = model(torch.randn(4, 16, device=device)).sum()
    loss.backward()

    scaled_newton_schulz(model[0].weight.grad, 6)
    scaled_newton_schulz(model[1].weight.grad, 6)


class TestTritonSyntacticallyValid(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @onlyAccelerator
    @requires_triton()
    def test_triton_sqrt(self, device):
        _test_triton_sqrt(device)


instantiate_device_type_tests(
    TestTritonSyntacticallyValid,
    globals(),
    except_for=("cpu", "hpu"),
    allow_xpu=True,
)
# The built-in device test bases do not include MTIA.
if torch.mtia.is_available() and has_triton():

    class TestTritonSyntacticallyValidMTIA(TestCase):
        hw_classification = HardwareClassification.ACCELERATOR

        def test_triton_sqrt(self):
            _test_triton_sqrt("mtia")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if has_triton():
        run_tests()
