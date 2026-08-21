# Owner(s): ["module: inductor"]

import torch
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    onlyAccelerator,
    requires_capabilities,
)
from torch.testing._internal.common_utils import HardwareClassification
from torch.utils._triton import has_triton


class TestTritonSyntacticallyValid(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @requires_capabilities(Capability.lib.triton)
    @onlyAccelerator
    def test_triton_sqrt(self, device):
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


_test_triton_sqrt_mtia = TestTritonSyntacticallyValid.__dict__[
    "test_triton_sqrt"
].__wrapped__.__wrapped__


instantiate_device_type_tests(
    TestTritonSyntacticallyValid,
    globals(),
    except_for=("hpu",),
    allow_xpu=True,
)
if torch.mtia.is_available() and has_triton():

    class TestTritonSyntacticallyValidMTIA(TestCase):
        hw_classification = HardwareClassification.ACCELERATOR

        def test_triton_sqrt(self):
            _test_triton_sqrt_mtia(self, "mtia")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
