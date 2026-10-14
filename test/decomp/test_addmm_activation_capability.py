# Owner(s): ["oncall: pt2"]

import unittest

import torch
from torch._decomp.decompositions import _addmm_activation
from torch._decomp.decompositions import register_eager_gelu_approximation
from torch._decomp.decompositions import _EAGER_GELU_APPROXIMATION
from torch.testing._internal.common_utils import TestCase, run_tests


def _make(m=8, n=16, k=4, device="cpu"):
    """bias `self` of shape [n] (broadcastable), mat1 [m,k], mat2 [k,n]."""
    self_ = torch.randn(n, device=device)
    mat1 = torch.randn(m, k, device=device)
    mat2 = torch.randn(k, n, device=device)
    return self_, mat1, mat2


class TestAddmmActivationCapability(TestCase):
    def test_cpu_uses_exact_gelu(self):
        # CPU is unregistered -> exact GELU, unchanged behavior.
        self_, mat1, mat2 = _make()
        out = _addmm_activation(self_, mat1, mat2, use_gelu=True)
        ref = torch.ops.aten.gelu(
            torch.ops.aten.addmm(self_, mat1, mat2), approximate="none"
        )
        self.assertEqual(out, ref)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_still_uses_tanh(self):
        # The registry pre-seeds {"cuda": "tanh"}, so the CUDA fast path is preserved.
        self_, mat1, mat2 = _make(device="cuda")
        out = _addmm_activation(self_, mat1, mat2, use_gelu=True)
        ref = torch.ops.aten.gelu(
            torch.ops.aten.addmm(self_, mat1, mat2), approximate="tanh"
        )
        self.assertEqual(out, ref)

    def test_default_registry_keeps_cuda_tanh(self):
        self.assertEqual(_EAGER_GELU_APPROXIMATION.get("cuda"), "tanh")

    def test_register_routes_lookup(self):
        register_eager_gelu_approximation("custom_accel", "tanh")
        self.assertEqual(_EAGER_GELU_APPROXIMATION.get("custom_accel"), "tanh")
        # An unregistered backend keeps the exact path.
        self.assertEqual(_EAGER_GELU_APPROXIMATION.get("another_accel", "none"), "none")

    def test_relu_path_unchanged(self):
        self_, mat1, mat2 = _make()
        out = _addmm_activation(self_, mat1, mat2, use_gelu=False)
        ref = torch.ops.aten.relu(torch.ops.aten.addmm(self_, mat1, mat2))
        self.assertEqual(out, ref)


if __name__ == "__main__":
    run_tests()
