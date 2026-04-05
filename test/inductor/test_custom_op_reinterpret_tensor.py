"""
Test that custom ops receiving Inductor-generated reinterpret_tensor views
can safely call .data_ptr() without crashing, and produce correct results.

Regression test for: Custom C++ ops crash when Inductor passes
reinterpret_tensor views that lack accessible storage.
"""
import unittest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests

HAS_CUDA = torch.cuda.is_available()

# Register test custom op at module level (avoid re-registration across tests)
_test_lib = torch.library.Library("test_reinterpret", "DEF")
_test_lib.define("identity(Tensor x) -> Tensor")


@torch.library.impl("test_reinterpret::identity", "cuda", lib=_test_lib)
def _identity_impl(x):
    x.data_ptr()  # this call must not crash
    return x.clone()


@torch.library.register_fake("test_reinterpret::identity", lib=_test_lib)
def _identity_fake(x):
    return x.new_empty(x.shape)


class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ops.test_reinterpret.identity(x)

    @staticmethod
    def backward(ctx, go):
        return torch.ops.test_reinterpret.identity(go)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(128, 128, bias=False) for _ in range(4)]
        )
        self.norm = torch.nn.LayerNorm(128)

    def forward(self, x):
        for layer in self.layers:
            x = x + MyOp.apply(self.norm(layer(x)))
        return x.sum()


@unittest.skipIf(not HAS_CUDA, "CUDA required")
class TestCustomOpReinterpretTensor(TestCase):
    def test_custom_op_data_ptr_with_compiled_model(self):
        """Custom op calling .data_ptr() should work inside torch.compile
        and produce results matching eager execution."""
        model = Model().cuda().bfloat16()
        model.train()

        # Create a copy for eager reference
        eager_model = Model().cuda().bfloat16()
        eager_model.load_state_dict(model.state_dict())

        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)

        # Eager reference
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            expected = eager_model(x)

        # Compiled should not crash AND should match eager
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual = compiled(x)

        self.assertEqual(actual, expected)

        # Backward should not crash
        actual.backward()
        expected.backward()

        # Gradients should match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), eager_model.named_parameters()
        ):
            self.assertEqual(
                p1.grad, p2.grad, msg=f"Gradient mismatch for {n1}"
            )


if __name__ == "__main__":
    run_tests()
