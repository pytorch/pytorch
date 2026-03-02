# Owner(s): ["module: inductor"]
"""
Tests for inductor lowering of functional custom ops to out-variant via ExternKernelOut.
"""

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


DEVICES = ("cpu", GPU_TYPE) if HAS_GPU else ("cpu",)


@instantiate_parametrized_tests
class TestCustomOpOutLowering(InductorTestCase):
    """Tests for lowering functional custom ops to out-variant ExternKernelOut."""

    def _register_add_one_ops(self, lib):
        """Register a simple add_one op with functional + .out overloads."""
        lib.define("add_one(Tensor x) -> Tensor")
        lib.define(
            "add_one.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)",
            tags=(torch.Tag.out_variant,),
        )

        def _add_one_impl(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        def _add_one_out_impl(x: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
            out.copy_(x + 1)
            return out

        lib.impl("add_one", _add_one_impl, "CompositeExplicitAutograd")
        lib.impl("add_one.out", _add_one_out_impl, "CompositeExplicitAutograd")

        @torch.library.register_fake("mylib::add_one", lib=lib)
        def _add_one_fake(x):
            return x.new_empty(x.shape)

        return torch.ops.mylib.add_one, torch.ops.mylib.add_one.out

    @parametrize("device", DEVICES)
    def test_add_one_lowered_to_out(self, device):
        """Test that a simple functional op gets lowered to its out-variant."""
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            self._register_add_one_ops(lib)

            def f(x):
                return torch.ops.mylib.add_one(x)

            x = torch.randn(4, 4, device=device)
            eager_out = f(x)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, backend="inductor", fullgraph=True), x
            )
            self.assertEqual(compiled_out, eager_out)

            self.assertIn(".out(", code)
            self.assertNotIn(".default(", code)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
