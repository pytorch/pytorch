# Owner(s): ["module: inductor"]
import unittest

from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.codegen.halide import HalideOverrides
from torch._inductor.codegen.mps import MetalOverrides
from torch._inductor.codegen.triton import TritonKernelOverrides
from torch._inductor.ops_handler import list_ops, OP_NAMES, OpsHandler
from torch._inductor.test_case import TestCase


class TestOpCompleteness(TestCase):
    def verify_ops_handler_completeness(self, handler):
        for op in OP_NAMES:
            self.assertIsNot(
                getattr(handler, op),
                getattr(OpsHandler, op),
                msg=f"{handler} must implement {op}",
            )
        extra_ops = list_ops(handler) - OP_NAMES
        if extra_ops:
            raise AssertionError(
                f"{handler} has an extra ops: {extra_ops}, add them to OpHandler class or prefix with `_`"
            )

    def test_triton_overrides(self):
        self.verify_ops_handler_completeness(TritonKernelOverrides)

    def test_cpp_overrides(self):
        self.verify_ops_handler_completeness(CppOverrides)

    def test_cpp_vec_overrides(self):
        self.verify_ops_handler_completeness(CppVecOverrides)

    def test_halide_overrides(self):
        self.verify_ops_handler_completeness(HalideOverrides)

    @unittest.skip("MPS backend not yet finished")
    def test_metal_overrides(self):
        self.verify_ops_handler_completeness(MetalOverrides)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
