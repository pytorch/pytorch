# Owner(s): ["module: inductor"]
import unittest

from torch._inductor.codegen.cpp import CppOverrides, CppVecOverrides
from torch._inductor.codegen.halide import HalideOverrides
from torch._inductor.codegen.mps import MetalOverrides
from torch._inductor.codegen.triton import TritonKernelOverrides
from torch._inductor.ops_handler import list_ops, OP_NAMES
from torch._inductor.test_case import TestCase


class TestOpCompleteness(TestCase):
    def verify_ops_handler_completeness(self, handler):
        op_names = list_ops(handler)
        if OP_NAMES == op_names:
            return
        print(f"Missing ops: {OP_NAMES - op_names}")
        print(f"Extra ops: {op_names - OP_NAMES}")
        self.assertEqual(", ".join(OP_NAMES - op_names), "")
        self.assertEqual(", ".join(op_names - OP_NAMES), "")

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
