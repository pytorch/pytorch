# Owner(s): ["module: inductor"]

import sympy

import torch
from torch._inductor import ir
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import IndentedBuffer
from torch.utils._ordered_set import OrderedSet


class TestPythonWrapperCodegen(TestCase):
    def _new_wrapper(self):
        wrapper = PythonWrapperCodegen.__new__(PythonWrapperCodegen)
        wrapper.prefix = IndentedBuffer()
        return wrapper

    def test_explicit_symbol_input_assignment(self):
        wrapper = self._new_wrapper()
        bound_vars = OrderedSet()
        s0 = sympy.Symbol("s0")

        wrapper.codegen_input_symbol_assignment("arg0_1", s0, bound_vars)

        self.assertEqual(wrapper.prefix.getvalue().strip(), "s0 = arg0_1")
        self.assertEqual(list(bound_vars), [s0])

    def test_tensor_input_does_not_bind_size_or_stride_symbols(self):
        wrapper = self._new_wrapper()
        bound_vars = OrderedSet()
        s0 = sympy.Symbol("s0")
        s1 = sympy.Symbol("s1")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0, s1],
                    stride=[s1, 1],
                ),
            )
        )

        wrapper.codegen_input_symbol_assignment("arg0_1", tensor, bound_vars)

        self.assertEqual(wrapper.prefix.getvalue(), "")
        self.assertEqual(list(bound_vars), [])


if __name__ == "__main__":
    run_tests()
