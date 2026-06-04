# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import sympy

import torch
from torch._inductor import ir
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.lowering import _record_symbolic_input_source
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


class TestPythonWrapperCodegen(TestCase):
    def _new_wrapper(self):
        wrapper = PythonWrapperCodegen.__new__(PythonWrapperCodegen)
        wrapper.prefix = IndentedBuffer()
        return wrapper

    def _graph_with_sizevars(self, **kwargs):
        return SimpleNamespace(
            sizevars=SimpleNamespace(simplify=lambda x: x),
            **kwargs,
        )

    def _new_cpp_wrapper(self):
        wrapper = CppWrapperCpu.__new__(CppWrapperCpu)
        wrapper.prefix = IndentedBuffer()
        return wrapper

    def test_explicit_symbol_input_assignment(self):
        wrapper = self._new_wrapper()
        bound_vars = OrderedSet()
        s0 = sympy.Symbol("s0")

        with V.set_graph_handler(self._graph_with_sizevars()):
            wrapper.codegen_input_symbol_assignment("arg0_1", s0, bound_vars)

        self.assertEqual(wrapper.prefix.getvalue().strip(), "s0 = arg0_1")
        self.assertEqual(list(bound_vars), [s0])

    def test_explicit_symbol_input_assignment_uses_canonical_symbol(self):
        wrapper = self._new_wrapper()
        bound_vars = OrderedSet()
        s0 = sympy.Symbol("s0")
        s1 = sympy.Symbol("s1")
        graph = self._graph_with_sizevars()
        graph.sizevars.simplify = lambda x: x.xreplace({s0: s1})

        with V.set_graph_handler(graph):
            wrapper.codegen_input_symbol_assignment("arg0_1", s0, bound_vars)

        self.assertExpectedInline(
            wrapper.prefix.getvalue().strip(),
            """\
s1 = arg0_1
s0 = arg0_1""",
        )
        self.assertEqual(list(bound_vars), [s1, s0])

    def test_explicit_symbol_input_assignment_preserves_raw_symbol(self):
        wrapper = self._new_wrapper()
        bound_vars = OrderedSet()
        s0 = sympy.Symbol("s0")
        s1 = sympy.Symbol("s1")
        graph = self._graph_with_sizevars()
        graph.sizevars.simplify = lambda x: x.xreplace({s0: s1})

        with V.set_graph_handler(graph):
            wrapper.codegen_input_symbol_assignment("arg0_1", s1, bound_vars)
            wrapper.codegen_input_symbol_assignment("arg1_1", s0, bound_vars)

        self.assertExpectedInline(
            wrapper.prefix.getvalue().strip(),
            """\
s1 = arg0_1
s0 = arg1_1""",
        )
        self.assertEqual(list(bound_vars), [s1, s0])

    def test_explicit_symbol_input_assignment_preserves_static_raw_symbol(self):
        wrapper = self._new_wrapper()
        bound_vars = OrderedSet()
        s0 = sympy.Symbol("s0")
        graph = self._graph_with_sizevars()
        graph.sizevars.simplify = lambda x: x.xreplace({s0: sympy.Integer(8)})

        with V.set_graph_handler(graph):
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

    def test_record_symbolic_input_source_ignores_non_input_tensorbox(self):
        s0 = sympy.Symbol("s0")
        tensor = ir.Pointwise.create(
            device=torch.device("cpu"),
            dtype=torch.float32,
            inner_fn=lambda index: index[0],
            ranges=[s0],
        )
        graph = SimpleNamespace(graph_inputs={}, symbolic_input_sources={})

        with V.set_graph_handler(graph):
            _record_symbolic_input_source(tensor, 0, s0, "size")

        self.assertEqual(graph.symbolic_input_sources, {})

    def test_record_symbolic_input_source_ignores_input_view(self):
        s0 = sympy.Symbol("s0")
        base = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[10],
                    stride=[1],
                ),
            )
        )
        view = ir.TensorBox.create(
            ir.ReinterpretView(
                data=base.data,
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0],
                    stride=[1],
                ),
            )
        )
        graph = SimpleNamespace(
            graph_inputs={"arg0_1": base}, symbolic_input_sources={}
        )

        with V.set_graph_handler(graph):
            _record_symbolic_input_source(view, 0, s0, "size")

        self.assertEqual(graph.symbolic_input_sources, {})

    def test_record_symbolic_input_source_records_direct_input(self):
        s0 = sympy.Symbol("s0")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0],
                    stride=[1],
                ),
            )
        )
        graph = SimpleNamespace(
            graph_inputs={"arg0_1": tensor}, symbolic_input_sources={}
        )

        with V.set_graph_handler(graph):
            _record_symbolic_input_source(tensor, 0, s0, "size")

        self.assertEqual(graph.symbolic_input_sources, {s0: ("arg0_1", "size", 0)})

    def test_codegen_inputs_binds_recorded_symbolic_input_source(self):
        wrapper = self._new_wrapper()
        s0 = sympy.Symbol("s0")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0, 3],
                    stride=[3, 1],
                ),
            )
        )
        graph = self._graph_with_sizevars(
            graph_inputs={"arg0_1": tensor},
            graph_input_names=["arg0_1"],
            symbolic_input_sources={s0: ("arg0_1", "size", 0)},
        )

        with V.set_graph_handler(graph):
            wrapper.codegen_inputs()

        self.assertEqual(wrapper.prefix.getvalue().strip(), "s0 = arg0_1.size()[0]")

    def test_codegen_inputs_binds_canonical_recorded_symbolic_input_source(self):
        wrapper = self._new_wrapper()
        raw = sympy.Symbol("s0")
        canonical = sympy.Symbol("s1")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[raw],
                    stride=[1],
                ),
            )
        )
        graph = self._graph_with_sizevars(
            graph_inputs={"arg0_1": tensor},
            graph_input_names=["arg0_1"],
            symbolic_input_sources={raw: ("arg0_1", "size", 0)},
        )
        graph.sizevars.simplify = lambda x: (
            x.xreplace({raw: canonical}) if isinstance(x, sympy.Basic) else x
        )
        graph.sizevars.shape_env = SimpleNamespace(replacements={raw: canonical})

        with (
            V.set_graph_handler(graph),
            torch._inductor.config.patch("size_asserts", False),
        ):
            wrapper.codegen_inputs()

        self.assertExpectedInline(
            wrapper.prefix.getvalue().strip(),
            """\
s1 = arg0_1.size()[0]
s0 = s1""",
        )

    def test_codegen_inputs_binds_size_assert_symbols(self):
        wrapper = self._new_wrapper()
        s0 = sympy.Symbol("s0")
        s1 = sympy.Symbol("s1")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0],
                    stride=[s1],
                ),
            )
        )
        graph = self._graph_with_sizevars(
            graph_inputs={"arg0_1": tensor},
            graph_input_names=["arg0_1"],
            symbolic_input_sources={},
        )

        with (
            V.set_graph_handler(graph),
            torch._inductor.config.patch("size_asserts", True),
        ):
            wrapper.codegen_inputs()

        self.assertExpectedInline(
            wrapper.prefix.getvalue().strip(),
            """\
s0 = arg0_1.size()[0]
s1 = arg0_1.stride()[0]""",
        )

    def test_codegen_inputs_binds_canonical_size_assert_symbol(self):
        wrapper = self._new_wrapper()
        raw = sympy.Symbol("s0")
        canonical = sympy.Symbol("s1")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[raw],
                    stride=[1],
                ),
            )
        )
        graph = self._graph_with_sizevars(
            graph_inputs={"arg0_1": tensor},
            graph_input_names=["arg0_1"],
            symbolic_input_sources={},
        )
        graph.sizevars.simplify = lambda x: (
            x.xreplace({raw: canonical}) if isinstance(x, sympy.Basic) else x
        )
        graph.sizevars.shape_env = SimpleNamespace(replacements={raw: canonical})

        with (
            V.set_graph_handler(graph),
            torch._inductor.config.patch("size_asserts", True),
        ):
            wrapper.codegen_inputs()

        self.assertExpectedInline(
            wrapper.prefix.getvalue().strip(),
            """\
s1 = arg0_1.size()[0]
s0 = s1""",
        )

    def test_codegen_inputs_rejects_unbound_input_symbol(self):
        wrapper = self._new_wrapper()
        s0 = sympy.Symbol("s0")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0 + 1, 3],
                    stride=[3, 1],
                ),
            )
        )
        graph = self._graph_with_sizevars(
            graph_inputs={"arg0_1": tensor},
            graph_input_names=["arg0_1"],
            symbolic_input_sources={},
        )

        with V.set_graph_handler(graph):
            with self.assertRaisesRegex(AssertionError, "expected .*s0"):
                wrapper.codegen_inputs()

    def test_cpp_bind_input_symbol_uses_cpp_tensor_access(self):
        wrapper = self._new_cpp_wrapper()
        bound_vars = OrderedSet()
        s0 = sympy.Symbol("s0")
        s1 = sympy.Symbol("s1")

        wrapper.bind_input_symbol(s0, "arg0_1", "size", 0, bound_vars)
        wrapper.bind_input_symbol(s1, "arg0_1", "stride", 1, bound_vars)

        self.assertExpectedInline(
            wrapper.prefix.getvalue().strip(),
            """\
int64_t s0 = arg0_1.sizes()[0];
int64_t s1 = arg0_1.strides()[1];""",
        )
        self.assertEqual(list(bound_vars), [s0, s1])


if __name__ == "__main__":
    run_tests()
