# Owner(s): ["module: inductor"]

import types
from itertools import count
from types import SimpleNamespace

import sympy

import torch
import torch.utils._pytree as pytree
from torch._inductor import ir
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.graph import GraphLowering
from torch._inductor.lowering import _record_symbolic_input_source
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import CallMethodKey
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.utils._ordered_set import OrderedSet


@instantiate_parametrized_tests
class TestPythonWrapperCodegen(TestCase):
    def _new_wrapper(self):
        wrapper = PythonWrapperCodegen.__new__(PythonWrapperCodegen)
        wrapper.prefix = IndentedBuffer()
        return wrapper

    def _new_autotune_storage_case(self):
        wrapper = self._new_wrapper()
        base = ir.InputBuffer(
            name="base",
            layout=ir.FixedLayout(
                torch.device("cpu"),
                torch.float32,
                size=[8, 8],
                stride=[8, 1],
            ),
        )
        view = ir.ReinterpretView(
            data=ir.StorageBox(base),
            layout=ir.FixedLayout(
                torch.device("cpu"),
                torch.float32,
                size=[8, 4],
                stride=[8, 1],
                offset=4,
            ),
        )
        view_buffer = ir.Buffer(name="view", layout=ir.NonOwningLayout(view))
        wrapper.args_to_buffers = {"base": base, "view": view_buffer}
        wrapper.kernel_autotune_calls = IndentedBuffer()
        wrapper.kernel_autotune_storage_idx = 0
        wrapper.kernel_autotune_tmp_arg_idx = 0
        wrapper.kernel_autotune_example_arg_cache = {}
        wrapper.kernel_autotune_storage_cache = {}
        return wrapper, base, view, view_buffer

    def _graph_with_sizevars(self, **kwargs):
        return SimpleNamespace(
            sizevars=SimpleNamespace(
                simplify=lambda x: x,
                free_symbols=lambda: OrderedSet(),
            ),
            **kwargs,
        )

    def _new_cpp_wrapper(self):
        wrapper = CppWrapperCpu.__new__(CppWrapperCpu)
        wrapper.prefix = IndentedBuffer()
        return wrapper

    def _codegen_output_symbol(self, outputs, keypath):
        wrapper = self._new_cpp_wrapper()
        wrapper.lines = []
        wrapper.unbacked_symbol_decls = OrderedSet()
        wrapper.declare = "auto "
        wrapper.ending = ";"
        graph = self._graph_with_sizevars(cpp_wrapper=True)
        graph.sizevars.shape_env = SimpleNamespace(unbacked_renamings={})
        symbol = sympy.Symbol("u0", integer=True)

        with V.set_graph_handler(graph):
            wrapper.codegen_unbacked_symbol_defs_for_outputs(
                "output", outputs, {symbol: keypath}
            )
            wrapper.lines.pop().codegen(IndentedBuffer())

        return wrapper.lines.pop()

    def _new_multi_output(self, *, indices=((list, 0),)):
        # Build a real MultiOutput. Its constructor self-registers with the
        # active graph, so bind the graph's register helpers onto a minimal
        # namespace instead of white-box constructing via object.__new__.
        device = torch.device("cpu")
        graph = self._graph_with_sizevars(
            name=None,
            cpp_wrapper=False,
            buffers=[],
            operations=[],
            name_to_buffer={},
            name_to_op={},
            current_node=None,
            add_device_info=lambda _device: None,
        )
        graph.qualify_name = types.MethodType(GraphLowering.qualify_name, graph)
        graph.register_buffer = types.MethodType(GraphLowering.register_buffer, graph)
        graph.register_operation = types.MethodType(
            GraphLowering.register_operation, graph
        )
        with V.set_graph_handler(graph):
            packed = ir.InputBuffer(name="packed", layout=ir.NoneLayout(device=device))
            return ir.MultiOutput(ir.NoneLayout(device=device), packed, list(indices))

    def test_cpp_output_symbol_traverses_nested_multi_output_with_indices(self):
        output = self._new_multi_output(indices=((list, 0), (list, 0)))
        keypath = (
            pytree.SequenceKey(0),
            pytree.SequenceKey(0),
            CallMethodKey("size"),
            pytree.SequenceKey(0),
        )

        self.assertEqual(
            self._codegen_output_symbol([[output]], keypath),
            f"auto u0 = {output.get_name()}.size(0);",
        )

    def test_cpp_output_symbol_preserves_single_multi_output_behavior(self):
        output = self._new_multi_output(indices=((list, 0),))
        keypath = (
            pytree.SequenceKey(7),
            CallMethodKey("size"),
            pytree.SequenceKey(0),
        )

        self.assertEqual(
            self._codegen_output_symbol([output], keypath),
            f"auto u0 = {output.get_name()}.size(0);",
        )

    @parametrize("bad_idx", [1, -1])
    def test_cpp_output_symbol_rejects_out_of_range_nested_index(self, bad_idx):
        output = self._new_multi_output()
        keypath = (pytree.SequenceKey(0), pytree.SequenceKey(bad_idx))

        with self.assertRaisesRegex(
            AssertionError,
            f"output index {bad_idx} is out of range for list with 1 elements",
        ):
            self._codegen_output_symbol([[output]], keypath)

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

    def test_autotune_storage_collection_includes_nonowning_tensor(self):
        wrapper, base, _, _ = self._new_autotune_storage_case()
        tensor = ir.TensorBox.create(base)
        alias_buffer = ir.Buffer(
            name="alias",
            layout=ir.NonOwningLayout(tensor),
        )
        wrapper.args_to_buffers["alias"] = alias_buffer

        storage_args = wrapper._collect_autotune_storage_args(
            ["alias"],
            [torch.float32],
            [None],
            [alias_buffer],
            None,
        )

        self.assertEqual(set(storage_args), {"alias"})
        self.assertIs(storage_args["alias"].base_buffer, base)
        self.assertIs(storage_args["alias"].tensor, tensor)

    def test_autotune_storage_collection_rejects_nested_reinterpret_views(self):
        wrapper, _, inner_view, _ = self._new_autotune_storage_case()
        nested_view = ir.ReinterpretView(
            data=ir.StorageBox(inner_view),
            layout=ir.FixedLayout(
                torch.device("cpu"),
                torch.float32,
                size=[7, 4],
                stride=[8, 1],
                offset=8,
            ),
        )
        nested_buffer = ir.Buffer(
            name="nested_view",
            layout=ir.NonOwningLayout(nested_view),
        )
        wrapper.args_to_buffers["nested_view"] = nested_buffer

        storage_args = wrapper._collect_autotune_storage_args(
            ["nested_view"],
            [torch.float32],
            [None],
            [nested_buffer],
            None,
        )
        self.assertEqual(storage_args, {})

        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                optimization_hint=lambda value: value,
                optimization_hints=lambda values: tuple(values),
            ),
            get_allocation_size=lambda buffer: tuple(buffer.get_size()),
        )
        with V.set_graph_handler(graph):
            wrapper.generate_example_arg_value(
                "nested_view",
                torch.float32,
                nested_buffer,
                storage_arg=None,
            )

        code = wrapper.kernel_autotune_calls.getvalue()
        self.assertIn("nested_view = generate_example_value", code)
        self.assertNotIn("torch.as_strided", code)

    def test_autotune_storage_collection_unwraps_nested_alias_layouts(self):
        wrapper, base, _, _ = self._new_autotune_storage_case()
        graph = SimpleNamespace(mark_buffer_mutated=lambda _name: None)
        with V.set_graph_handler(graph):
            mutation_buffer = ir.Buffer(
                name="mutation",
                layout=ir.MutationLayoutSHOULDREMOVE(base),
            )
            nested_mutation_buffer = ir.Buffer(
                name="nested_mutation_owner",
                layout=ir.MutationLayoutSHOULDREMOVE(mutation_buffer),
            )
        nonowning_buffer = ir.Buffer(
            name="nonowning",
            layout=ir.NonOwningLayout(ir.TensorBox.create(base)),
        )

        for name, inner_buffer in (
            ("nested_mutation", nested_mutation_buffer),
            ("nested_nonowning", nonowning_buffer),
        ):
            nested_view = ir.ReinterpretView(
                data=ir.StorageBox(inner_buffer),
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[7, 4],
                    stride=[8, 1],
                    offset=1,
                ),
            )
            nested_buffer = ir.Buffer(
                name=name,
                layout=ir.NonOwningLayout(nested_view),
            )
            wrapper.args_to_buffers[name] = nested_buffer

            storage_args = wrapper._collect_autotune_storage_args(
                [name],
                [torch.float32],
                [None],
                [nested_buffer],
                None,
            )

            self.assertIs(storage_args[name].base_buffer, base)
            self.assertIs(storage_args[name].tensor, nested_view)

    def test_autotune_storage_collection_rejects_second_view_through_aliases(
        self,
    ):
        wrapper, _, inner_view, view_buffer = self._new_autotune_storage_case()
        graph = SimpleNamespace(mark_buffer_mutated=lambda _name: None)
        with V.set_graph_handler(graph):
            mutation_buffer = ir.Buffer(
                name="mutation",
                layout=ir.MutationLayoutSHOULDREMOVE(inner_view),
            )

        for name, inner_buffer in (
            ("nested_mutation", mutation_buffer),
            ("nested_nonowning", view_buffer),
        ):
            nested_view = ir.ReinterpretView(
                data=ir.StorageBox(inner_buffer),
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[7, 4],
                    stride=[8, 1],
                    offset=1,
                ),
            )
            nested_buffer = ir.Buffer(
                name=name,
                layout=ir.NonOwningLayout(nested_view),
            )
            wrapper.args_to_buffers[name] = nested_buffer

            storage_args = wrapper._collect_autotune_storage_args(
                [name],
                [torch.float32],
                [None],
                [nested_buffer],
                None,
            )

            self.assertEqual(storage_args, {})

            graph = SimpleNamespace(
                sizevars=SimpleNamespace(
                    optimization_hint=lambda value: value,
                    optimization_hints=lambda values: tuple(values),
                ),
                get_allocation_size=lambda buffer: tuple(buffer.get_size()),
            )
            with V.set_graph_handler(graph):
                wrapper.generate_example_arg_value(
                    name,
                    torch.float32,
                    nested_buffer,
                    storage_arg=None,
                )

        code = wrapper.kernel_autotune_calls.getvalue()
        self.assertIn("nested_mutation = generate_example_value", code)
        self.assertIn("nested_nonowning = generate_example_value", code)
        self.assertNotIn("torch.as_strided", code)

    def test_autotune_storage_collection_groups_owning_and_view_mutations(self):
        wrapper, base, view, _ = self._new_autotune_storage_case()
        graph = SimpleNamespace(mark_buffer_mutated=lambda _name: None)
        with V.set_graph_handler(graph):
            base_mutation = ir.Buffer(
                name="base_mutation",
                layout=ir.MutationLayoutSHOULDREMOVE(base),
            )
            view_mutation = ir.Buffer(
                name="view_mutation",
                layout=ir.MutationLayoutSHOULDREMOVE(view),
            )
        wrapper.args_to_buffers.update(
            {
                "base_mutation": base_mutation,
                "view_mutation": view_mutation,
            }
        )

        storage_args = wrapper._collect_autotune_storage_args(
            ["base_mutation", "view_mutation"],
            [torch.float32, torch.float32],
            [None, None],
            [base_mutation, view_mutation],
            None,
        )

        self.assertEqual(set(storage_args), {"base_mutation", "view_mutation"})
        self.assertIs(storage_args["base_mutation"].base_buffer, base)
        self.assertIs(storage_args["base_mutation"].tensor, base)
        self.assertIs(storage_args["view_mutation"].base_buffer, base)
        self.assertIs(storage_args["view_mutation"].tensor, view)

    def test_autotune_storage_collection_includes_mutation_through_nonowning(self):
        wrapper, base, view, view_buffer = self._new_autotune_storage_case()
        mutation_graph = SimpleNamespace(mark_buffer_mutated=lambda _name: None)
        with V.set_graph_handler(mutation_graph):
            mutation_buffer = ir.Buffer(
                name="mutation",
                layout=ir.MutationLayoutSHOULDREMOVE(view_buffer),
            )
        wrapper.args_to_buffers["mutation"] = mutation_buffer

        storage_args = wrapper._collect_autotune_storage_args(
            ["mutation"],
            [torch.float32],
            [None],
            [mutation_buffer],
            None,
        )
        self.assertEqual(set(storage_args), {"mutation"})
        self.assertIs(storage_args["mutation"].base_buffer, base)
        self.assertIs(storage_args["mutation"].tensor, view)

        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                optimization_hint=lambda value: value,
                optimization_hints=lambda values: tuple(values),
            ),
            get_allocation_storage_size=lambda _buffer: 64,
        )
        with V.set_graph_handler(graph):
            wrapper.generate_example_arg_value(
                "mutation",
                torch.float32,
                mutation_buffer,
                kernel_name="kernel0",
                storage_cache={},
                storage_arg=storage_args["mutation"],
            )

        code = wrapper.kernel_autotune_calls.getvalue()
        self.assertIn("_autotune_storage_0 = generate_example_value", code)
        self.assertIn(
            "mutation = torch.as_strided(_autotune_storage_0, (8, 4), (8, 1), 4)",
            code,
        )

    def test_autotune_storage_collection_rejects_dtype_mismatch(self):
        wrapper, base, _, _ = self._new_autotune_storage_case()
        dtype_view = ir.ReinterpretView(
            data=ir.StorageBox(base),
            layout=ir.FixedLayout(
                torch.device("cpu"),
                torch.float16,
                size=[8, 16],
                stride=[16, 1],
            ),
        )
        dtype_buffer = ir.Buffer(
            name="dtype_view",
            layout=ir.NonOwningLayout(dtype_view),
        )
        wrapper.args_to_buffers["dtype_view"] = dtype_buffer

        storage_args = wrapper._collect_autotune_storage_args(
            ["dtype_view"],
            [torch.float16],
            [None],
            [dtype_buffer],
            None,
        )

        self.assertEqual(storage_args, {})

        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                optimization_hint=lambda value: value,
                optimization_hints=lambda values: tuple(values),
            ),
            get_allocation_size=lambda buffer: tuple(buffer.get_size()),
        )
        with V.set_graph_handler(graph):
            wrapper.generate_example_arg_value(
                "dtype_view",
                torch.float16,
                dtype_buffer,
                kernel_name="kernel0",
                storage_cache={},
                storage_arg=None,
            )

        code = wrapper.kernel_autotune_calls.getvalue()
        self.assertIn("dtype_view = generate_example_value", code)
        self.assertNotIn("torch.as_strided", code)

    def test_autotune_storage_collection_materializes_base_and_view(self):
        wrapper, base, view, view_buffer = self._new_autotune_storage_case()
        storage_args = wrapper._collect_autotune_storage_args(
            ["base", "view"],
            [torch.float32, torch.float32],
            [None, None],
            [base, view_buffer],
            None,
        )

        self.assertEqual(set(storage_args), {"base", "view"})
        self.assertIs(storage_args["base"].base_buffer, base)
        self.assertIs(storage_args["base"].tensor, base)
        self.assertIs(storage_args["view"].base_buffer, base)
        self.assertIs(storage_args["view"].tensor, view)

        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                optimization_hint=lambda value: value,
                optimization_hints=lambda values: tuple(values),
            ),
            get_allocation_storage_size=lambda _buffer: 64,
        )
        storage_cache = {}

        with V.set_graph_handler(graph):
            for arg, buf in (("base", base), ("view", view_buffer)):
                wrapper.generate_example_arg_value(
                    arg,
                    torch.float32,
                    buf,
                    kernel_name="kernel0",
                    storage_cache=storage_cache,
                    storage_arg=storage_args[arg],
                )

        code = wrapper.kernel_autotune_calls.getvalue()
        self.assertEqual(code.count("_autotune_storage_0 = generate_example_value"), 1)
        self.assertIn(
            "base = torch.as_strided(_autotune_storage_0, (8, 8), (8, 1), 0)",
            code,
        )
        self.assertIn(
            "view = torch.as_strided(_autotune_storage_0, (8, 4), (8, 1), 4)",
            code,
        )

    def test_autotune_storage_groups_duplicate_owning_arguments(self):
        wrapper, base, _, _ = self._new_autotune_storage_case()
        wrapper.args_to_buffers["base_alias"] = base
        storage_args = wrapper._collect_autotune_storage_args(
            ["base", "base_alias"],
            [torch.float32, torch.float32],
            [None, None],
            [base, base],
            None,
        )

        self.assertEqual(set(storage_args), {"base", "base_alias"})
        for arg in ("base", "base_alias"):
            self.assertIs(storage_args[arg].base_buffer, base)
            self.assertIs(storage_args[arg].tensor, base)

        graph = SimpleNamespace(
            sizevars=SimpleNamespace(
                optimization_hint=lambda value: value,
                optimization_hints=lambda values: tuple(values),
            ),
            get_allocation_storage_size=lambda _buffer: 64,
        )
        storage_cache = {}

        with V.set_graph_handler(graph):
            for arg in ("base", "base_alias"):
                wrapper.generate_example_arg_value(
                    arg,
                    torch.float32,
                    base,
                    kernel_name="kernel0",
                    storage_cache=storage_cache,
                    storage_arg=storage_args[arg],
                )

        code = wrapper.kernel_autotune_calls.getvalue()
        self.assertEqual(code.count("_autotune_storage_0 = generate_example_value"), 1)
        self.assertIn(
            "base = torch.as_strided(_autotune_storage_0, (8, 8), (8, 1), 0)",
            code,
        )
        self.assertIn(
            "base_alias = torch.as_strided(_autotune_storage_0, (8, 8), (8, 1), 0)",
            code,
        )

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
arg0_1_size = arg0_1.size()
s0 = arg0_1_size[0]
arg0_1_stride = arg0_1.stride()
s1 = arg0_1_stride[0]""",
        )

    def test_codegen_inputs_ignores_unused_compound_input_symbols(self):
        wrapper = self._new_wrapper()
        s0 = sympy.Symbol("s0")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[1],
                    stride=[4 * s0**2],
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
            torch._inductor.config.patch("size_asserts", False),
        ):
            wrapper.codegen_inputs()

        self.assertEqual(wrapper.prefix.getvalue().strip(), "")

    def test_codegen_inputs_ignores_zero_size_compound_input_symbols(self):
        wrapper = self._new_wrapper()
        s0 = sympy.Symbol("s0")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0 + 1, 0],
                    stride=[1, 1],
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

        self.assertEqual(wrapper.prefix.getvalue().strip(), "")

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
arg0_1_size = arg0_1.size()
s0 = arg0_1_size[0]
s1 = s0""",
        )

    def test_codegen_inputs_rejects_unbound_input_symbol(self):
        wrapper = self._new_wrapper()
        s0 = sympy.Symbol("s0")
        s1 = sympy.Symbol("s1")
        tensor = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device("cpu"),
                    torch.float32,
                    size=[s0 + s1, 3],
                    stride=[3, 1],
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
            with self.assertRaisesRegex(AssertionError, "expected .*s0"):
                wrapper.codegen_inputs()

    def test_cpp_bind_input_symbol_emits_raw_replacement_alias(self):
        wrapper = self._new_cpp_wrapper()
        bound_vars = OrderedSet()
        raw = sympy.Symbol("s0")
        canonical = sympy.Symbol("s1")
        graph = self._graph_with_sizevars()
        graph.sizevars.shape_env = SimpleNamespace(replacements={raw: canonical})

        with V.set_graph_handler(graph):
            wrapper.bind_input_symbol(canonical, "arg0_1", "size", 0, bound_vars)

        self.assertExpectedInline(
            wrapper.prefix.getvalue().strip(),
            """\
int64_t s1 = arg0_1.sizes()[0];
int64_t s0 = s1;""",
        )
        self.assertEqual(list(bound_vars), [canonical, raw])

    def _new_int_array_wrapper(self):
        wrapper = CppWrapperCpu.__new__(CppWrapperCpu)
        wrapper.int_array_id = count()
        wrapper.declared_int_array_vars = OrderedSet()
        wrapper.codegen_int_array_var_cache = {}
        wrapper._int_array_writeline_targets = []
        return wrapper

    def test_int_array_declared_in_every_fresh_writeline_target(self):
        # codegen_int_array_var emits no declaration on a cache hit, so a key that
        # aliases a dead target hands back a name whose declaration went into a
        # discarded buffer. Short-lived targets (per-callsite IndentedBuffers, as
        # used by ReinterpretView.codegen_reference) make that reachable: CPython
        # reuses the address of a freed object, in practice on the very next
        # allocation.
        wrapper = self._new_int_array_wrapper()
        for _ in range(64):
            buffer = IndentedBuffer()
            var = wrapper.codegen_int_array_var("{2, 3}", buffer.writeline)
            self.assertIn(f"{var}[] = {{2, 3}};", buffer.getvalue())
            del buffer

    def test_int_array_declared_once_per_live_writeline_target(self):
        wrapper = self._new_int_array_wrapper()
        buffer = IndentedBuffer()

        first = wrapper.codegen_int_array_var("{2, 3}", buffer.writeline)
        second = wrapper.codegen_int_array_var("{2, 3}", buffer.writeline)

        self.assertEqual(first, second)
        self.assertEqual(buffer.getvalue().count(f"{first}[] = {{2, 3}};"), 1)

    def test_int_array_distinct_per_writeline_target(self):
        wrapper = self._new_int_array_wrapper()
        first_buffer = IndentedBuffer()
        second_buffer = IndentedBuffer()

        first = wrapper.codegen_int_array_var("{2, 3}", first_buffer.writeline)
        second = wrapper.codegen_int_array_var("{2, 3}", second_buffer.writeline)

        self.assertNotEqual(first, second)
        self.assertIn(f"{first}[] = {{2, 3}};", first_buffer.getvalue())
        self.assertIn(f"{second}[] = {{2, 3}};", second_buffer.getvalue())


if __name__ == "__main__":
    run_tests()
