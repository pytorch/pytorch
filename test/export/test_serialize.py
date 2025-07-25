"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_sym_bool)
"""

# Owner(s): ["oncall: export"]
import copy
import io
import math
import tempfile
import unittest
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import torch
import torch._dynamo as torchdynamo
import torch._export.serde.schema as schema
import torch.export._trace
import torch.utils._pytree as pytree
from torch._export.db.case import ExportCase, SupportLevel
from torch._export.db.examples import all_examples
from torch._export.serde.serialize import (
    _to_json_bytes,
    canonicalize,
    deserialize,
    ExportedProgramDeserializer,
    ExportedProgramSerializer,
    GraphModuleSerializer,
    serialize,
    SerializeError,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import Dim, export_for_training, load, save, unflatten
from torch.export.pt2_archive.constants import ARCHIVE_VERSION_PATH
from torch.fx.experimental.symbolic_shapes import is_concrete_int, ValueRanges
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    TemporaryFileName,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations


def get_filtered_export_db_tests():
    return [
        (name, case)
        for name, case in all_examples().items()
        if case.support_level == SupportLevel.SUPPORTED
    ]


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    def test_export_with_extension_op_serialization(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        class FooExtensionOp:
            def __hash__(self):
                return 0

            def __eq__(self, other):
                return type(other) == type(self)

            def __call__(self, *args, **kwargs):
                return torch.ops.aten.add.Tensor(*args, **kwargs)

            @property
            def __name__(self):
                return "foo.my_op"

        class ExtensionVerifier(torch._export.verifier.Verifier):
            dialect = "FOO"

            def allowed_op_types(self):
                return super().allowed_op_types() + (FooExtensionOp,)

        class FooExtensionHandler(torch._export.serde.serialize.ExtensionHandler):
            @classmethod
            def namespace(cls):
                return "foo"

            @classmethod
            def to_op_name(cls, op):
                return "my_op"

            @classmethod
            def from_op_name(cls, name: str):
                self.assertEqual(name, "my_op")
                return FooExtensionOp()

            @classmethod
            def op_schema(cls, op):
                return torch.ops.aten.add.Tensor._schema

        inp = (torch.ones(10),)
        ep = export_for_training(TestModule(), inp, strict=True)

        # Register the custom op handler.
        foo_custom_op = FooExtensionOp()
        torch._export.serde.serialize.register_extension(
            FooExtensionOp, FooExtensionHandler
        )

        new_gm = copy.deepcopy(ep.graph_module)
        # Inject the custom operator.
        for node in new_gm.graph.nodes:
            if node.name == "add":
                node.target = foo_custom_op

        new_ep = ep._update(new_gm, ep.graph_signature, verifiers=[ExtensionVerifier])
        serialized = serialize(new_ep)
        deserialized = deserialize(serialized)
        self.assertEqual(
            len(
                deserialized.graph.find_nodes(op="call_function", target=foo_custom_op)
            ),
            1,
        )

    def test_predispatch_export_with_autograd_op(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                with torch.enable_grad():
                    return x + x

        inp = (torch.ones(10),)
        with torch.no_grad():
            from torch.export._trace import _export

            ep = _export(Foo(), inp, pre_dispatch=True)

        buffer = io.BytesIO()
        torch.export.save(ep, buffer)
        buffer.seek(0)
        loaded_ep = torch.export.load(buffer)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)
        self.assertEqual(exp_out.requires_grad, actual_out.requires_grad)

    def test_export_example_inputs_preserved(self):
        class MyModule(torch.nn.Module):
            """A test module with that has multiple args and uses kwargs"""

            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(2, 3))

            def forward(self, x, y, use_p=False):
                out = x + y
                if use_p:
                    out += self.p
                return out

        model = MyModule().eval()
        random_inputs = (torch.rand([2, 3]), torch.rand([2, 3]))
        exp_program = export_for_training(
            model, random_inputs, {"use_p": True}, strict=True
        )

        output_buffer = io.BytesIO()
        # Tests that example inputs are preserved when saving and loading module.
        torch.export.save(exp_program, output_buffer)
        loaded_model = torch.export.load(output_buffer)
        # Extract the example inputs from before and after saving.
        orig_args, orig_kwargs = exp_program.example_inputs
        loaded_args, loaded_kwargs = loaded_model.example_inputs
        # Run both modules and confirm that outputs match.
        orig_out = exp_program.module()(*orig_args, **orig_kwargs)
        loaded_out = loaded_model.module()(*loaded_args, **loaded_kwargs)
        self.assertEqual(orig_out, loaded_out)

    def test_metadata_run_decomp_serder(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        exp_program = export_for_training(M(), (torch.randn(4, 4),), strict=True)

        output_buffer = io.BytesIO()
        # Tests that example forward arg names are preserved when saving and loading module.
        torch.export.save(exp_program, output_buffer)
        loaded_model = torch.export.load(output_buffer)

        ep = loaded_model.run_decompositions({})
        # We should preserve the original module name
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, x):
    sin = torch.ops.aten.sin.default(x);  x = None
    return (sin,)""",
        )

    def test_metadata_parsing_with_layer_split(self):
        # Tests that modules with more complicated layer patterns can be serialized
        # and deserialized correctly.
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                )

            def forward(self, x):
                # Splitting layers of a sequential stack introduces commas and parens
                # into metadata trace.
                out_start, out_rest = self.layers[0], self.layers[1:]
                h = out_start(x)
                h = out_rest(h)
                return h

        inp = (torch.ones(10),)
        # Module will only be able to roundtrip if metadata
        # can be correctly parsed.
        ep = export_for_training(MyModule(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        # Check that both modules run to confirm load was successful.
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_nested_layer_split(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                    torch.nn.SiLU(),
                )

            def forward(self, x):
                out_start, out_rest = self.layers[0], self.layers[1:]
                h = out_start(x)
                h = out_rest(h) + 2
                return h

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_module("a[(1)]", Bar())
                self.register_module("b[(2)]", Bar())
                self.register_buffer("c:[22]", torch.randn(1))

            def forward(self, x):
                out_a, out_b = getattr(self, "a[(1)]"), getattr(self, "b[(2)]")
                out_c = getattr(self, "c:[22]")
                h = out_a(x)
                h = out_b(h)
                return h + out_c

        inp = (torch.ones(10),)
        ep = export_for_training(Foo(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        # Check that both modules run to confirm load was successful.
        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_serialize_constant_outputs(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                # Along with tensor output, return Nonetype
                # and constant. Although these outputs aren't
                # very useful, they do show up in graphs.
                return x + 1, None, 1024

        # Check that module can be roundtripped, thereby confirming proper deserialization.
        inp = (torch.ones(10),)
        ep = export_for_training(MyModule(), inp, strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        loaded_ep = load(buffer)

        exp_out = ep.module()(*inp)
        actual_out = loaded_ep.module()(*inp)
        self.assertEqual(exp_out, actual_out)

    def test_serialize_multiple_returns_from_node(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w, b):
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        exported_module = export_for_training(
            MyModule(),
            (
                torch.ones([512, 512], requires_grad=True),
                torch.ones([512]),
                torch.ones([512]),
            ),
            strict=True,
        ).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.native_layer_norm.default")
        # aten::native_layer_norm returns 3 tensors
        self.assertEqual(len(node.outputs), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_serialize_sym_int(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dim1_bc = torch.export.Dim("dim1_b")
        dynamic_shapes = {
            "a": {0: dim0_ac},
            "b": {1: dim1_bc},
            "c": {0: dim0_ac, 1: dim1_bc},
        }
        exported_module = export_for_training(
            DynamicShapeSimpleModel(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        ).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)
        sym_size_nodes = [
            node
            for node in serialized.exported_program.graph_module.graph.nodes
            if node.target == "torch.ops.aten.sym_size.int"
        ]
        for node in sym_size_nodes:
            self.assertEqual(node.inputs[0].name, "self")
            self.assertEqual(node.inputs[1].name, "dim")

    def test_serialize_sym_float(self) -> None:
        # TODO(rec): This doesn't seem to test anything!

        class DynamicFloatSimpleModel(torch.nn.Module):
            def __init__(self, multiplier: torch.SymFloat):
                super().__init__()
                self.multiplier = multiplier

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                e = d * self.multiplier
                e_s0 = e.shape[0]
                e_s1 = e.shape[1]
                e_s3 = e_s0 * e_s1
                f = e.view(e_s3)
                return torch.cat([f, f])

        multiplier_sym = torch.SymFloat("multiplier_sym")
        _model = DynamicFloatSimpleModel(multiplier_sym)
        _inputs = (
            torch.randn(2, 4),
            torch.randn(4, 7),
            torch.randn(2, 7),
        )
        _dim0_ac = Dim("dim0_ac")
        _dim1_bc = Dim("dim1_b")

    def test_serialize_infinite_sym_int(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dim1_bc = torch.export.Dim("dim1_b")
        dynamic_shapes = {
            "a": {0: dim0_ac},
            "b": {1: dim1_bc},
            "c": {0: dim0_ac, 1: dim1_bc},
        }
        exported_module = export_for_training(
            DynamicShapeSimpleModel(),
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=True,
        ).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)
        for v in serialized.exported_program.range_constraints.values():
            self.assertEqual(v.max_val, None)

    def test_symint_list(self):
        # This reflects the behavior from inductor's ExternFallbackNode
        shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
        symint = shape_env.create_unbacked_symint()
        serializer = GraphModuleSerializer(None, None)  # type: ignore[arg-type]
        res = serializer.serialize_inputs(
            torch.ops.aten.ones.default, ([1, symint, 3],), {}
        )
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].arg._type, "as_sym_ints")

    def test_serialize_list_returns(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.split(x, 2)

        input = torch.arange(10.0).reshape(5, 2)
        exported_module = export_for_training(
            MyModule(), (input,), strict=True
        ).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        # split.Tensor gets decomposed to split_with_sizes by the core ATen decomposition table
        self.assertEqual(node.target, "torch.ops.aten.split_with_sizes.default")
        self.assertEqual(len(node.outputs), 1)
        # Input looks like:
        # tensor([[0, 1],
        #         [2, 3],
        #         [4, 5],
        #         [6, 7],
        #         [8, 9]])
        # Output looks like:
        # (tensor([[0, 1],
        #          [2, 3]]),
        #  tensor([[4, 5],
        #          [6, 7]]),
        #  tensor([[8, 9]]))
        self.assertEqual(len(node.outputs[0].as_tensors), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs[0].as_tensors:
            name = output.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_nonfinite_inputs(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                x = torch.ops.aten.add.Scalar(x, math.inf)
                x = torch.ops.aten.add.Scalar(x, -math.inf)
                return torch.ops.aten.add.Scalar(x, math.nan)

        fn = Module()
        ep = torch.export.export(
            fn,
            (torch.randn(3, 2),),
        )
        json_bytes = _to_json_bytes(
            ExportedProgramSerializer().serialize(ep).exported_program
        )
        import json

        def parse_constant(x):
            raise RuntimeError(f"Invalid JSON float: {x}")

        json.loads(json_bytes, parse_constant=parse_constant)

    def test_multi_return_some_unused(self) -> None:
        """
        Make sure the serialized output matches the op schema, even if some of
        the arguments are never used in the graph.
        """

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.var_mean.correction(x, [1])[0]

        exported_module = export_for_training(
            MyModule(), (torch.ones([512, 512], requires_grad=True),), strict=True
        ).run_decompositions()

        serialized = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.var_mean.correction")
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_rational_ranges(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return x + x

        ep = export_for_training(
            M(), (torch.randn(4),), dynamic_shapes=({0: Dim("temp")},), strict=True
        )

        range_constraints = list(ep.range_constraints.keys())
        assert len(range_constraints) == 1
        symint = range_constraints[0]

        import sympy

        upper_range = sympy.Rational(10, 3)
        lower_range = sympy.Rational(10, 6)
        ep.range_constraints[symint] = ValueRanges(lower=lower_range, upper=upper_range)

        serialized = ExportedProgramSerializer().serialize(ep)
        self.assertEqual(
            serialized.exported_program.range_constraints[symint.name].min_val, 2
        )
        self.assertEqual(
            serialized.exported_program.range_constraints[symint.name].max_val, 3
        )

    def test_kwargs_default(self) -> None:
        """
        Tests that the kwargs default values are serialized even if they are not
        specified
        """

        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                values = torch.randn(3, 2)
                return torch.searchsorted(x, values, side="right", right=True)

        f = Foo()

        x, _ = torch.sort(torch.randn(3, 4))
        exported_module = export_for_training(f, (x,), strict=True).run_decompositions()
        serialized = ExportedProgramSerializer().serialize(exported_module)

        node = serialized.exported_program.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch.ops.aten.searchsorted.Tensor")
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(node.inputs[2].name, "right")
        self.assertEqual(node.inputs[2].arg.as_bool, True)
        self.assertEqual(node.inputs[3].name, "side")
        self.assertEqual(node.inputs[3].arg.as_string, "right")

    def test_canonicalize(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                a = y + x
                b = x + y
                return b + a

        ep = export_for_training(
            Module(), (torch.randn(3, 2), torch.randn(3, 2)), strict=True
        )
        s = ExportedProgramSerializer().serialize(ep)
        c = canonicalize(s.exported_program)
        g = c.graph_module.graph
        self.assertLess(
            g.nodes[0].inputs[0].arg.as_tensor.name,
            g.nodes[1].inputs[0].arg.as_tensor.name,
        )

    def test_int_list(self) -> None:
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.sum.dim_IntList(x, [])

        ep = torch.export.export_for_training(M(), (torch.randn(3, 2),), strict=True)
        serialized = ExportedProgramSerializer().serialize(ep)
        for node in serialized.exported_program.graph_module.graph.nodes:
            if "aten.sum.dim_IntList" in node.target:
                self.assertEqual(node.inputs[1].arg.type, "as_ints")


@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestDeserialize(TestCase):
    def setUp(self):
        super().setUp()
        init_torchbind_implementations()

    def _check_graph_nodes(self, gm1, gm2, _check_meta=True):
        # TODO: The _check_meta flag bypasses checking for
        # source_fn/nn_module_stack as there is an issue with
        # roundtripping the source_fn value on torch.ops.map nodes
        # original source_fn: <functorch.experimental._map.MapWrapper object at 0x7f80a0549930>
        # deserialized source_fn: 'functorch.experimental._map.map'

        self.assertEqual(len(gm1.graph.nodes), len(gm2.graph.nodes))

        for node1, node2 in zip(gm1.graph.nodes, gm2.graph.nodes):
            self.assertEqual(node1.op, node2.op)
            if node1.op == "call_function":
                # Check "val" metadata
                val1 = node1.meta.get("val", None)
                val2 = node2.meta.get("val", None)
                self.assertEqual(len(node1.args), len(node2.args))
                self.assertEqual(set(node1.kwargs.keys()), set(node2.kwargs.keys()))
                if val1 is None or val2 is None:
                    # Either both are None
                    self.assertEqual(val1, val2)
                elif isinstance(val1, FakeTensor) and isinstance(val2, FakeTensor):
                    # Or both are fake tensors with the same shape/dtype
                    self.assertEqual(len(val1.shape), len(val2.shape))
                    for s1, s2 in zip(val1.shape, val2.shape):
                        if is_concrete_int(s1) and is_concrete_int(s2):
                            self.assertEqual(s1, s2)
                        else:
                            self.assertEqual(str(s1), str(s2))
                    self.assertEqual(val1.dtype, val2.dtype)
                elif isinstance(val1, (list, tuple)) and isinstance(
                    val2, (list, tuple)
                ):
                    # Or both are fake tensors lists with one element and with the
                    # same shape/dtype
                    for v1, v2 in zip(
                        pytree.tree_leaves(val1), pytree.tree_leaves(val2)
                    ):
                        if isinstance(v1, FakeTensor):
                            self.assertEqual(v1.shape, v2.shape)
                            self.assertEqual(v1.dtype, v2.dtype)
                else:
                    # For expressions like 's0 < 10' can only compare through string
                    self.assertEqual(str(val1), str(val2))

                # Check "stack_trace" metadata
                self.assertEqual(
                    node1.meta.get("stack_trace", None),
                    node2.meta.get("stack_trace", None),
                )

                if node1.target == torch.ops.higher_order.cond:
                    true_graph1 = getattr(gm1, node1.args[1].target)
                    true_graph2 = getattr(gm2, node2.args[1].target)
                    self._check_graph_nodes(true_graph1, true_graph2)

                    false_graph1 = getattr(gm1, node1.args[2].target)
                    false_graph2 = getattr(gm2, node2.args[2].target)
                    self._check_graph_nodes(false_graph1, false_graph2)
                elif node1.target == torch.ops.higher_order.map_impl:
                    map_graph1 = getattr(gm1, node1.args[0].target)
                    map_graph2 = getattr(gm2, node2.args[0].target)
                    self._check_graph_nodes(map_graph1, map_graph2, False)

            if _check_meta and node1.op not in ("get_attr", "placeholder", "output"):
                # Check "nn_module_stack" metadata
                self.assertEqual(
                    node1.meta.get("nn_module_stack", None),
                    node2.meta.get("nn_module_stack", None),
                )
                # Check "source_fn_stack" metadata
                self.assertEqual(
                    node1.meta.get("source_fn_stack", None),
                    node2.meta.get("source_fn_stack", None),
                )

    def check_graph(
        self,
        fn,
        inputs,
        dynamic_shapes=None,
        _check_meta=True,
        use_pre_dispatch=True,
        strict=True,
    ) -> None:
        """Export a graph, serialize it, deserialize it, and compare the results."""

        def _deepcopy_inputs(inputs):
            # copy.deepcopy(deepcopy) can fail if tensor inputs have attribute (i.e. __dict__).
            # we remove __dict__ when deepcopying.
            dict_mapping = dict()
            inputs_clone = ()
            for idx, i in enumerate(inputs):
                if isinstance(i, torch.Tensor) and hasattr(inputs[0], "__dict__"):
                    dict_mapping[idx] = i.__dict__
                    i.__dict__ = {}
                inputs_clone += (copy.deepcopy(i),)

            # Add __dict__ back.
            for k, v in dict_mapping.items():
                inputs[k].__dict__ = v
                inputs_clone[k].__dict__ = v
            return inputs_clone

        def _check_graph(pre_dispatch):
            if pre_dispatch:
                ep = torch.export.export_for_training(
                    fn,
                    _deepcopy_inputs(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
            else:
                # We should have this branch because
                # PT2 Inference goes through this private
                # export API.
                ep = torch.export._trace._export(
                    fn,
                    _deepcopy_inputs(inputs),
                    {},
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                    pre_dispatch=False,
                )
            ep.graph.eliminate_dead_code()

            serialized_artifact = serialize(ep, opset_version={"aten": 0})
            deserialized_ep = deserialize(
                serialized_artifact, expected_opset_version={"aten": 0}
            )
            deserialized_ep.graph.eliminate_dead_code()

            orig_outputs = ep.module()(*_deepcopy_inputs(inputs))
            loaded_outputs = deserialized_ep.module()(*_deepcopy_inputs(inputs))

            flat_orig_outputs = pytree.tree_leaves(orig_outputs)
            flat_loaded_outputs = pytree.tree_leaves(loaded_outputs)

            for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
                self.assertEqual(type(orig), type(loaded))
                # torch.allclose doesn't work for float8
                if isinstance(orig, torch.Tensor) and orig.dtype not in [
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ]:
                    if orig.is_meta:
                        self.assertEqual(orig, loaded)
                    else:
                        self.assertTrue(torch.allclose(orig, loaded))
                else:
                    self.assertEqual(orig, loaded)
            self._check_graph_nodes(
                ep.graph_module, deserialized_ep.graph_module, _check_meta
            )

        if use_pre_dispatch:
            _check_graph(pre_dispatch=True)
            _check_graph(pre_dispatch=False)
        else:
            _check_graph(pre_dispatch=False)

    def test_optional_tuple(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo",
                "(Tensor a, Tensor b, Tensor? c) -> (Tensor, Tensor?)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo", "cpu", lib=lib)
            @torch.library.impl_abstract("mylib::foo")
            def foo_impl(a, b, c):
                res2 = None
                if c is not None:
                    res2 = c + a + b
                return a + b, res2

            class M(torch.nn.Module):
                def forward(self, a, b, c):
                    return torch.ops.mylib.foo(a, b, c)

            self.check_graph(M(), (torch.randn(3), torch.randn(3), torch.randn(3)))

    def test_unbacked_bindings_serialize(self):
        from torch._export.utils import _get_shape_env_from_gm
        from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT

        class M(torch.nn.Module):
            def forward(self, x, y):
                x += 2
                n = x.item()
                n = n * 2 + y.item()
                return n + 2

        inps = (
            torch.tensor(4),
            torch.tensor(5),
        )
        for _strict in [True, False]:
            ep = torch.export.export(M(), inps, strict=_strict).run_decompositions()

            # check bindings after deserialization
            buffer = io.BytesIO()
            save(ep, buffer)
            buffer.seek(0)
            loaded_ep = load(buffer)
            bound = set()
            for old_node, new_node in zip(ep.graph.nodes, loaded_ep.graph.nodes):
                self.assertEqual(
                    "unbacked_bindings" in old_node.meta,
                    "unbacked_bindings" in new_node.meta,
                )
                bound.update(new_node.meta.get("unbacked_bindings", {}))

            # check ShapeEnv counters
            shape_env = _get_shape_env_from_gm(loaded_ep.graph_module)
            next_index = next(shape_env.unbacked_symint_counter)
            for symbol in bound:
                self.assertTrue(symbol_is_type(symbol, SymT.UNBACKED_INT))
                self.assertTrue(
                    int(str(symbol)[len(prefix_str[SymT.UNBACKED_INT]) :]) < next_index
                )

    def test_sym_bool_dynamic_shapes(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                z = x[:, -y.shape[0] :, :]
                return z

        inputs = (torch.ones(4, 5, 10), torch.ones(3))
        dynamic_shapes = {"x": {}, "y": {0: Dim("seqlen", max=4)}}
        # Compile with dynamic_shapes set to get operator.neg involved
        self.check_graph(MyModule(), inputs, dynamic_shapes=dynamic_shapes)

    def test_auto_functionalize(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::foo1",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> Tensor",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::foo2",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> (Tensor, Tensor)",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )
            torch.library.define(
                "mylib::foo3",
                "(Tensor(a!) x, Tensor[] y, Tensor(b!) z, SymInt w, Tensor n) -> ()",
                tags=torch.Tag.pt2_compliant_tag,
                lib=lib,
            )

            @torch.library.impl("mylib::foo1", "cpu", lib=lib)
            @torch.library.impl_abstract("mylib::foo1")
            def foo1_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return n + n

            @torch.library.impl("mylib::foo2", "cpu", lib=lib)
            @torch.library.impl_abstract("mylib::foo2")
            def foo2_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return (n + n, n * n)

            @torch.library.impl("mylib::foo3", "cpu", lib=lib)
            @torch.library.impl_abstract("mylib::foo3")
            def foo3_impl(x, y, z, w, n):
                x.add_(y[0] + w)
                z.add_(y[1] + n)
                return

            class M(torch.nn.Module):
                def forward(self, x, y, z, n):
                    n = torch.ops.mylib.foo1(x, y, z, 2, n)
                    torch.ops.mylib.foo3(x, y, z, 2, n)
                    return torch.ops.mylib.foo2(x, y, z, 2, n)

            x = torch.randn(3)
            y = (torch.randn(3), torch.randn(3))
            z = torch.randn(3)
            n = torch.randn(3)
            orig_args = (x, y, z, n)

            # TODO Auto_functionalize is not supported on pre_dispatch IR
            self.check_graph(M(), orig_args, use_pre_dispatch=False)

    def test_hoo_symint_input(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                num = c.item()
                return torch.cond(
                    pred=torch.tensor([True]),
                    true_fn=lambda a, b: a + b + num,
                    false_fn=lambda a, b: a - b - num,
                    operands=(a, b),
                )

        inp = (torch.ones(3, 3), torch.ones(3, 3), torch.tensor(2))
        self.check_graph(Mod(), inp, use_pre_dispatch=False)

    def test_none_input(self):
        """
        Testing a backwards-compatibility breakage where old models do not have
        an input spec with the node name.
        """

        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + z

        ep = torch.export.export(M(), (torch.ones(3, 3), None, torch.ones(3, 3)))

        serialized_program = ExportedProgramSerializer(None, 2).serialize(ep)
        serialized_program.exported_program.graph_module.signature.input_specs[1] = (
            schema.InputSpec.create(
                user_input=schema.UserInputSpec(
                    arg=schema.Argument.create(as_none=True)
                )
            )
        )
        ep = ExportedProgramDeserializer(None).deserialize(
            serialized_program.exported_program, {}, {}, {}
        )
        ep.graph_module.recompile()
        unflattened = torch.export.unflatten(ep)
        inp = (torch.rand(3, 3), None, torch.rand(3, 3))
        self.assertEqual(unflattened(*inp), M()(*inp))

    def test_multi_return(self) -> None:
        """
        Test multiple return from a single node (ex. layer_norm has 2 outputs)
        """

        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w, b):
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        inputs = (
            torch.ones([512, 512], requires_grad=True),
            torch.ones([512]),
            torch.ones([512]),
        )
        self.check_graph(MyModule(), inputs)

    def test_basic(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                x = x + x
                x = x * x
                x = x / x
                return x, x.clone()

        inputs = (torch.ones([512], requires_grad=True),)
        self.check_graph(MyModule(), inputs)

    def test_dynamic(self) -> None:
        class DynamicShapeSimpleModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, a, b, c) -> torch.Tensor:
                d = (torch.matmul(a, b) + c) / 2
                d_s0 = d.shape[0]
                d_s1 = d.shape[1]
                d_s3 = d_s0 * d_s1
                e = d.view(d_s3)
                return torch.cat([e, e])

        inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
        dim0_ac = torch.export.Dim("dim0_ac")
        dynamic_shapes = {"a": {0: dim0_ac}, "b": None, "c": {0: dim0_ac}}
        self.check_graph(DynamicShapeSimpleModel(), inputs, dynamic_shapes)

    def test_sym_bool(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                assert x.size(0) in y
                return x + y

        f = Module()
        self.check_graph(f, (torch.ones(1), torch.ones(3)))

    def test_sym_bool_torch_check_equal(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.size(0)
                torch._check_is_size(z)
                torch._check(z == 2)
                return y

        self.check_graph(Module(), (torch.Tensor([1, 0, 1, 0]),))

    def test_sym_int_torch_check_equal(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.size(0)
                torch._check_is_size(z)
                torch._check(z % 3 == 0)
                torch._check(z == 3)
                return y

        self.check_graph(Module(), (torch.Tensor([1, 0, 1, 0, 1, 0]),))

    def test_shape(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                z, y = x.size()
                return z + y + x[0], z

        inputs = (torch.ones(2, 3),)
        dim0_x, dim1_x = torch.export.dims("dim0_x", "dim1_x")
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        self.check_graph(Foo(), inputs, dynamic_shapes)

    def test_module(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(3, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear1(x)
                x = torch.nn.functional.relu(x)
                x = self.linear2(x)
                return x

        inputs = (torch.randn(3, 3),)
        self.check_graph(M(), inputs)

    def test_module_meta(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.Parameter(torch.ones(3, 3))

            def forward(self, x):
                return self.p + x

        with torch.device("meta"):
            mod = M()

        inputs = (torch.randn(3, 3, device="meta"),)
        self.check_graph(mod, inputs)

    def test_pytree_namedtuple(self):
        N1 = namedtuple("N1", ["a", "b"])

        class N2(NamedTuple):
            a: torch.Tensor
            b: torch.Tensor

        class M(torch.nn.Module):
            def forward(self, x, y):
                return N2(x.a + y.a, x.b * y.b)

        pytree._register_namedtuple(
            N1,
            serialized_type_name="test.export.test_serialize.test_pytree_namedtuple.N1",
        )
        pytree._register_namedtuple(
            N2,
            serialized_type_name="test.export.test_serialize.test_pytree_namedtuple.N2",
        )

        inp = (N1(torch.randn(3), torch.randn(3)), N1(torch.randn(3), torch.randn(3)))
        ep = torch.export.export(M(), inp)
        ep.example_inputs = None  # Can't pickle the input since the namedtuple class is not at a global namespace
        serialized = ExportedProgramSerializer().serialize(ep)
        self.assertEqual(
            len(serialized.exported_program.graph_module.treespec_namedtuple_fields), 2
        )
        deserialized = ExportedProgramDeserializer().deserialize(
            serialized.exported_program,
            serialized.state_dict,
            serialized.constants,
        )
        self.assertTrue("treespec_namedtuple_fields" in deserialized.graph_module.meta)
        self.assertEqual(
            deserialized.graph_module.meta["treespec_namedtuple_fields"],
            {
                "test.export.test_serialize.test_pytree_namedtuple.N1": ["a", "b"],
                "test.export.test_serialize.test_pytree_namedtuple.N2": ["a", "b"],
            },
        )

        unlifted = deserialized.module()
        self.assertTrue("treespec_namedtuple_fields" in unlifted.meta)
        self.assertEqual(len(unlifted.meta["treespec_namedtuple_fields"]), 2)

        unflattened = unflatten(deserialized)
        self.assertTrue("treespec_namedtuple_fields" in unflattened.meta)
        self.assertEqual(len(unflattened.meta["treespec_namedtuple_fields"]), 2)

    def test_cond(self):
        from functorch.experimental.control_flow import cond

        inputs = torch.ones(4, 3), torch.zeros(4, 3)

        class M(torch.nn.Module):
            def forward(self, x, y):
                def t(x, y):
                    return x + y

                def f(x, y):
                    return x - y

                return cond(x[0][0] > 4, t, f, [x, y])

        self.check_graph(M(), inputs)

    def test_sym_float(self):
        class M(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                return b * 0.1

        self.check_graph(M(), (torch.tensor(1.0),))

    def test_arg_from(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("compress_weight", torch.ones((10, 10)))
                self.register_buffer("compress_bias", torch.ones(10))

            def forward(self) -> None:
                if self.compress_weight is None or self.compress_bias is None:
                    return
                torch.nn.init.kaiming_uniform_(self.compress_weight, a=math.sqrt(5))
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                    self.compress_weight
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.compress_bias, -bound, bound)

        with torch.no_grad():
            self.check_graph(M(), ())

    def test_map(self):
        from functorch.experimental import control_flow

        def f(x, y):
            return x + y

        class Module(torch.nn.Module):
            def forward(self, xs, y):
                return control_flow.map(f, xs, y)

        g = Module()
        inputs = (torch.ones(3, 2, 2), torch.ones(2))
        self.check_graph(g, inputs, _check_meta=False)

    def test_positional_argument_with_default_value(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(10, 10)
                self.bias = torch.randn(10)

            def forward(self, x):
                # bias has an default value here but it should be preserved
                # as a positional argument.
                return torch.ops.aten.linear.default(x, self.weight, self.bias)

        self.check_graph(MyLinear(), (torch.randn(10, 10),))

    def test_tensor_tensor_list(self):
        with torch.library._scoped_library("_export", "FRAGMENT") as lib:
            lib.define(
                "_test_tensor_tensor_list_output(Tensor x, Tensor y) -> (Tensor, Tensor[])",
                tags=torch.Tag.pt2_compliant_tag,
            )

            def _test_tensor_tensor_list_output(x, y):
                return y, [x]

            lib.impl(
                "_test_tensor_tensor_list_output",
                _test_tensor_tensor_list_output,
                "CPU",
            )
            lib.impl(
                "_test_tensor_tensor_list_output",
                _test_tensor_tensor_list_output,
                "Meta",
            )

            class M(torch.nn.Module):
                def forward(self, x, y):
                    a, b = torch.ops._export._test_tensor_tensor_list_output.default(
                        x, y
                    )
                    return a + b[0]

            self.check_graph(M(), (torch.rand(3, 2), torch.rand(3, 2)))

    def test_list_of_optional_tensors(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y, z):
                indices = [None, None, torch.tensor([1, 3, 5, 7])]
                indexed = torch.ops.aten.index.Tensor(x + y, indices)
                return indexed + z

        inputs = (torch.rand(8, 8, 8), torch.rand(8, 8, 8), torch.rand(8, 8, 4))
        self.check_graph(MyModule(), inputs)

    def test_sym_ite(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                b = x.shape[0] == 5
                ret = torch.sym_ite(b, x.shape[0], x.shape[1])
                return ret

        dynamic_shapes = {"x": {0: Dim("dim0"), 1: Dim("dim1")}}
        self.check_graph(Foo(), (torch.ones(4, 5),), dynamic_shapes=dynamic_shapes)

    def test_multiple_getitem(self):
        class M(torch.nn.Module):
            def forward(self, x):
                a, b = torch.topk(x, 2)
                a = a * 2
                return a, b

        ep = torch.export.export_for_training(M(), (torch.ones(3),), strict=True)

        # insert another getitem node
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor:
                getitem_0 = node.args[0]
                with ep.graph.inserting_before(getitem_0):
                    getitem_copy = ep.graph.node_copy(getitem_0)
                    mul_node = ep.graph.call_function(
                        torch.ops.aten.mul.Tensor, (getitem_copy, 2)
                    )
                    mul_node.meta = copy.copy(getitem_copy.meta)
                    node.args = (getitem_0, mul_node)

        deserialized_ep = deserialize(serialize(ep))

        inp = (torch.randn(3),)
        orig_res = ep.module()(*inp)
        res = deserialized_ep.module()(*inp)
        self.assertTrue(torch.allclose(orig_res[0], res[0]))
        self.assertTrue(torch.allclose(orig_res[1], res[1]))

        # The deserialized graph should have deduped getitem calls
        self.assertExpectedInline(
            deserialized_ep.graph_module.code.strip("\n"),
            """\
def forward(self, x):
    topk_default = torch.ops.aten.topk.default(x, 2);  x = None
    getitem = topk_default[0]
    getitem_1 = topk_default[1];  topk_default = None
    mul_tensor = torch.ops.aten.mul.Tensor(getitem, 2)
    mul = torch.ops.aten.mul.Tensor(getitem, mul_tensor);  getitem = mul_tensor = None
    return (mul, getitem_1)
    """,
        )

    @parametrize(
        "name,case",
        get_filtered_export_db_tests(),
        name_fn=lambda name, case: f"case_{name}",
    )
    def test_exportdb_supported(self, name: str, case: ExportCase) -> None:
        model = case.model
        _check_meta = "map" not in name
        self.check_graph(model, case.example_args, _check_meta=_check_meta)

    def test_constraints(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n)
                return y.sum() + torch.ones(n, 5).sum()

        f = Module()
        self.check_graph(f, (torch.tensor(3), torch.randn(4, 5)))

    def test_get_attr(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor(3)

        f = Module()
        self.check_graph(f, (torch.tensor(3),))

    def test_get_attr_list(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                return torch.cat([x, torch.tensor([1, 1])])

        f = Module()
        self.check_graph(f, (torch.tensor([1, 1]),))

    @unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
    def test_device(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                conv = self.conv(x)
                relu = self.relu(conv)
                mul = relu * 0.5
                return mul

        inp = torch.randn((1, 3, 224, 224), dtype=torch.float).to("cuda")
        model = MyModule().eval().cuda()
        self.check_graph(model, (inp,))

    def test_custom_obj_tuple_out(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        self.check_graph(m, inputs, strict=False)

    def test_custom_obj(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo(self.attr, x)
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, a)
                return x + b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        self.check_graph(m, inputs, strict=False)

    def test_custom_obj_list_out(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                a = torch.ops._TorchScriptTesting.takes_foo_list_return(self.attr, x)
                y = a[0] + a[1] + a[2]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return x + b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        self.check_graph(m, inputs, strict=False)

    def test_export_no_inputs(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = torch.ones(3, 3)

            def forward(self):
                return self.p * self.p

        ep = torch.export.export_for_training(M(), (), strict=True)
        ep._example_inputs = None
        roundtrip_ep = deserialize(serialize(ep))
        self.assertTrue(torch.allclose(ep.module()(), roundtrip_ep.module()()))

    def test_serialize_float8(self):
        for dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:

            class MyModule(torch.nn.Module):
                def forward(self, x):
                    return x.to(dtype)

            m = MyModule()
            inputs = (torch.ones(2, 3),)
            self.check_graph(m, inputs, strict=False)


instantiate_parametrized_tests(TestDeserialize)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSchemaVersioning(TestCase):
    def test_error(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Module()
        ep = export_for_training(f, (torch.randn(1, 3),), strict=True)

        serialized_program = ExportedProgramSerializer().serialize(ep)
        serialized_program.exported_program.schema_version.major = -1
        with self.assertRaisesRegex(
            SerializeError, r"Serialized schema version .* does not match our current"
        ):
            ExportedProgramDeserializer().deserialize(
                serialized_program.exported_program,
                serialized_program.state_dict,
                serialized_program.constants,
                serialized_program.example_inputs,
            )


# We didn't set up kwargs input yet
unittest.expectedFailure(TestDeserialize.test_exportdb_supported_case_fn_with_kwargs)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSaveLoad(TestCase):
    def test_save_buffer(self):
        inp = (torch.tensor([0.1, 0.1]),)

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = x + 1
                y = x.t()
                y = y.relu()
                y = self.linear(y)
                return y

        ep = export_for_training(Module(), inp, strict=True)

        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    @unittest.skipIf(IS_WINDOWS, "Cannot modify file in windows")
    def test_save_file(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * x

        f = Foo()

        inp = (torch.randn(2, 2),)
        ep = export_for_training(f, inp, strict=True)

        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            save(ep, f.name)
            f.seek(0)
            loaded_ep = load(f.name)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_save_path(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        f = Foo()

        inp = (torch.tensor([6]), torch.tensor([7]))
        ep = export_for_training(f, inp, strict=True)

        with TemporaryFileName(suffix=".pt2") as fname:
            path = Path(fname)
            save(ep, path)
            loaded_ep = load(path)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))

    def test_save_extra(self):
        inp = (torch.tensor([0.1, 0.1]),)

        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * x + x

        f = Foo()

        ep = export_for_training(f, inp, strict=True)

        buffer = io.BytesIO()
        save(ep, buffer, extra_files={"extra.txt": "moo"})
        buffer.seek(0)
        extra_files = {"extra.txt": ""}
        loaded_ep = load(buffer, extra_files=extra_files)

        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))
        self.assertEqual(extra_files["extra.txt"], "moo")

    @unittest.skipIf(
        IS_FBCODE or IS_MACOS or IS_WINDOWS, "The file path is different in fbcode CI"
    )
    def test_version_error(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        ep = export_for_training(f, (torch.randn(1, 3),), strict=True)

        with self.assertRaisesRegex(
            ValueError, r"Saved archive version -1 does not match our current"
        ):
            with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
                save(ep, f.name)
                f.seek(0)
                file_prefix = f.name.split("/")[2].split(".")[0]

                # Modify the version
                with zipfile.ZipFile(f, "a") as zipf:
                    zipf.writestr(f"{file_prefix}/{ARCHIVE_VERSION_PATH}", "-1")

                f.seek(0)
                load(f.name)

    def test_save_constants(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export_for_training(Foo(), (torch.tensor(1),), strict=True)
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)

        inp = (torch.tensor(1),)
        self.assertTrue(torch.allclose(ep.module()(*inp), loaded_ep.module()(*inp)))


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerializeCustomClass(TestCase):
    def setUp(self):
        super().setUp()
        init_torchbind_implementations()

    def test_custom_class(self):
        custom_obj = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export_for_training(f, inputs, strict=True)

        # Replace one of the values with an instance of our custom class
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                with ep.graph.inserting_before(node):
                    custom_node = ep.graph.call_function(
                        torch.ops._TorchScriptTesting.take_an_instance.default,
                        (custom_obj,),
                    )
                    custom_node.meta["val"] = torch.ones(4, 4)
                    custom_node.meta["torch_fn"] = (
                        "take_an_instance",
                        "take_an_instance",
                    )
                    arg0, _ = node.args
                    node.args = (arg0, custom_node)

        serialized_vals = serialize(ep)

        ep_str = serialized_vals.exported_program.decode("utf-8")
        assert "class_fqn" in ep_str
        assert custom_obj._type().qualified_name() in ep_str

        deserialized_ep = deserialize(serialized_vals)

        for node in deserialized_ep.graph.nodes:
            if (
                node.op == "call_function"
                and node.target
                == torch.ops._TorchScriptTesting.take_an_instance.default
            ):
                arg = node.args[0]
                self.assertTrue(isinstance(arg, torch._C.ScriptObject))
                self.assertEqual(arg._type(), custom_obj._type())
                self.assertEqual(arg.__getstate__(), custom_obj.__getstate__())
                self.assertEqual(arg.top(), 7)

    def test_custom_class_containing_fake_tensor(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.custom_obj = torch.classes._TorchScriptTesting._ContainsTensor(
                    torch.rand(2, 3)
                )

            def forward(self, x):
                return x + self.custom_obj.get()

        with FakeTensorMode():
            f = Foo()

        inputs = (torch.zeros(2, 3),)
        with enable_torchbind_tracing():
            ep = export_for_training(f, inputs, strict=False)

        serialized_vals = serialize(ep)
        ep = deserialize(serialized_vals)
        self.assertTrue(isinstance(ep.constants["custom_obj"].get(), FakeTensor))

    def test_custom_class_input_to_function(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x)

        with FakeTensorMode():
            f = Foo()

        inputs = (torch.zeros(2, 3),)
        with enable_torchbind_tracing():
            ep = export_for_training(f, inputs, strict=False)

        serialized_vals = serialize(ep)
        ep = deserialize(serialized_vals)
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, obj_attr, x):
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(obj_attr, x);  obj_attr = None
    add = torch.ops.aten.add.Tensor(x, takes_foo);  x = takes_foo = None
    return (add,)""",
        )
        self.assertTrue(isinstance(ep.constants["attr"], torch.ScriptObject))
        gm = ep.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    attr = self.attr
    takes_foo = torch.ops._TorchScriptTesting.takes_foo.default(attr, x);  attr = None
    add = torch.ops.aten.add.Tensor(x, takes_foo);  x = takes_foo = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )
        self.assertTrue(isinstance(gm.attr, torch.ScriptObject))

    def test_custom_tag_metadata_serialization(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export_for_training(f, inputs, strict=True)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        for node in new_gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"

        new_ep = ep._update(new_gm, ep.graph_signature)
        serialized_vals = serialize(new_ep)
        new_ep = deserialize(serialized_vals)

        self.assertEqual(new_ep.graph_module.meta["custom"]["f"], "bar")
        counter = 0
        for node in new_ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
        self.assertEqual(counter, 1)

    def test_custom_tag_metadata_decomp(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        f = Foo()

        inputs = (torch.ones(2, 2),)
        ep = export_for_training(f, inputs, strict=True)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        counter = 0
        for node in new_gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.linear.default
            ):
                counter += 1
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"
        self.assertEqual(counter, 1)

        new_ep = ep._update(new_gm, ep.graph_signature)
        new_ep = new_ep.run_decompositions()

        self.assertEqual(new_ep.graph_module.meta["custom"]["f"], "bar")
        counter = 0
        for node in new_ep.graph.nodes:
            if node.op == "call_function":
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
        self.assertTrue(counter > 1)

    def test_custom_tag_metadata_copy(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x

        f = Foo()

        inputs = (torch.zeros(4, 4),)
        ep = export_for_training(f, inputs, strict=True)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        for node in new_gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"

        new_gm = copy.deepcopy(new_gm)

        self.assertEqual(new_gm.meta["custom"]["f"], "bar")
        counter = 0
        for node in new_gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor:
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
        self.assertEqual(counter, 1)

    def test_unbacked_range_serdes(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check_is_size(n, max=y.size(0) - 1)
                return torch.empty(n), y[n]

        ep = torch.export.export(
            Foo(),
            (torch.tensor([5]), torch.randn(10)),
            dynamic_shapes={
                "x": None,
                "y": (Dim.DYNAMIC,),
            },
        )
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)

        # pre-serialize ep
        pre_shape_env = torch._guards.detect_fake_mode(
            [node.meta.get("val") for node in ep.graph.nodes]
        ).shape_env
        post_shape_env = torch._guards.detect_fake_mode(
            [node.meta.get("val") for node in loaded_ep.graph.nodes]
        ).shape_env
        self.assertEqual(pre_shape_env.var_to_range, post_shape_env.var_to_range)

    def test_backed_size_oblivious_serdes(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y + z.item()

        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = torch.export.export(
                Foo(),
                (torch.randn(1), torch.randn(1), torch.tensor([5])),
                dynamic_shapes={
                    "x": (Dim.DYNAMIC,),
                    "y": (Dim.DYNAMIC,),
                    "z": None,
                },
            )
        buffer = io.BytesIO()
        save(ep, buffer)
        buffer.seek(0)
        loaded_ep = load(buffer)
        shape_env = torch._guards.detect_fake_mode(
            [node.meta.get("val") for node in loaded_ep.graph.nodes]
        ).shape_env
        s0 = next(iter(ep.graph.nodes)).meta["val"].size(0)
        self.assertEqual(shape_env.var_to_range[s0.node.expr].lower, 0)


if __name__ == "__main__":
    run_tests()
