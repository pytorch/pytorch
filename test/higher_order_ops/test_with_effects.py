# Owner(s): ["module: functorch"]
# ruff: noqa: F841
import operator
import unittest
from collections import deque
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from functorch.compile import (
    aot_function,
    default_decompositions,
    min_cut_rematerialization_partition,
    nop,
)
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._functorch.aot_autograd import aot_export_module
from torch._guards import tracing, TracingContext
from torch._higher_order_ops._effect_token_utils import EffectTokenAnalyzer
from torch._higher_order_ops.cond import cond_op
from torch._higher_order_ops.effects import (
    _EffectType,
    _get_effect,
    _register_effectful_op,
    with_effects,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import has_side_effect
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM70OrLater, SM80OrLater
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_CUDA,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

from torch.testing._internal.two_tensor import TwoTensor


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


def get_fw_bw_graph(
    f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False
):
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    requires_grad = False

    def fn_req_grad(t):
        nonlocal requires_grad
        requires_grad = requires_grad or t.requires_grad
        return t

    torch.utils._pytree.tree_map_only(torch.Tensor, fn_req_grad, inps)

    out = aot_function(
        f,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
        bw_compiler=(
            partial(extract_graph, graph_cell=bw_graph_cell) if requires_grad else nop
        ),
        partition_fn=partitioner,
        decompositions=default_decompositions,
        dynamic=dynamic,
    )(*inps)

    if requires_grad:
        out.sum().backward()

    return (fw_graph_cell[0], bw_graph_cell[0])


def make_inputs_non_leaves(inps):
    return torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: t.add(1), inps)


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestWithEffects(TestCase):
    def setUp(self):
        super().setUp()
        init_torchbind_implementations()

    def test_print(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._print("moo")
                res = x + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # Without functionalization, print should just appear in the graph directly
        gm = make_fx(M())(*inputs)
        FileCheck().check_count("torch.ops.aten._print.default", 2, exactly=True).run(
            gm.code
        )

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

        with torch._functorch.config.patch(unlift_effect_tokens=True):
            gm, gs = aot_export_module(M(), inputs, trace_joint=False)
            self.assertExpectedInline(
                str(gm.code).strip(),
                """\
def forward(self, arg1_1):
    _make_token_default = torch.ops.prims._make_token.default()
    with_effects = torch.ops.higher_order.with_effects(_make_token_default, torch.ops.aten._print.default, 'moo');  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem_2]);  getitem_2 = _sink_tokens_default = None
    return (add,)""",
            )

    def test_effect_token_analyzer_rejects_negative_getitem(self):
        graph = torch.fx.Graph()
        token = graph.placeholder("token")
        x = graph.placeholder("x")
        effect = graph.call_function(
            with_effects, (token, torch.ops.aten.sin.default, x)
        )
        result = graph.call_function(operator.getitem, (effect, -1))
        graph.output((result,))
        module = torch.fx.GraphModule({}, graph)

        analyzer = EffectTokenAnalyzer(lambda module, node: 0)
        self.assertFalse(analyzer.is_definite_token_output(module, result))

    def test_torchbind_custom_op(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                return (x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x),)

        with enable_torchbind_tracing():
            gm, gs = aot_export_module(M(), (torch.ones(2, 3),), trace_joint=False)

        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _torchbind_obj0 = self._torchbind_obj0
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, _torchbind_obj0, arg1_1);  arg0_1 = _torchbind_obj0 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
    return (getitem, add)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

    def test_print_with_buffer_mutations(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(3))

            def forward(self, x):
                torch.ops.aten._print("moo")
                res = x + x
                self.buf.add_(res)
                res = self.buf + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg2_1, arg2_1)
    add_1 = torch.ops.aten.add.Tensor(arg1_1, add);  arg1_1 = add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, arg2_1);  arg2_1 = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add_1, add_2)""",
        )
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.buffers_to_mutate), 1)

    def test_print_with_input_mutations(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                torch.ops.aten._print("moo")
                res = x + x
                x.add_(res)
                res = x + x
                torch.ops.aten._print("moo")
                return (res,)

        inputs = (torch.randn(3),)

        # With functionalization, it should appear wrapped with with_effects()
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)
        self.assertEqual(len(gs.user_inputs_to_mutate), 1)

    def test_alias_op(self):
        def f(token, x):
            token, out = with_effects(token, torch.ops.aten.absolute_.default, x)
            return token, out

        with self.assertRaisesRegex(
            AssertionError, r"Ops with aliasing is not supported"
        ):
            make_fx(f)(torch.tensor([]), torch.tensor(4))

    def test_compile_aot_eager(self):
        def f(x):
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3),)

        res = torch.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM70OrLater, "triton")
    def test_compile_inductor(self):
        def f(x):
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3),)

        res = torch.compile(f, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @skipIfNoDynamoSupport
    def test_compile_inductor_external_op_return_none(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch.library.define(
                "mylib::inplace_add",
                "(Tensor input, Tensor(a!) output) -> ()",
                lib=lib,
            )

            def inplace_add(input: torch.Tensor, output: torch.Tensor) -> None:
                if input.device != output.device:
                    raise AssertionError(
                        f"Expected input.device == output.device, "
                        f"got {input.device} vs {output.device}"
                    )
                output.add_(input)

            lib.impl("inplace_add", inplace_add, "CompositeExplicitAutograd")

            def f(x):
                out = torch.empty(3)
                out = torch.zeros_like(out)
                torch.ops.mylib.inplace_add(x, out)
                return out

            inputs = (torch.randn(3),)

            res = torch.compile(f, backend="inductor")(*inputs)
            self.assertTrue(torch.allclose(res, f(*inputs)))

    def test_compile_aot_eager_requires_grad(self):
        def f(x):
            torch.ops.aten._print("moo")
            res = x + x
            torch.ops.aten._print("moo")
            return res

        inputs = (torch.randn(2, 3, requires_grad=True),)

        res = torch.compile(f, backend="aot_eager")(*inputs)
        self.assertTrue(torch.allclose(res, f(*inputs)))

        res.sum().backward()

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(not TEST_CUDA, "triton")
    @skipIfNoDynamoSupport
    def test_register_effectful_custom_op(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            # global variable to store the recorded tensor and prefix.
            recorded_dict = {}

            # Pytorch custom op implementation
            @torch.library.custom_op("mylib::record_scalar_tensor", mutates_args=())
            def record_scalar_tensor(x: torch.Tensor, prefix: str) -> None:
                recorded_dict[prefix] = x.clone()
                return

            # Meta function of the custom op
            @record_scalar_tensor.register_fake
            def record_scalar_tensor_meta(x, prefix):
                return

            record_scalar_tensor.register_effect(_EffectType.ORDERED)

            self.assertEqual(_get_effect(record_scalar_tensor), _EffectType.ORDERED)

            my_config = {}
            my_config["MockModule"] = "mean"
            my_config["MockModule.linear"] = "mean"
            my_config["MockModule.relu"] = "mean"

            class MyLinear(torch.nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.randn(out_features, in_features), requires_grad=True
                    )
                    self.bias = torch.nn.Parameter(
                        torch.randn(out_features), requires_grad=True
                    )

                def forward(self, x):
                    return torch.nn.functional.linear(x, self.weight, self.bias)

            class MockModule(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = MyLinear(10, 10)
                    self.register_buffer(
                        "buf0", torch.randn(10, 10, requires_grad=True)
                    )

                def forward(self, x):
                    return torch.nn.functional.relu(self.linear(x) + self.buf0)

            def forward_hook(
                module: torch.nn.Module,
                inputs: torch.Tensor,
                output: torch.Tensor,
                prefix: str,
                aggregate_method: str,
            ) -> torch.Tensor:
                if aggregate_method == "mean":
                    torch.ops.mylib.record_scalar_tensor(output.mean(), prefix)
                elif aggregate_method == "max":
                    torch.ops.mylib.record_scalar_tensor(output.max(), prefix)
                else:
                    # demo purpose, using "min"
                    torch.ops.mylib.record_scalar_tensor(output.sum(), prefix)
                return output

            def add_hooks(module, config):
                handles: list[RemovableHandle] = []
                q = deque([(module.__class__.__name__, module)])
                while q:
                    name, m = q.pop()
                    children = [(name + "." + n, y) for (n, y) in m.named_children()]
                    q.extend(children)
                    aggregate_method = config.get(name, "mean")
                    prefix = name + ":" + aggregate_method
                    handle = m.register_forward_hook(
                        partial(
                            forward_hook,
                            prefix=prefix,
                            aggregate_method=aggregate_method,
                        )
                    )
                    if handle:
                        handles.append(handle)
                return handles

            x = torch.randn(10, 10, device="cuda")
            mod = MockModule().to("cuda")

            add_hooks(mod, my_config)

            opt_mod = torch.compile(backend="inductor")(mod)
            y = opt_mod(x)

            self.assertTrue(torch.allclose(y, mod(x)))
            # Ensure it works well with backward
            y.sum().backward()
            # Ensure the grad is existing
            self.assertTrue(isinstance(opt_mod.linear.weight.grad, torch.Tensor))

            self.assertEqual(len(recorded_dict), 2)
            self.assertTrue("MockModule.linear:mean" in recorded_dict)
            self.assertTrue("MockModule:mean" in recorded_dict)

    @skipIfNoDynamoSupport
    def test_effectful_custom_op_with_subclasses(self):
        with torch.library._scoped_library("_mylib", "FRAGMENT") as lib:
            lib.define("zoo(Tensor x) -> Tensor")
            lib.define("zoo2(Tensor x) -> Tensor")

            d = {"fw": 0, "bw": 0}

            def reset_counter():
                d["fw"] = 0
                d["bw"] = 0

            def assert_counter(fw, bw):
                self.assertEqual(d["fw"], fw)
                self.assertEqual(d["bw"], bw)

            def foo_impl(a):
                d["fw"] = d["fw"] + 1
                return 2 * a.clone()

            def foo_meta(a):
                return a.clone()

            def foo2_impl(x):
                d["bw"] = d["bw"] + 1
                return x.clone()

            def foo2_meta(a):
                return a.clone()

            for backend in ["CPU", "CUDA"]:
                lib.impl("zoo", foo_impl, backend)
                lib.impl("zoo2", foo2_impl, backend)
            lib.impl("zoo", foo_meta, "Meta")
            lib.impl("zoo2", foo2_meta, "Meta")

            def foo_bwd(ctx, grad):
                torch.ops._mylib.zoo2(grad)
                return grad.clone()

            torch.library.register_autograd("_mylib::zoo", foo_bwd, lib=lib)

            torch.library._register_effectful_op(
                torch.ops._mylib.zoo.default, _EffectType.ORDERED
            )
            torch.library._register_effectful_op(
                torch.ops._mylib.zoo2.default, _EffectType.ORDERED
            )

            def fn(x, y):
                return torch.ops._mylib.zoo(x) + y

            def ins_sc():
                return (
                    TwoTensor(
                        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
                    ),
                    torch.tensor([4.0, 5.0, 6.0]),
                )

            def ins_dense():
                return torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])

            for ins_fn, expected_fw_count in zip([ins_sc, ins_dense], [2, 1]):
                reset_counter()
                ref_out = fn(*ins_fn())
                assert_counter(expected_fw_count, 0)

                compiled_fn = torch.compile(fn, backend="aot_eager")
                out = compiled_fn(*ins_fn())
                reset_counter()
                out = compiled_fn(*ins_fn())
                assert_counter(expected_fw_count, 0)

                self.assertEqual(ref_out, out)

            def ins_dense_req_grad():
                return (
                    torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                    torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                )

            def ins_sc_req_grad():
                return (
                    TwoTensor(
                        torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    ),
                    TwoTensor(
                        torch.tensor([7.0, 8.0, 9.0], requires_grad=True),
                        torch.tensor([10.0, 11.0, 12.0], requires_grad=True),
                    ),
                )

            for (
                ins_fn_req_grad,
                (
                    expected_fw_count,
                    expected_fw_count_after_bw,
                    expected_bw_count_after_bw,
                ),
            ) in zip([ins_dense_req_grad, ins_sc_req_grad], [(1, 1, 1), (2, 2, 2)]):
                ref_ins = ins_fn_req_grad()
                reset_counter()
                ref_out = fn(*ref_ins)
                assert_counter(expected_fw_count, 0)
                ref_out.sum().backward()
                assert_counter(expected_fw_count_after_bw, expected_bw_count_after_bw)

                compiled_fn = torch.compile(fn, fullgraph=True)

                ins = ins_fn_req_grad()
                out = compiled_fn(*ins)
                reset_counter()
                out = compiled_fn(*ins)
                assert_counter(expected_fw_count, 0)
                self.assertEqual(ref_out, out)
                out.sum().backward()
                assert_counter(expected_fw_count_after_bw, expected_bw_count_after_bw)
                self.assertEqual(ref_ins[1].grad, ins[1].grad)
                self.assertEqual(ref_ins[0].grad, ins[0].grad)

            fw_graph, bw_graph = get_fw_bw_graph(fn, ins_sc_req_grad())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops._mylib.zoo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._mylib.zoo.default, primals_3);  getitem = primals_3 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_1, primals_4);  getitem_1 = primals_4 = None
    add_1 = torch.ops.aten.add.Tensor(getitem_3, primals_5);  getitem_3 = primals_5 = None
    return (getitem_2, add, add_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, tangents_1, tangents_2, tangents_token):
    with_effects_2 = torch.ops.higher_order.with_effects(tangents_token, torch.ops._mylib.zoo2.default, tangents_1);  tangents_token = None
    getitem_4 = with_effects_2[0];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._mylib.zoo2.default, tangents_2);  getitem_4 = None
    getitem_6 = with_effects_3[0];  with_effects_3 = None
    clone = torch.ops.aten.clone.default(tangents_1)
    clone_1 = torch.ops.aten.clone.default(tangents_2)
    return (clone, clone_1, tangents_1, tangents_2, getitem_6)""",
            )

    def test_dce(self):
        # If an operator is marked as side effectful, it should not get DCEd by
        # FX's eliminate_dead_code

        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            log3 = []

            @torch.library.custom_op(
                "mylib::my_logger3",
                mutates_args=(),
            )
            def my_logger3(s: str, t: torch.Tensor) -> torch.Tensor:
                log3.append(s)
                return torch.zeros(1)

            @my_logger3.register_fake
            def my_logger3(s, t) -> torch.Tensor:
                return torch.zeros(1)

            # Registering an op as being effectful should also prevent FX DCE
            from torch._library.effects import EffectType

            torch.library._register_effectful_op(
                "mylib::my_logger3", EffectType.ORDERED
            )

            def foo(x):
                b = torch.scalar_tensor(x.shape[0])
                torch.ops.mylib.my_logger3("moo", b)
                return x + x

            gm = make_fx(foo, tracing_mode="symbolic")(torch.ones(3, 3))
            gm.graph.eliminate_dead_code()
            gm.recompile()
            gm(torch.ones(3, 3))
            self.assertTrue(len(log3), 1)

    def test_effects_and_input_mutation_return(self):
        def fn(a, b):
            torch.ops.aten._print("effect")
            return torch.sin(a, out=b)

        inp = [torch.randn(3, 3), torch.ones(3, 3)]
        ref_out = fn(*inp)
        out = torch.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'effect');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    return (getitem, sin, sin)""",
        )

    def test_effects_and_input_output_view_simple(self):
        def fn(a):
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        ref_out = fn(*inp)
        out = torch.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        ref_out = fn(*inp)
        out = torch.compile(fn, fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    view = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
    return (view,)""",
        )

    def test_effects_and_aliased_outputs(self):
        def fn(a):
            b = a.mul(2)
            torch.ops.aten._print("effect")
            c = b.view(-1)
            return b, c

        f_compiled = aot_function(fn, nop)
        for req_grad in [True, False]:
            inp = torch.ones(3, requires_grad=req_grad)
            out_ref = fn(inp)
            out_test = f_compiled(inp)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])
            # Try mutating one of the outputs, which is aliased.
            out_ref[0].mul_(3)
            out_test[0].mul_(3)
            # Assert that the aliasing relationship was preserved
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])

    def test_effects_and_input_mutation_is_output(self):
        def fn(a):
            a.mul_(2)
            torch.ops.aten._print("effect")
            return a

        inp = make_inputs_non_leaves([torch.ones(3, 3, requires_grad=True)])
        ref_out = fn(*inp)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        inp = [torch.ones(3, 3, requires_grad=False)]
        ref_out = fn(*inp)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(*inp)
        self.assertEqual(ref_out, out)

        fw_graph, bw_graph = get_fw_bw_graph(fn, inp)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mul = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.aten._print.default, 'effect');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    return (getitem, mul, mul)""",
        )

    @skipIfTorchDynamo()
    def test_effectful_op_in_backward(self):
        with torch.library._scoped_library("_mylib", "FRAGMENT") as lib:
            lib.define("foo(Tensor x) -> Tensor")

            def foo_impl(a):
                return a.clone()

            def foo_bwd(ctx, grad):
                return torch.ops._mylib.foo(grad)

            for backend in ["CPU", "CUDA", "Meta"]:
                lib.impl("foo", foo_impl, backend)

            torch.library.register_autograd("_mylib::foo", foo_bwd, lib=lib)

            handle = _register_effectful_op(
                torch.ops._mylib.foo.default, _EffectType.ORDERED
            )
            self.assertEqual(
                _get_effect(torch.ops._mylib.foo.default), _EffectType.ORDERED
            )

            try:

                def fn(x, y):
                    return torch.ops._mylib.foo(x) + y

                def ins_dense_req_grad():
                    return (
                        torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    )

                def ins_sc_req_grad():
                    return (
                        TwoTensor(
                            torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                            torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                        ),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    )

                for i, ins_fn in enumerate([ins_dense_req_grad, ins_sc_req_grad]):
                    ref_ins = ins_fn()

                    ref_out = fn(*ref_ins)
                    ref_out.sum().backward()

                    compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
                    ins = ins_fn()
                    out = compiled_fn(*ins)
                    self.assertEqual(ref_out, out)
                    out.sum().backward()
                    self.assertEqual(ref_ins[1].grad, ins[1].grad)
                    self.assertEqual(ref_ins[0].grad, ins[0].grad)

                    fw_graph, bw_graph = get_fw_bw_graph(fn, ins)
                    if i == 0:
                        self.assertExpectedInline(
                            fw_graph.code.strip(),
                            """\
def forward(self, primals_1, primals_2, primals_3):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops._mylib.foo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    add = torch.ops.aten.add.Tensor(getitem_1, primals_3);  getitem_1 = primals_3 = None
    return (getitem, add)""",
                        )
                        self.assertExpectedInline(
                            bw_graph.code.strip(),
                            """\
def forward(self, tangents_1, tangents_token):
    with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops._mylib.foo.default, tangents_1);  tangents_token = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    return (getitem_3, tangents_1, getitem_2)""",
                        )
                    elif i == 1:
                        self.assertExpectedInline(
                            fw_graph.code.strip(),
                            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops._mylib.foo.default, primals_2);  primals_1 = primals_2 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops._mylib.foo.default, primals_3);  getitem = primals_3 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    add = torch.ops.aten.add.Tensor(getitem_1, primals_4);  getitem_1 = None
    add_1 = torch.ops.aten.add.Tensor(getitem_3, primals_4);  getitem_3 = primals_4 = None
    return (getitem_2, add, add_1)""",
                        )
                        self.assertExpectedInline(
                            bw_graph.code.strip(),
                            """\
def forward(self, tangents_1, tangents_2, tangents_token):
    with_effects_2 = torch.ops.higher_order.with_effects(tangents_token, torch.ops._mylib.foo.default, tangents_1);  tangents_token = None
    getitem_4 = with_effects_2[0]
    getitem_5 = with_effects_2[1];  with_effects_2 = None
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_4, torch.ops._mylib.foo.default, tangents_2);  getitem_4 = None
    getitem_6 = with_effects_3[0]
    getitem_7 = with_effects_3[1];  with_effects_3 = None
    return (getitem_5, getitem_7, tangents_1, tangents_2, getitem_6)""",
                        )
                    else:
                        raise NotImplementedError
            finally:
                handle.destroy()

            self.assertEqual(_get_effect(torch.ops._mylib.foo.default), None)

    @skipIfNoDynamoSupport
    def test_regular_effectful_op_only_in_backward(self):
        handle = _register_effectful_op(torch.ops.aten.cos.default, _EffectType.ORDERED)
        try:

            def fn(x):
                return x.sin()

            def inps_fn():
                return (torch.tensor([1.0, 2.0, 3.0], requires_grad=True),)

            torch.compile(fn, backend="inductor", fullgraph=True)(*inps_fn())

            fw_graph, bw_graph = get_fw_bw_graph(fn, inps_fn())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1):
    sin = torch.ops.aten.sin.default(primals_1)
    return (sin, primals_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_1, tangents_1, tangents_token):
    with_effects = torch.ops.higher_order.with_effects(tangents_token, torch.ops.aten.cos.default, primals_1);  tangents_token = primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, getitem_1);  tangents_1 = getitem_1 = None
    return (mul, getitem)""",
            )

            def inps_fn_sc():
                return (
                    TwoTensor(
                        torch.tensor([1.0, 2.0, 3.0], requires_grad=True),
                        torch.tensor([4.0, 5.0, 6.0], requires_grad=True),
                    ),
                )

            torch.compile(fn, backend="inductor", fullgraph=True)(*inps_fn_sc())
            fw_graph, bw_graph = get_fw_bw_graph(fn, inps_fn_sc())
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2):
    sin = torch.ops.aten.sin.default(primals_1)
    sin_1 = torch.ops.aten.sin.default(primals_2)
    return (sin, sin_1, primals_1, primals_2)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2, tangents_1, tangents_2, tangents_token):
    with_effects = torch.ops.higher_order.with_effects(tangents_token, torch.ops.aten.cos.default, primals_1);  tangents_token = primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    with_effects_1 = torch.ops.higher_order.with_effects(getitem, torch.ops.aten.cos.default, primals_2);  getitem = primals_2 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, getitem_1);  tangents_1 = getitem_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_2, getitem_3);  tangents_2 = getitem_3 = None
    return (mul, mul_1, getitem_2)""",
            )
        finally:
            handle.destroy()

    @skipIfNoDynamoSupport
    def test_regular_effectful_op_in_forward_and_backward(self):
        handle = _register_effectful_op(torch.ops.aten.cos.default, _EffectType.ORDERED)
        try:

            def fn(x):
                x = x.cos()
                return x.sin()

            inps = (torch.tensor([1.0, 2.0, 3.0], requires_grad=True),)
            torch.compile(fn, backend="inductor", fullgraph=True)(*inps)

            fw_graph, bw_graph = get_fw_bw_graph(fn, inps)
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, primals_1, primals_2):
    with_effects = torch.ops.higher_order.with_effects(primals_1, torch.ops.aten.cos.default, primals_2);  primals_1 = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    sin = torch.ops.aten.sin.default(getitem_1)
    return (getitem, sin, primals_2, getitem_1)""",
            )
            self.assertExpectedInline(
                bw_graph.code.strip(),
                """\
def forward(self, primals_2, getitem_1, tangents_1, tangents_token):
    with_effects_1 = torch.ops.higher_order.with_effects(tangents_token, torch.ops.aten.cos.default, getitem_1);  tangents_token = getitem_1 = None
    getitem_2 = with_effects_1[0]
    getitem_3 = with_effects_1[1];  with_effects_1 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, getitem_3);  tangents_1 = getitem_3 = None
    sin_1 = torch.ops.aten.sin.default(primals_2);  primals_2 = None
    neg = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, neg);  mul = neg = None
    return (mul_1, getitem_2)""",
            )
        finally:
            handle.destroy()

    def test_compile_cpu_cond_with_effect_in_branch(self):
        with torch.library._scoped_library("mylib_cpu_cond_effect", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op("mylib_cpu_cond_effect::record", mutates_args=())
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_cpu_cond_effect.record.default)

            def fn(x, pred):
                def true_fn(x):
                    torch.ops.mylib_cpu_cond_effect.record(x, "true")
                    return x + 1

                def false_fn(x):
                    return x - 1

                return torch.cond(pred, true_fn, false_fn, (x,))

            for backend in ("aot_eager", "inductor"):
                compiled = torch.compile(fn, backend=backend, fullgraph=True)

                x = torch.ones(2, requires_grad=True)
                recorded.clear()
                out = compiled(x, torch.tensor(True))
                self.assertEqual(out, x + 1)
                out.sum().backward()
                self.assertEqual(x.grad, torch.ones_like(x))
                self.assertEqual(recorded, ["true"])

                x = torch.ones(2, requires_grad=True)
                recorded.clear()
                out = compiled(x, torch.tensor(False))
                self.assertEqual(out, x - 1)
                out.sum().backward()
                self.assertEqual(x.grad, torch.ones_like(x))
                self.assertEqual(recorded, [])

    def test_compile_effect_only_cond_preserves_order(self):
        with torch.library._scoped_library("mylib_effect_only_cond", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op("mylib_effect_only_cond::record", mutates_args=())
            def record(x: torch.Tensor, label: str) -> None:
                recorded.append(label)

            @record.register_fake
            def record_fake(x, label):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_effect_only_cond.record.default)

            def fn(x, pred):
                def true_fn(x):
                    torch.ops.mylib_effect_only_cond.record(x, "inside")
                    return ()

                def false_fn(x):
                    return ()

                torch.cond(pred, true_fn, false_fn, (x,))
                torch.ops.mylib_effect_only_cond.record(x, "after")
                return x + 1

            for backend in ("aot_eager", "inductor"):
                compiled = torch.compile(fn, backend=backend, fullgraph=True)
                for pred, expected in ((True, ["inside", "after"]), (False, ["after"])):
                    x = torch.ones(2)
                    recorded.clear()
                    self.assertEqual(
                        compiled(x, torch.tensor(pred)), torch.full_like(x, 2)
                    )
                    self.assertEqual(recorded, expected)

    def test_compile_cond_with_effect_and_input_mutation(self):
        with torch.library._scoped_library("mylib_cond_effect_mutation", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op(
                "mylib_cond_effect_mutation::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> None:
                recorded.append(x.clone())

            @record.register_fake
            def record_fake(x):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_cond_effect_mutation.record.default)

            def fn(x, pred):
                def true_fn(x):
                    torch.ops.mylib_cond_effect_mutation.record(x)
                    x.add_(1)
                    return x

                def false_fn(x):
                    x.sub_(1)
                    return x

                with torch.no_grad():
                    return torch.cond(pred, true_fn, false_fn, (x,))

            for backend in ("aot_eager", "inductor"):
                compiled = torch.compile(fn, backend=backend, fullgraph=True)
                for pred, expected in ((True, 2), (False, 0)):
                    x = torch.ones(2)
                    recorded.clear()
                    self.assertEqual(
                        compiled(x, torch.tensor(pred)), torch.full_like(x, expected)
                    )
                    self.assertEqual(x, torch.full_like(x, expected))
                    self.assertEqual(len(recorded), int(pred))

            def effect_before_cond(x, y, pred):
                def true_fn(x, y):
                    x.add_(1)
                    return y

                def false_fn(x, y):
                    x.sub_(1)
                    return y

                torch.ops.mylib_cond_effect_mutation.record(y)
                with torch.no_grad():
                    return torch.cond(pred, true_fn, false_fn, (x, y))

            x = torch.ones(2)
            y = torch.arange(2.0)
            recorded.clear()
            self.assertEqual(
                torch.compile(effect_before_cond, backend="inductor", fullgraph=True)(
                    x, y, torch.tensor(True)
                ),
                y,
            )
            self.assertEqual(recorded, [y])

            def ignored_cond_result(x, y, pred):
                def true_fn(x):
                    torch.ops.mylib_cond_effect_mutation.record(x)
                    x.add_(1)
                    return x

                def false_fn(x):
                    x.sub_(1)
                    return x

                with torch.no_grad():
                    torch.cond(pred, true_fn, false_fn, (x,))
                torch.ops.mylib_cond_effect_mutation.record(y)
                return y + 1

            x = torch.ones(2)
            y = torch.arange(2.0)
            recorded.clear()
            self.assertEqual(
                torch.compile(ignored_cond_result, backend="inductor", fullgraph=True)(
                    x, y, torch.tensor(True)
                ),
                y + 1,
            )
            self.assertEqual(recorded, [torch.ones(2), y])
            self.assertEqual(x, torch.full_like(x, 2))

            def empty_cond_result(x, pred):
                def true_fn(x):
                    torch.ops.mylib_cond_effect_mutation.record(x)
                    x.add_(1)
                    return ()

                def false_fn(x):
                    x.sub_(1)
                    return ()

                with torch.no_grad():
                    torch.cond(pred, true_fn, false_fn, (x,))
                return x * 2

            compiled = torch.compile(
                empty_cond_result, backend="inductor", fullgraph=True
            )
            for pred, expected, num_records in (
                (True, 4, 1),
                (False, 0, 0),
            ):
                x = torch.ones(2)
                recorded.clear()
                self.assertEqual(
                    compiled(x, torch.tensor(pred)), torch.full_like(x, expected)
                )
                self.assertEqual(x, torch.full_like(x, expected / 2))
                self.assertEqual(len(recorded), num_records)

    def test_compile_nested_cond_with_effect_and_input_mutation(self):
        with torch.library._scoped_library(
            "mylib_nested_cond_effect_mutation", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_nested_cond_effect_mutation::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> None:
                recorded.append(x.clone())

            @record.register_fake
            def record_fake(x):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_nested_cond_effect_mutation.record.default)

            def outer_true(x, inner_pred):
                def inner_true(x):
                    torch.ops.mylib_nested_cond_effect_mutation.record(x)
                    x.add_(1)
                    return (x,)

                def inner_false(x):
                    x.sub_(1)
                    return (x,)

                return cond_op(inner_pred, inner_true, inner_false, (x,))

            def outer_false(x, inner_pred):
                x.mul_(2)
                return (x,)

            true_graph = make_fx(outer_true)(torch.ones(2), torch.tensor(True))
            false_graph = make_fx(outer_false)(torch.ones(2), torch.tensor(True))

            def fn(x, outer_pred, inner_pred):
                with torch.no_grad():
                    return cond_op(
                        outer_pred,
                        true_graph,
                        false_graph,
                        (x, inner_pred),
                    )[0]

            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            for outer_pred, inner_pred, expected, num_records in (
                (True, True, 2, 1),
                (True, False, 0, 0),
                (False, True, 2, 0),
            ):
                x = torch.ones(2)
                recorded.clear()
                self.assertEqual(
                    compiled(
                        x,
                        torch.tensor(outer_pred),
                        torch.tensor(inner_pred),
                    ),
                    torch.full_like(x, expected),
                )
                self.assertEqual(x, torch.full_like(x, expected))
                self.assertEqual(len(recorded), num_records)

    def test_compile_nested_cond_does_not_replay_forward_effect_in_backward(self):
        with torch.library._scoped_library(
            "mylib_nested_cond_backward_replay", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_nested_cond_backward_replay::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> None:
                recorded.append("forward")

            @record.register_fake
            def record_fake(x):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_nested_cond_backward_replay.record.default)

            def fn(x, outer_pred, inner_pred):
                def outer_true(x, inner_pred):
                    def inner_true(x):
                        torch.ops.mylib_nested_cond_backward_replay.record(x)
                        return x.sin()

                    def inner_false(x):
                        return x.sin()

                    return torch.cond(inner_pred, inner_true, inner_false, (x,))

                def outer_false(x, inner_pred):
                    return x.sin()

                return torch.cond(outer_pred, outer_true, outer_false, (x, inner_pred))

            x = torch.ones(2, requires_grad=True)
            compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
            recorded.clear()
            compiled(x, torch.tensor(True), torch.tensor(True)).sum().backward()
            self.assertEqual(recorded, ["forward"])

    def test_cond_rejects_live_forward_effect_result_in_backward(self):
        with torch.library._scoped_library("mylib_cond_live_effect_result", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op(
                "mylib_cond_live_effect_result::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> torch.Tensor:
                recorded.append("forward")
                return x.clone()

            @record.register_fake
            def record_fake(x):
                return x.clone()

            def backward(ctx, grad):
                return grad

            record.register_autograd(backward)
            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_cond_live_effect_result.record.default)

            def false_fn(x):
                return (x.sin(),)

            def live_true_fn(x):
                return (torch.ops.mylib_cond_live_effect_result.record(x).sin(),)

            x = torch.ones(2, requires_grad=True)
            recorded.clear()
            out = cond_op(torch.tensor(True), live_true_fn, false_fn, (x,))[0]
            self.assertEqual(recorded, ["forward"])
            with self.assertRaisesRegex(
                RuntimeError,
                "effectful branch operation whose result is required for backward",
            ):
                out.sum().backward()
            self.assertEqual(recorded, ["forward"])

            def outer_true_fn(x, inner_pred):
                return cond_op(inner_pred, live_true_fn, false_fn, (x,))

            def outer_false_fn(x, inner_pred):
                return false_fn(x)

            x = torch.ones(2, requires_grad=True)
            recorded.clear()
            out = cond_op(
                torch.tensor(True),
                outer_true_fn,
                outer_false_fn,
                (x, torch.tensor(True)),
            )[0]
            self.assertEqual(recorded, ["forward"])
            with self.assertRaisesRegex(
                RuntimeError,
                "effectful branch operation whose result is required for backward",
            ):
                out.sum().backward()
            self.assertEqual(recorded, ["forward"])

            def dead_true_fn(x):
                return (torch.ops.mylib_cond_live_effect_result.record(x) * 2,)

            x = torch.ones(2, requires_grad=True)
            recorded.clear()
            cond_op(torch.tensor(True), dead_true_fn, false_fn, (x,))[
                0
            ].sum().backward()
            self.assertEqual(x.grad, torch.full_like(x, 2))
            self.assertEqual(recorded, ["forward"])

            recorded.clear()
            with torch.no_grad():
                out = cond_op(
                    torch.tensor(True), live_true_fn, false_fn, (torch.ones(2),)
                )[0]
            self.assertEqual(out, torch.ones(2).sin())
            self.assertEqual(recorded, ["forward"])

            def compiled_fn(x, pred):
                def true_fn(x):
                    return torch.ops.mylib_cond_live_effect_result.record(x).sin()

                def false_fn(x):
                    return x.sin()

                return torch.cond(pred, true_fn, false_fn, (x,))

            recorded.clear()
            with self.assertRaisesRegex(
                torch._dynamo.exc.BackendCompilerFailed,
                "effectful branch operation whose result is required for backward",
            ):
                torch.compile(compiled_fn, backend="aot_eager", fullgraph=True)(
                    torch.ones(2, requires_grad=True),
                    torch.tensor(True),
                )
            self.assertEqual(recorded, [])

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_compile_cond_with_effectful_invoke_subgraph(self):
        with torch.library._scoped_library(
            "mylib_cond_effectful_invoke_subgraph", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_cond_effectful_invoke_subgraph::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> None:
                recorded.append(x.shape)

            @record.register_fake
            def record_fake(x):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(
                torch.ops.mylib_cond_effectful_invoke_subgraph.record.default
            )

            @torch.compiler.nested_compile_region
            def region(x):
                torch.ops.mylib_cond_effectful_invoke_subgraph.record(x)
                return x + 1

            def fn(x, pred):
                def true_fn(x):
                    return region(x)

                def false_fn(x):
                    return x - 1

                return torch.cond(pred, true_fn, false_fn, (x,))

            compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
            for pred, expected_count in ((True, 1), (False, 0)):
                x = torch.ones(2, requires_grad=True)
                recorded.clear()
                out = compiled(x, torch.tensor(pred))
                self.assertEqual(out, x + 1 if pred else x - 1)
                out.sum().backward()
                self.assertEqual(x.grad, torch.ones_like(x))
                self.assertEqual(len(recorded), expected_count)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_export_nested_effectful_invoke_subgraphs(self):
        with torch.library._scoped_library(
            "mylib_nested_effectful_invoke_subgraphs", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_nested_effectful_invoke_subgraphs::record", mutates_args=()
            )
            def record(x: torch.Tensor, label: str) -> None:
                recorded.append(label)

            @record.register_fake
            def record_fake(x, label):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(
                torch.ops.mylib_nested_effectful_invoke_subgraphs.record.default
            )

            @torch.compiler.nested_compile_region
            def inner(x):
                torch.ops.mylib_nested_effectful_invoke_subgraphs.record(x, "inner")
                return x + 1

            @torch.compiler.nested_compile_region
            def outer(x):
                return inner(x) * 2

            class M(torch.nn.Module):
                def forward(self, x):
                    torch.ops.mylib_nested_effectful_invoke_subgraphs.record(x, "root")
                    return outer(x)

            x = torch.ones(2)
            ep = torch.export.export(M(), (x,)).run_decompositions()
            for _ in range(2):
                recorded.clear()
                self.assertEqual(ep.module()(x), (x + 1) * 2)
                self.assertEqual(recorded, ["root", "inner"])
                ep.validate()

            @torch.compiler.nested_compile_region
            def effect_only(x):
                torch.ops.mylib_nested_effectful_invoke_subgraphs.record(
                    x, "effect_only"
                )
                return ()

            class EffectOnlyM(torch.nn.Module):
                def forward(self, x):
                    effect_only(x)
                    return x + 1

            ep = torch.export.export(EffectOnlyM(), (x,)).run_decompositions()
            for materialize in (ep.module, lambda: torch.export.unflatten(ep)):
                recorded.clear()
                self.assertEqual(materialize()(x), x + 1)
                self.assertEqual(recorded, ["effect_only"])
                ep.validate()

            @torch.compiler.nested_compile_region
            def dynamic_output(x):
                torch.ops.mylib_nested_effectful_invoke_subgraphs.record(x, "dynamic")
                return torch.nonzero(x)

            class DynamicOutputM(torch.nn.Module):
                def forward(self, x):
                    return dynamic_output(x)

            ep = torch.export.export(DynamicOutputM(), (x,)).run_decompositions()
            mod = ep.module()
            from torch._guards import detect_fake_mode
            from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts

            placeholder_values = [
                node.meta["val"]
                for node in mod.graph.nodes
                if node.op == "placeholder" and "val" in node.meta
            ]
            fake_mode = detect_fake_mode(placeholder_values)
            self.assertIsNotNone(fake_mode)
            with fake_mode:
                PropagateUnbackedSymInts(mod).run(*placeholder_values)
            recorded.clear()
            self.assertEqual(mod(x), torch.nonzero(x))
            self.assertEqual(recorded, ["dynamic"])
            ep.validate()

    def test_export_effect_result_negative_getitem_preserves_source(self):
        with torch.library._scoped_library(
            "mylib_effect_result_negative_getitem", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_effect_result_negative_getitem::record_and_add",
                mutates_args=(),
            )
            def record_and_add(x: torch.Tensor) -> torch.Tensor:
                recorded.append("effect")
                return x + 1

            @record_and_add.register_fake
            def record_and_add_fake(x):
                return torch.empty_like(x)

            record_and_add.register_effect(_EffectType.ORDERED)
            has_side_effect(
                torch.ops.mylib_effect_result_negative_getitem.record_and_add.default
            )

            class M(torch.nn.Module):
                def forward(self, x):
                    return (
                        torch.ops.mylib_effect_result_negative_getitem.record_and_add(x)
                    )

            x = torch.ones(2)
            ep = torch.export.export(M(), (x,)).run_decompositions()
            effect_node = next(
                node
                for node in ep.graph.nodes
                if node.target is torch.ops.higher_order.with_effects
            )
            result_node = next(
                user
                for user in effect_node.users
                if user.target is operator.getitem and user.args[1] == 1
            )
            result_node.args = (effect_node, -1)
            ep.graph_module.recompile()
            ep.validate()

            for materialize in (ep.module, lambda: torch.export.unflatten(ep)):
                for _ in range(2):
                    recorded.clear()
                    self.assertEqual(materialize()(x), x + 1)
                    self.assertEqual(recorded, ["effect"])
                    ep.validate()

            self.assertIn("torch.ops.higher_order.with_effects", ep.graph_module.code)

            @torch.library.custom_op(
                "mylib_effect_result_negative_getitem::record_none",
                mutates_args=(),
            )
            def record_none(x: torch.Tensor) -> None:
                recorded.append("none")

            @record_none.register_fake
            def record_none_fake(x):
                return

            record_none.register_effect(_EffectType.ORDERED)
            has_side_effect(
                torch.ops.mylib_effect_result_negative_getitem.record_none.default
            )

            class NoReturnM(torch.nn.Module):
                def forward(self, x):
                    torch.ops.mylib_effect_result_negative_getitem.record_none(x)
                    return x + 1

            ep = torch.export.export(NoReturnM(), (x,)).run_decompositions()
            effect_node = next(
                node
                for node in ep.graph.nodes
                if node.target is torch.ops.higher_order.with_effects
            )
            token_node = next(
                user
                for user in effect_node.users
                if user.target is operator.getitem and user.args[1] == 0
            )
            token_node.args = (effect_node, -2)
            ep.graph_module.recompile()
            ep.validate()
            recorded.clear()
            self.assertEqual(ep.module()(x), x + 1)
            self.assertEqual(recorded, ["none"])

            @torch.library.custom_op(
                "mylib_effect_result_negative_getitem::dynamic_pair",
                mutates_args=(),
            )
            def dynamic_pair(
                x: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                recorded.append("pair")
                return torch.nonzero(x), torch.nonzero(x + 1)

            @dynamic_pair.register_fake
            def dynamic_pair_fake(x):
                ctx = torch.library.get_ctx()
                first = ctx.new_dynamic_size()
                second = ctx.new_dynamic_size()
                return (
                    torch.empty((first, x.dim()), dtype=torch.int64, device=x.device),
                    torch.empty((second, x.dim()), dtype=torch.int64, device=x.device),
                )

            dynamic_pair.register_effect(_EffectType.ORDERED)
            has_side_effect(
                torch.ops.mylib_effect_result_negative_getitem.dynamic_pair.default
            )

            class DynamicPairM(torch.nn.Module):
                def forward(self, x):
                    return torch.ops.mylib_effect_result_negative_getitem.dynamic_pair(
                        x
                    )

            ep = torch.export.export(DynamicPairM(), (x,)).run_decompositions()
            mod = ep.module()
            from torch._guards import detect_fake_mode
            from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts

            placeholder_values = [
                node.meta["val"]
                for node in mod.graph.nodes
                if node.op == "placeholder" and "val" in node.meta
            ]
            fake_mode = detect_fake_mode(placeholder_values)
            self.assertIsNotNone(fake_mode)
            with fake_mode:
                PropagateUnbackedSymInts(mod).run(*placeholder_values)
            recorded.clear()
            self.assertEqual(
                mod(x),
                (torch.nonzero(x), torch.nonzero(x + 1)),
            )
            self.assertEqual(recorded, ["pair"])
            ep.validate()

    @skipIfTorchDynamo("Tests the error from a nested torch.compile call")
    def test_compile_cond_rejects_none_operand(self):
        def fn(x, pred):
            def true_fn(x, unused):
                return x + 1

            def false_fn(x, unused):
                return x - 1

            return torch.cond(pred, true_fn, false_fn, (x, None))

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            r"None is not a valid torch\.cond operand",
        ):
            compiled(torch.ones(2), torch.tensor(True))

    def test_cond_effect_mismatched_output_structure(self):
        with torch.library._scoped_library(
            "mylib_cond_effect_mismatched_output", "FRAGMENT"
        ):

            @torch.library.custom_op(
                "mylib_cond_effect_mismatched_output::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> None:
                pass

            @record.register_fake
            def record_fake(x):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(
                torch.ops.mylib_cond_effect_mismatched_output.record.default
            )

            def true_fn(x):
                torch.ops.mylib_cond_effect_mismatched_output.record(x)
                return (x + 1, x + 2)

            def false_fn(x):
                return [x - 1, x - 2]

            x = torch.ones(2)
            true_graph = make_fx(true_fn)(x)
            false_graph = make_fx(false_fn)(x)

            def fn(x, pred):
                return cond_op(pred, true_graph, false_graph, (x,))

            with self.assertRaisesRegex(
                RuntimeError, "Unmatched output spec from torch.cond branches"
            ):
                aot_function(fn, nop)(x, torch.tensor(True))

    def test_functionalize_cond_with_effect_in_branch(self):
        with torch.library._scoped_library(
            "mylib_functionalize_cond_effect", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_functionalize_cond_effect::record", mutates_args=()
            )
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_functionalize_cond_effect.record.default)

            def fn(x, pred):
                def true_fn(x):
                    torch.ops.mylib_functionalize_cond_effect.record(x, "true")
                    return x + 1

                def false_fn(x):
                    return x - 1

                return torch.cond(pred, true_fn, false_fn, (x,))

            functionalized = torch.func.functionalize(fn)
            x = torch.ones(2)

            recorded.clear()
            self.assertEqual(functionalized(x, torch.tensor(True)), x + 1)
            self.assertEqual(recorded, ["true"])

            recorded.clear()
            self.assertEqual(functionalized(x, torch.tensor(False)), x - 1)
            self.assertEqual(recorded, [])

    def test_aot_function_cond_with_python_callable_effect(self):
        with torch.library._scoped_library(
            "mylib_aot_cond_callable_effect", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_aot_cond_callable_effect::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> None:
                recorded.append("effect")

            @record.register_fake
            def record_fake(x):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_aot_cond_callable_effect.record.default)

            def true_fn(x):
                torch.ops.mylib_aot_cond_callable_effect.record(x)
                return ()

            def false_fn(x):
                return ()

            def fn(x, pred):
                cond_op(pred, true_fn, false_fn, (x,))
                return x + 1

            compiled = aot_function(fn, nop)
            x = torch.ones(2)
            for pred, expected_effects in ((True, ["effect"]), (False, [])):
                recorded.clear()
                self.assertEqual(compiled(x, torch.tensor(pred)), x + 1)
                self.assertEqual(recorded, expected_effects)

            def mutation_true_fn(x):
                torch.ops.mylib_aot_cond_callable_effect.record(x)
                x.add_(1)
                return (x,)

            def mutation_false_fn(x):
                x.sub_(1)
                return (x,)

            def mutation_fn(x, pred):
                return cond_op(pred, mutation_true_fn, mutation_false_fn, (x,))[0]

            compiled_mutation = aot_function(mutation_fn, nop)
            for pred, expected, expected_effects in (
                (True, 2, ["effect"]),
                (False, 0, []),
            ):
                x = torch.ones(2)
                recorded.clear()
                self.assertEqual(
                    compiled_mutation(x, torch.tensor(pred)),
                    torch.full_like(x, expected),
                )
                self.assertEqual(x, torch.full_like(x, expected))
                self.assertEqual(recorded, expected_effects)

            def child_fn(x):
                torch.ops.mylib_aot_cond_callable_effect.record(x)
                return x + 1

            false_graph = make_fx(lambda x: (x - 1,))(torch.ones(2))

            def check_call_module_child(child):
                parent_graph = torch.fx.Graph()
                parent_x = parent_graph.placeholder("x")
                parent_out = parent_graph.call_module("child", (parent_x,))
                parent_graph.output((parent_out,))
                true_graph = torch.fx.GraphModule({"child": child}, parent_graph)

                def call_module_fn(x, pred):
                    return cond_op(pred, true_graph, false_graph, (x,))[0]

                compiled_call_module = aot_function(call_module_fn, nop)
                for pred, expected, expected_effects in (
                    (True, 2, ["effect"]),
                    (False, 0, []),
                ):
                    x = torch.ones(2)
                    recorded.clear()
                    self.assertEqual(
                        compiled_call_module(x, torch.tensor(pred)),
                        torch.full_like(x, expected),
                    )
                    self.assertEqual(recorded, expected_effects)

            check_call_module_child(make_fx(child_fn)(torch.ones(2)))

            class ChildModule(torch.nn.Module):
                def forward(self, x):
                    return child_fn(x)

            check_call_module_child(ChildModule())

    def test_compile_cond_with_aliased_effectful_branches(self):
        with torch.library._scoped_library("mylib_aliased_cond_effect", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op(
                "mylib_aliased_cond_effect::record", mutates_args=()
            )
            def record(x: torch.Tensor) -> None:
                recorded.append("effect")

            @record.register_fake
            def record_fake(x):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_aliased_cond_effect.record.default)

            def branch(x):
                torch.ops.mylib_aliased_cond_effect.record(x)
                return (x + 1,)

            branch_graph = make_fx(branch)(torch.ones(2))

            def fn(x, pred):
                return cond_op(pred, branch_graph, branch_graph, (x,))[0]

            for backend in ("aot_eager", "inductor"):
                recorded.clear()
                self.assertEqual(
                    torch.compile(fn, backend=backend, fullgraph=True)(
                        torch.ones(2), torch.tensor(True)
                    ),
                    torch.full((2,), 2.0),
                )
                self.assertEqual(recorded, ["effect"])

    def test_cond_preserves_backward_only_effect(self):
        with torch.library._scoped_library("mylib_cond_backward_effect", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op(
                "mylib_cond_backward_effect::record", mutates_args=()
            )
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_cond_backward_effect.record.default)

            class IdentityWithBackwardEffect(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x.clone()

                @staticmethod
                def backward(ctx, grad):
                    torch.ops.mylib_cond_backward_effect.record(grad, "backward")
                    return grad

            def true_fn(x):
                torch.ops.mylib_cond_backward_effect.record(x, "forward")
                return (IdentityWithBackwardEffect.apply(x),)

            def false_fn(x):
                return (IdentityWithBackwardEffect.apply(x),)

            for pred, expected in (
                (torch.tensor(True), ["forward", "backward"]),
                (torch.tensor(False), ["backward"]),
            ):
                x = torch.ones(2, requires_grad=True)
                recorded.clear()
                cond_op(pred, true_fn, false_fn, (x,))[0].sum().backward()
                self.assertEqual(recorded, expected)

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not TEST_CUDA, "triton")
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_export_invoke_subgraph(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            recorded_list = []

            @torch.library.custom_op("mylib::record_memory", mutates_args=())
            def record_memory(prefix: str, module_name: str) -> None:
                torch.cuda.synchronize()
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                memory_str = f"[{prefix}] {module_name}: allocated={mem_alloc:.2f} MB, reserved={mem_reserved:.2f} MB"
                recorded_list.append(memory_str)

            @record_memory.register_fake
            def record_memory_fake(prefix, module_name):
                return

            record_memory.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib.record_memory.default)

            class N(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear1 = torch.nn.Linear(1024, 1024)
                    self.relu = torch.nn.ReLU()
                    self.linear2 = torch.nn.Linear(1024, 1024)

                @torch.compiler.nested_compile_region
                def forward(self, x):
                    torch.ops.mylib.record_memory("forward", "N")
                    x = self.linear1(x)
                    x = self.relu(x)
                    x = self.linear2(x)
                    return x

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mod_list = torch.nn.ModuleList(N() for _ in range(3))

                def forward(self, x):
                    for m in self.mod_list:
                        x = m(x)
                    torch.ops.mylib.record_memory("forward", "N")
                    return (x,)

            model = M().to("cuda")
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(32, 1024, requires_grad=True, device="cuda")

            # Test torch.export
            ep = torch.export.export(model, (x,))
            decomp = ep.run_decompositions()
            self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

            self.assertExpectedInline(
                decomp.graph_module.code.strip(),
                """\
def forward(self, token, p_mod_list_0_linear1_weight, p_mod_list_0_linear1_bias, p_mod_list_0_linear2_weight, p_mod_list_0_linear2_bias, p_mod_list_1_linear1_weight, p_mod_list_1_linear1_bias, p_mod_list_1_linear2_weight, p_mod_list_1_linear2_bias, p_mod_list_2_linear1_weight, p_mod_list_2_linear1_bias, p_mod_list_2_linear2_weight, p_mod_list_2_linear2_bias, x):
    repeated_subgraph0 = self.repeated_subgraph0
    invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', token, x, p_mod_list_0_linear1_weight, p_mod_list_0_linear1_bias, p_mod_list_0_linear2_weight, p_mod_list_0_linear2_bias);  repeated_subgraph0 = token = x = p_mod_list_0_linear1_weight = p_mod_list_0_linear1_bias = p_mod_list_0_linear2_weight = p_mod_list_0_linear2_bias = None
    getitem = invoke_subgraph[0]
    getitem_1 = invoke_subgraph[1];  invoke_subgraph = None
    repeated_subgraph0_1 = self.repeated_subgraph0
    invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', getitem, getitem_1, p_mod_list_1_linear1_weight, p_mod_list_1_linear1_bias, p_mod_list_1_linear2_weight, p_mod_list_1_linear2_bias);  repeated_subgraph0_1 = getitem = getitem_1 = p_mod_list_1_linear1_weight = p_mod_list_1_linear1_bias = p_mod_list_1_linear2_weight = p_mod_list_1_linear2_bias = None
    getitem_2 = invoke_subgraph_1[0]
    getitem_3 = invoke_subgraph_1[1];  invoke_subgraph_1 = None
    repeated_subgraph0_2 = self.repeated_subgraph0
    invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_2, 'subgraph_0', getitem_2, getitem_3, p_mod_list_2_linear1_weight, p_mod_list_2_linear1_bias, p_mod_list_2_linear2_weight, p_mod_list_2_linear2_bias);  repeated_subgraph0_2 = getitem_2 = getitem_3 = p_mod_list_2_linear1_weight = p_mod_list_2_linear1_bias = p_mod_list_2_linear2_weight = p_mod_list_2_linear2_bias = None
    getitem_4 = invoke_subgraph_2[0]
    getitem_5 = invoke_subgraph_2[1];  invoke_subgraph_2 = None
    with_effects = torch.ops.higher_order.with_effects(getitem_4, torch.ops.mylib.record_memory.default, 'forward', 'N');  getitem_4 = None
    getitem_6 = with_effects[0];  with_effects = None
    return (getitem_6, getitem_5)""",
            )

            self.assertExpectedInline(
                decomp.graph_module.repeated_subgraph0.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    with_effects = torch.ops.higher_order.with_effects(arg0_1, torch.ops.mylib.record_memory.default, 'forward', 'N');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    addmm = torch.ops.aten.addmm.default(arg3_1, arg1_1, permute);  arg3_1 = arg1_1 = permute = None
    relu = torch.ops.aten.relu.default(addmm);  addmm = None
    permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    addmm_1 = torch.ops.aten.addmm.default(arg5_1, relu, permute_1);  arg5_1 = relu = permute_1 = None
    return (getitem, addmm_1)""",
            )

            recorded_list.clear()
            out2 = ep.module()(x)
            self.assertEqual(len(recorded_list), 4)
            self.assertTrue(torch.allclose(model(x)[0], out2[0]))

            # Test when we unlift the tokens from the graph. This is used in the inductor path.
            with (
                tracing(TracingContext(None)),
                torch._functorch.config.patch(unlift_effect_tokens=True),
            ):
                gm, gs = aot_export_module(ep.module(), (x,), trace_joint=False)
                self.assertExpectedInline(
                    str(gm.code).strip(),
                    """\
def forward(self, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1):
    _make_token_default = torch.ops.prims._make_token.default()
    repeated_subgraph0 = self.repeated_subgraph0
    with_effects_1 = torch.ops.higher_order.with_effects(_make_token_default, torch.ops.higher_order.invoke_subgraph, repeated_subgraph0, 'subgraph_0', arg13_1, arg1_1, arg2_1, arg3_1, arg4_1);  _make_token_default = repeated_subgraph0 = arg13_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = None
    getitem = with_effects_1[0]
    getitem_1 = with_effects_1[1];  with_effects_1 = None
    repeated_subgraph0_1 = self.repeated_subgraph0
    with_effects_2 = torch.ops.higher_order.with_effects(getitem, torch.ops.higher_order.invoke_subgraph, repeated_subgraph0_1, 'subgraph_0', getitem_1, arg5_1, arg6_1, arg7_1, arg8_1);  getitem = repeated_subgraph0_1 = getitem_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = None
    getitem_2 = with_effects_2[0]
    getitem_3 = with_effects_2[1];  with_effects_2 = None
    repeated_subgraph0_2 = self.repeated_subgraph0
    with_effects_3 = torch.ops.higher_order.with_effects(getitem_2, torch.ops.higher_order.invoke_subgraph, repeated_subgraph0_2, 'subgraph_0', getitem_3, arg9_1, arg10_1, arg11_1, arg12_1);  getitem_2 = repeated_subgraph0_2 = getitem_3 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = None
    getitem_4 = with_effects_3[0]
    getitem_5 = with_effects_3[1];  with_effects_3 = None
    with_effects = torch.ops.higher_order.with_effects(getitem_4, torch.ops.mylib.record_memory.default, 'forward', 'N');  getitem_4 = None
    getitem_6 = with_effects[0];  with_effects = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem_6]);  getitem_6 = _sink_tokens_default = None
    return (getitem_5,)""",
                )
                self.assertExpectedInline(
                    str(gm.repeated_subgraph0.code).strip(),
                    """\
def forward(self, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    _make_token_default = torch.ops.prims._make_token.default()
    with_effects = torch.ops.higher_order.with_effects(_make_token_default, torch.ops.mylib.record_memory.default, 'forward', 'N');  _make_token_default = None
    getitem = with_effects[0];  with_effects = None
    t = torch.ops.aten.t.default(arg2_1);  arg2_1 = None
    addmm = torch.ops.aten.addmm.default(arg3_1, arg1_1, t);  arg3_1 = arg1_1 = t = None
    relu = torch.ops.aten.relu.default(addmm);  addmm = None
    t_1 = torch.ops.aten.t.default(arg4_1);  arg4_1 = None
    addmm_1 = torch.ops.aten.addmm.default(arg5_1, relu, t_1);  arg5_1 = relu = t_1 = None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default([getitem]);  getitem = _sink_tokens_default = None
    return (addmm_1,)""",
                )

        recorded_list.clear()
        out2 = torch.compile(model)(x)
        self.assertEqual(len(recorded_list), 4)
        self.assertTrue(torch.allclose(model(x)[0], out2[0], atol=1e-7, rtol=1e-4))

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not TEST_CUDA, "cuda")
    def test_compile_cond_with_effect_in_branch(self):
        with torch.library._scoped_library("mylib_cond_effect", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op("mylib_cond_effect::record", mutates_args=())
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_cond_effect.record.default)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)

                def forward(self, x, pred):
                    def true_fn(x):
                        torch.ops.mylib_cond_effect.record(x.mean(), "true")
                        return x.clone()

                    def false_fn(x):
                        return x.clone()

                    x = torch.relu(self.linear(x))
                    return torch.cond(pred, true_fn, false_fn, (x,))

            x = torch.randn(5, 5, device="cuda")
            mod = torch.compile(M().cuda())

            recorded.clear()
            self.assertEqual(
                mod(x, torch.tensor([True], device="cuda")).shape, torch.Size([5, 5])
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, ["true"])

            recorded.clear()
            self.assertEqual(
                mod(x, torch.tensor([False], device="cuda")).shape, torch.Size([5, 5])
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, [])

            class NoGradM(M):
                def forward(self, x, pred):
                    with torch.no_grad():
                        return super().forward(x, pred)

            no_grad_mod = torch.compile(NoGradM().cuda())

            recorded.clear()
            self.assertEqual(
                no_grad_mod(x, torch.tensor([True], device="cuda")).shape,
                torch.Size([5, 5]),
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, ["true"])

            recorded.clear()
            self.assertEqual(
                no_grad_mod(x, torch.tensor([False], device="cuda")).shape,
                torch.Size([5, 5]),
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, [])

            class GradM(torch.nn.Module):
                def forward(self, x, pred):
                    def true_fn(x):
                        torch.ops.mylib_cond_effect.record(x.mean(), "true")
                        return x.sin() * x.cos()

                    def false_fn(x):
                        return x.sin() * x.cos()

                    return torch.cond(pred, true_fn, false_fn, (x,))

            grad_mod = torch.compile(GradM().cuda())

            x_grad = torch.randn(5, 5, device="cuda", requires_grad=True)
            recorded.clear()
            grad_mod(x_grad, torch.tensor([True], device="cuda")).sum().backward()
            torch.cuda.synchronize()
            self.assertEqual(recorded, ["true"])

            x_grad = torch.randn(5, 5, device="cuda", requires_grad=True)
            recorded.clear()
            grad_mod(x_grad, torch.tensor([False], device="cuda")).sum().backward()
            torch.cuda.synchronize()
            self.assertEqual(recorded, [])

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not TEST_CUDA, "cuda")
    def test_compile_nested_cond_with_effect_in_branch(self):
        with torch.library._scoped_library("mylib_nested_cond_effect", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op(
                "mylib_nested_cond_effect::record", mutates_args=()
            )
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_nested_cond_effect.record.default)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)

                def forward(self, x, pred, pred2):
                    def true_outer(pred2, x):
                        def true_inner(x):
                            torch.ops.mylib_nested_cond_effect.record(x.mean(), "inner")
                            return x.clone()

                        def false_inner(x):
                            return x.clone()

                        return torch.cond(pred2, true_inner, false_inner, (x,))

                    def false_outer(pred2, x):
                        return x.clone()

                    x = torch.relu(self.linear(x))
                    return torch.cond(pred, true_outer, false_outer, (pred2, x))

            x = torch.randn(5, 5, device="cuda")
            mod = torch.compile(M().cuda())

            recorded.clear()
            self.assertEqual(
                mod(
                    x,
                    torch.tensor([True], device="cuda"),
                    torch.tensor([True], device="cuda"),
                ).shape,
                torch.Size([5, 5]),
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, ["inner"])

            recorded.clear()
            self.assertEqual(
                mod(
                    x,
                    torch.tensor([True], device="cuda"),
                    torch.tensor([False], device="cuda"),
                ).shape,
                torch.Size([5, 5]),
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, [])

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not TEST_CUDA, "cuda")
    def test_compile_cond_with_effect_before_and_in_branch(self):
        with torch.library._scoped_library("mylib_cond_prior_effect", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op("mylib_cond_prior_effect::record", mutates_args=())
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_cond_prior_effect.record.default)

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(5, 5)

                def forward(self, x, pred):
                    def true_fn(x):
                        torch.ops.mylib_cond_prior_effect.record(x.mean(), "inside")
                        return x.clone()

                    def false_fn(x):
                        return x.clone()

                    x = torch.relu(self.linear(x))
                    torch.ops.mylib_cond_prior_effect.record(x.mean(), "before")
                    return torch.cond(pred, true_fn, false_fn, (x,))

            x = torch.randn(5, 5, device="cuda")
            mod = torch.compile(M().cuda())

            recorded.clear()
            self.assertEqual(
                mod(x, torch.tensor([True], device="cuda")).shape, torch.Size([5, 5])
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, ["before", "inside"])

            recorded.clear()
            self.assertEqual(
                mod(x, torch.tensor([False], device="cuda")).shape, torch.Size([5, 5])
            )
            torch.cuda.synchronize()
            self.assertEqual(recorded, ["before"])

    def test_export_run_decompositions_cond_with_effect_in_branch(self):
        with torch.library._scoped_library("mylib_export_cond_effect", "FRAGMENT"):
            recorded = []

            @torch.library.custom_op(
                "mylib_export_cond_effect::record", mutates_args=()
            )
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)
                return

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_export_cond_effect.record.default)

            class M(torch.nn.Module):
                def forward(self, x, pred):
                    def true_fn(x):
                        torch.ops.mylib_export_cond_effect.record(x.mean(), "true")
                        return x.clone()

                    def false_fn(x):
                        return x.clone()

                    return torch.cond(pred, true_fn, false_fn, (x,))

            ep = torch.export.export(M(), (torch.randn(2, 2), torch.tensor(True)))
            decomp = ep.run_decompositions()
            self.assertIn("torch.ops.higher_order.cond", decomp.graph_module.code)
            self.assertIn(
                "torch.ops.higher_order.with_effects",
                decomp.graph_module.true_graph_0.code,
            )

            # A negative index still selects the last user result after the
            # leading token is removed; it must not be classified as a token.
            cond_node = next(
                node
                for node in decomp.graph_module.graph.nodes
                if node.target is torch.ops.higher_order.cond
            )
            false_graph_node = cond_node.args[2]
            holder = torch.nn.Module()
            holder.add_module("branch", decomp.graph_module.false_graph_0)
            decomp.graph_module.add_module("aliased", holder)
            false_graph_node.target = "aliased.branch"
            token_node = next(
                user
                for user in cond_node.users
                if user.target is operator.getitem and user.args[1] == 0
            )
            token_node.args = (cond_node, -2)
            result_node = next(
                user
                for user in cond_node.users
                if user.target is operator.getitem and user.args[1] == 1
            )
            result_node.args = (cond_node, -1)
            decomp.graph_module.recompile()

            # FX permits list-valued output nodes. Keep that container while the
            # module path removes the cond branch token prefix.
            for branch in (
                decomp.graph_module.true_graph_0,
                decomp.graph_module.false_graph_0,
            ):
                output = next(
                    node for node in branch.graph.nodes if node.op == "output"
                )
                output.args = (list(output.args[0]),)
                branch.recompile()

            x = torch.randn(2, 2)
            original_true_graph = decomp.graph_module.true_graph_0
            for _ in range(2):
                mod = decomp.module()
                mod_cond = next(
                    node
                    for node in mod.graph.nodes
                    if node.target is torch.ops.higher_order.cond
                )
                branches = tuple(
                    mod.get_submodule(branch_node.target)
                    for branch_node in mod_cond.args[1:3]
                )
                self.assertIsNot(branches[0], original_true_graph)
                for branch in branches:
                    output = next(
                        node for node in branch.graph.nodes if node.op == "output"
                    )
                    self.assertIsInstance(output.args[0], list)
                recorded.clear()
                mod(x, torch.tensor(True))
                self.assertEqual(recorded, ["true"])
                recorded.clear()
                mod(x, torch.tensor(False))
                self.assertEqual(recorded, [])

            # Materializing a module must not strip tokens from the source EP.
            self.assertIn(
                "torch.ops.higher_order.with_effects", original_true_graph.code
            )

            for _ in range(2):
                unflattened = torch.export.unflatten(decomp)
                recorded.clear()
                unflattened(x, torch.tensor(True))
                self.assertEqual(recorded, ["true"])

            class PriorEffectM(torch.nn.Module):
                def forward(self, x, pred):
                    def true_fn(x):
                        torch.ops.mylib_export_cond_effect.record(x.mean(), "inside")
                        return x.clone()

                    def false_fn(x):
                        return x.clone()

                    torch.ops.mylib_export_cond_effect.record(x.mean(), "before")
                    return torch.cond(pred, true_fn, false_fn, (x,))

            ep = torch.export.export(
                PriorEffectM(), (torch.randn(2, 2), torch.tensor(True))
            )
            decomp = ep.run_decompositions()
            mod = decomp.module()
            recorded.clear()
            mod(x, torch.tensor(True))
            self.assertEqual(recorded, ["before", "inside"])
            recorded.clear()
            mod(x, torch.tensor(False))
            self.assertEqual(recorded, ["before"])

            class EffectOnlyCondM(torch.nn.Module):
                def forward(self, x, pred):
                    def true_fn(x):
                        torch.ops.mylib_export_cond_effect.record(
                            x.mean(), "effect_only"
                        )
                        return ()

                    def false_fn(x):
                        return ()

                    torch.cond(pred, true_fn, false_fn, (x,))
                    return x + 1

            ep = torch.export.export(
                EffectOnlyCondM(), (x, torch.tensor(True))
            ).run_decompositions()
            for materialize in (ep.module, lambda: torch.export.unflatten(ep)):
                recorded.clear()
                self.assertEqual(materialize()(x, torch.tensor(True)), x + 1)
                self.assertEqual(recorded, ["effect_only"])
                recorded.clear()
                self.assertEqual(materialize()(x, torch.tensor(False)), x + 1)
                self.assertEqual(recorded, [])
                ep.validate()

    def test_export_run_decompositions_cond_with_empty_tensor_operand(self):
        with torch.library._scoped_library(
            "mylib_export_cond_empty_effect", "FRAGMENT"
        ):
            recorded = []

            @torch.library.custom_op(
                "mylib_export_cond_empty_effect::record", mutates_args=()
            )
            def record(x: torch.Tensor, prefix: str) -> None:
                recorded.append(prefix)
                return

            @record.register_fake
            def record_fake(x, prefix):
                return

            record.register_effect(_EffectType.ORDERED)
            has_side_effect(torch.ops.mylib_export_cond_empty_effect.record.default)

            class M(torch.nn.Module):
                def forward(self, empty, x, pred):
                    def true_fn(empty, x):
                        torch.ops.mylib_export_cond_empty_effect.record(
                            x.mean(), "true"
                        )
                        return x + empty.sum() + 1

                    def false_fn(empty, x):
                        return x - empty.sum() - 1

                    return torch.cond(pred, true_fn, false_fn, (empty, x))

            empty = torch.empty(0)
            x = torch.randn(2, 2)
            ep = torch.export.export(M(), (empty, x, torch.tensor(True)))
            decomp = ep.run_decompositions()
            mod = decomp.module()

            recorded.clear()
            true_out = mod(empty, x, torch.tensor(True))
            self.assertEqual(recorded, ["true"])
            self.assertTrue(torch.allclose(true_out, x + 1))

            recorded.clear()
            false_out = mod(empty, x, torch.tensor(False))
            self.assertEqual(recorded, [])
            self.assertTrue(torch.allclose(false_out, x - 1))

    @skipIfTorchDynamo()
    def test_effect_autograd_function(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:

            @torch.library.custom_op("mylib::log_grad", mutates_args=())
            def log_grad(x: torch.Tensor) -> torch.Tensor:
                return x.clone()

            @torch.library.register_fake("mylib::log_grad")
            def log_grad_fake(x: torch.Tensor) -> torch.Tensor:
                return x.clone()

            log_grad.register_effect(_EffectType.ORDERED)

            class NoOpWithLoggingBackward(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x * x

                @staticmethod
                def backward(ctx, grad_output):
                    logged_grad = torch.ops.mylib.log_grad(grad_output)
                    return logged_grad

            def fn(x):
                y = NoOpWithLoggingBackward.apply(x)
                return y.sum()

            x = torch.randn(3, 4, requires_grad=True)
            x_clone = x.detach().clone().requires_grad_(True)

            backend = AotEagerAndRecordGraphs()
            compiled_fn = torch.compile(fn, backend=backend)
            loss = compiled_fn(x)
            loss.backward()

            loss_ref = fn(x_clone)
            loss_ref.backward()
            self.assertEqual(loss, loss_ref)

            self.assertExpectedInline(
                backend.fw_graphs[0].code.strip(),
                """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, primals_1);  primals_1 = None
    sum_1 = torch.ops.aten.sum.default(mul);  mul = None
    return (sum_1,)""",
            )

            self.assertExpectedInline(
                backend.bw_graphs[0].code.strip(),
                """\
def forward(self, tangents_1, tangents_token):
    expand = torch.ops.aten.expand.default(tangents_1, [3, 4]);  tangents_1 = None
    with_effects = torch.ops.higher_order.with_effects(tangents_token, torch.ops.mylib.log_grad.default, expand);  tangents_token = expand = None
    getitem = with_effects[0]
    getitem_1 = with_effects[1];  with_effects = None
    return (getitem_1, getitem)""",
            )

    def test_with_effects_through_functional_tensor_mode(self):
        from torch._subclasses.functional_tensor import (
            FunctionalTensor,
            FunctionalTensorMode,
        )

        def fn_with_effects(x, y):
            token = torch.ops.prims._make_token()
            new_token, result = with_effects(
                token,
                torch.ops.aten.add.Tensor,
                x,
                y,
            )
            return result

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        with (
            torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
            ),
            FunctionalTensorMode(),
        ):
            x_func = FunctionalTensor.to_functional(x)
            y_func = FunctionalTensor.to_functional(y)
            result = fn_with_effects(x_func, y_func)

        expected = x + y
        if isinstance(result, FunctionalTensor):
            result = torch._from_functional_tensor(result.elem)
        self.assertEqual(result, expected)

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_effectful_op_with_flex_attention(self):
        """Test that effectful custom ops work with flex_attention."""
        from torch._library.effects import EffectType
        from torch.nn.attention.flex_attention import flex_attention

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:

            @torch.library.custom_op("mylib::noop", mutates_args=())
            def noop(x: torch.Tensor) -> torch.Tensor:
                return x.clone()

            @noop.register_fake
            def noop_fake(x: torch.Tensor) -> torch.Tensor:
                return x.clone()

            noop.register_effect(EffectType.ORDERED)

            def score_mod(score, b, h, q_idx, kv_idx):
                return score

            def fn(q, k, v):
                q = torch.ops.mylib.noop(q)
                out = flex_attention(q, k, v, score_mod=score_mod)
                return out

            batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
            q = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            k = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            v = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                device="cuda",
                dtype=torch.float16,
            )

            compiled_fn = torch.compile(fn)
            out = compiled_fn(q, k, v)
            self.assertEqual(out.shape, (batch_size, num_heads, seq_len, head_dim))


if __name__ == "__main__":
    run_tests()
