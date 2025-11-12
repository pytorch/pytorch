# Owner(s): ["module: functorch"]
# ruff: noqa: F841
# flake8: noqa: B950
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
from torch._functorch.aot_autograd import aot_export_module
from torch._guards import tracing, TracingContext
from torch._higher_order_ops.effects import (
    _EffectType,
    _get_effect,
    _register_effectful_op,
    with_effects,
)
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM70OrLater, SM80OrLater
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_WITH_ROCM,
    TestCase,
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
    return (add,)""",  # noqa: B950
            )

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
    return (getitem, add)""",  # noqa: B950
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
                assert input.device == output.device
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
    @unittest.skipIf(TEST_WITH_ROCM, "triton")
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

    @unittest.skipIf(not TEST_CUDA, "triton")
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
    return (getitem_5,)""",  # noqa: B950
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
    return (addmm_1,)""",  # noqa: B950
                )

        recorded_list.clear()
        out2 = torch.compile(model)(x)
        self.assertEqual(len(recorded_list), 4)
        self.assertTrue(torch.allclose(model(x)[0], out2[0], atol=1e-7, rtol=1e-4))


if __name__ == "__main__":
    run_tests()
