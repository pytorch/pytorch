# Owner(s): ["module: ProxyTensor"]
# ruff: noqa: F841

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch._dynamo
import unittest
import warnings
import operator
from collections.abc import Iterable
from torch.nn.utils import stateless
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, skip, xfail, skipOps
from torch._subclasses.fake_tensor import DynamicOutputShapeException, DataDependentOutputException, FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._decomp import decomposition_table
from torch.fx.experimental.symbolic_shapes import (
    eval_guards, bind_symbols, fx_placeholder_vals, fx_placeholder_targets,
    guard_int, GuardOnDataDependentSymNode
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.common_device_type import ops
import torch.testing._internal.optests as optests
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import make_fx, DecompositionInterpreter, get_isolated_graphmodule
from torch.utils._pytree import tree_map
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch import nn
import torch._functorch.config
import re

import functools
import itertools

aten = torch.ops.aten

HAS_CUDA = torch.cuda.is_available()


def strip_end(s, suffix):
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


def show_guards(gm):
    names = [strip_end(n, "_1") for n in fx_placeholder_targets(gm)]
    return "\n".join(
        gm.shape_env.produce_guards(fx_placeholder_vals(gm), names, _simplified=True, input_contexts=None)
    )


def process_failures():
    """
    Takes file containing failures like

    FAILED test/test_proxy_tensor.py::TestProxyTensorOpInfoCPU::test_make_fx_symbolic_exhaustive___getitem___cpu_float32 - RuntimeError: aten.size.default - couldn't find symbolic meta function/decomposition  # noqa: B950

    and processes them into a list of opinfo xfails
    """
    f = open('pytest_failures')
    failures = f.readlines()
    failures = [i.strip() for i in failures]

    def process_failure_string(s, matcher):
        out = re.search(matcher, s)
        return out.groups()

    SYMBOLIC_TRACE_MATCH = r'exhaustive_(.*)_cpu.*: (.*)'
    failures = [process_failure_string(s, SYMBOLIC_TRACE_MATCH) for s in failures]

    def create_normalized_name(op):
        if op.variant_test_name == '':
            s = op.name
        else:
            s = f"{op.name}.{op.variant_test_name}"
        return s.replace('.', '_')

    remap_opinfo = {create_normalized_name(op): (op.name, op.variant_test_name) for op in op_db}

    print("symbolic_tensor_failures = {")
    for failure, reason in failures:
        print(f"    xfail{remap_opinfo[failure]},  # {reason}")
    print("}")


USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)


def _create_new_input(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype != torch.float:
        return x + 1
    if x.is_leaf:
        return torch.rand_like(x, requires_grad=x.requires_grad)
    else:
        return torch.rand_like(x)

"""
Delays a cos being executed on the unwraptensor until its used. Simulates a CommTensor used
"""
class UnwrapTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )
        r._tensor = tensor
        return r

    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"UnwrapTensor({self._tensor})"

    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            ret = e
            if isinstance(e, UnwrapTensor):
                ret = e._tensor.cos()

            return ret

        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)
        return func(*args, **kwargs)

class TestGenericProxyTensor(TestCase):
    # WARNING: if any of your inputs are index tensors, DO NOT use this
    # function
    def _test(self, f, inps):
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(*inps)
        new_inps = tree_map(_create_new_input, inps)
        r1 = fx_f(*new_inps)
        r2 = f(*new_inps)
        self.assertEqual(r1, r2)

    def test_pre_dispatch_mode_stack(self):
        def f(a):
            b = torch.ones(4, 4)
            return torch.matmul(a, b)
        # We expect to see matmul in the trace - it should NOT be decomposed into mm.
        # Also, torch.ones() doesn't show up in the trace.
        # This is annoying but expected: ones() never dispatches to the Autograd dispatch key,
        # so our mode never sees it - it goes directly to the BackendSelect key.
        inp = torch.ones(4, 4)
        # Test that make_fx(pre_dispatch=True) clears caches properly.
        from torch._dispatch.python import enable_python_dispatcher
        with enable_python_dispatcher():
            out1 = f(inp)
        fx_g = make_fx(f, pre_dispatch=True)(inp)
        self.assertExpectedInline(fx_g.code.strip(), """\
def forward(self, a_1):
    ones = torch.ops.aten.ones.default([4, 4], device = device(type='cpu'), pin_memory = False)
    matmul = torch.ops.aten.matmul.default(a_1, ones);  a_1 = ones = None
    return matmul""")

    def test_pre_dispatch_linear(self):
        def f(a, b, c):
            return torch.nn.functional.linear(a, b, c)
        a = torch.ones(4, 4)
        b = torch.ones(4, 4)
        c = torch.ones(4)
        fx_g = make_fx(f, pre_dispatch=True)(a, b, c)
        out1 = f(a, b, c)
        out2 = fx_g(a, b, c)
        self.assertEqual(out1, out2)

    def test_pre_dispatch_no_grad(self):
        def f(a):
            b = a.sin()
            torch.set_grad_enabled(False)
            c = b.cos()
            torch.set_grad_enabled(True)
            return b + c.sin()
        a1 = torch.randn(4, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        a_tmp = a1.detach().clone().requires_grad_(True)
        fx_g = make_fx(f, pre_dispatch=True)(a_tmp)
        out1 = f(a1)
        out2 = fx_g(a2)
        self.assertEqual(out1, out2)
        out1.sum().backward()
        out2.sum().backward()
        self.assertEqual(a1.grad, a2.grad)

    def test_make_fx_simple(self):
        def f(x):
            return torch.sin(x)
        self._test(f, (torch.randn(3),))

    def test_scalar_device(self, device='cpu'):
        def f(a, b):
            return a + b
        self._test(f, [torch.randn(3, device=device), torch.tensor(5)])

    def test_isolated_graphmodule(self):
        def is_any_sum(gm):
            return any(node.target == torch.ops.aten.sum.default for node in gm.graph.nodes)

        def is_any_digamma(gm):
            return any(node.target == torch.ops.aten.digamma.default for node in gm.graph.nodes)

        def is_any_sigmoid(gm):
            return any(node.target == torch.ops.aten.sigmoid.default for node in gm.graph.nodes)

        def inner(x):
            return torch.sum(x)

        def f(x):
            gm = get_isolated_graphmodule(inner, (x,), {})
            self.assertTrue(is_any_sum(gm))
            return x + torch.randn(x.shape)

        # get_isolated_graphmodule uses make_fx internally that shouldn't be traced
        # by the outer make_fx call
        traced = make_fx(f)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))

        # When factory functions are used, they should not be traced
        # by the outer make_fx call
        def inner_with_factory():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((10, 10), val).sum()

        def f1(x):
            gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2(x):
            gm = get_isolated_graphmodule(f1, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify nested make_fx calls don't make factory functions to be leaked
        # into the outer graph. Verify that `make_fx`` itself does not leak its execution.
        def f2(x):
            gm = make_fx(f1)(x)
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify that the `forward`` function of a graph module produced as a
        # side effect of an interior `make_fx` is still traced
        def f3(x):
            gm = make_fx(f1)(x)
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            # `gm.forward`` is still traced
            return torch.digamma(gm(x))

        traced = make_fx(f3)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertTrue(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify interaction with non-ProxyTensor modes
        from torch.testing._internal.logging_tensor import LoggingTensorMode

        def f1_logging(x):
            with LoggingTensorMode():
                gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2_logging(x):
            with LoggingTensorMode(), LoggingTensorMode():
                gm = get_isolated_graphmodule(f1_logging, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2_logging)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))
        self.assertTrue(is_any_digamma(traced))

        # Verify interaction with another tensor subclass
        # This case currently doesn't work and should raise an error
        # See: https://github.com/pytorch/pytorch/pull/81764#issuecomment-1200472068
        from torch.testing._internal.logging_tensor import LoggingTensor

        def f1_logging_tensor(x):
            gm = get_isolated_graphmodule(inner_with_factory, (), {})
            self.assertTrue(is_any_sum(gm))
            return torch.sigmoid(x)

        def f2_logging_tensor(x):
            x = LoggingTensor(x)
            gm = get_isolated_graphmodule(f1_logging_tensor, (x,), {})
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2_logging_tensor)(torch.randn(3))
        self.assertFalse(is_any_sum(traced))
        self.assertFalse(is_any_sigmoid(traced))  # this fails, sigmoid is traced with LoggingTensor
        self.assertTrue(is_any_digamma(traced))

    # See https://github.com/pytorch/pytorch/issues/97541
    def test_empty_like_doesnt_burn_in_defaults(self):
        def f(x):
            return torch.empty_like(x)
        out = make_fx(f)(torch.randn(3))
        self.assertExpectedInline(out.code.strip(), """\
def forward(self, x_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False);  x_1 = None
    return empty_like""")

    def test_proxy_tensor_mode_with_decomp_table_preserves_proxy(self):
        def f(x):
            y = x.new_zeros(x.size())
            y.copy_(x)
            return y

        def _new_zeros_decomp(inp, size, dtype=None, layout=None, device=None, pin_memory=None):
            return torch.zeros(size, dtype=inp.dtype, device=inp.device)

        factory_func_decomp = {torch.ops.aten.new_zeros.default: _new_zeros_decomp}

        # When new_zeros() decomposes into torch.zero(), we expect ProxyTensorMode
        # to still be (re-entrantly) enabled, so that the `torch.zero()` call
        # returns a ProxyTensor.
        out = make_fx(f, decomposition_table=factory_func_decomp)(torch.ones(2))
        self.assertExpectedInline(out.code, """\



def forward(self, x_1):
    zeros = torch.ops.aten.zeros.default([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    copy_ = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    return copy_
    """)

    def test_make_fx_reentrant_dispatch(self):
        def f(x):
            return torch.ops.aten.norm.Scalar(x, 2.0)

        def norm_decomp(x, p=2.0):
            if p != 2.0:
                raise RuntimeError("can't handle with p != 2")
            return torch.sqrt(torch.sum(torch.square(x)))

        decomp = {torch.ops.aten.norm.Scalar: norm_decomp}

        traced = make_fx(f, decomposition_table=decomp, tracing_mode=self.tracing_mode)(torch.rand(3))

        for n in traced.graph.nodes:
            self.assertTrue("square" not in str(n.target))
            self.assertTrue("norm" not in str(n.target))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_backward_trace(self):
        mod = torchvision.models.resnet18()

        # An old version of this test called the module directly.  This works
        # for tracing_mode == "real", but for fake tensors, we also have to
        # ensure that the parameters and buffers get wrapped in fake tensors
        # because free fake tensors are not supported.  Fortunately functional_call
        # does precisely this for us.
        def f(x, params, buffers):
            for p in params.values():
                p.grad = None
            loss = torch.func.functional_call(mod, {**params, **buffers}, (x,)).sum()
            # I could have done this with the functional API, but there is
            # plenty of exercising this; I want to show mutating API still
            # works
            loss.backward()
            return [p.grad for p in params.values()]

        inp = torch.randn(3, 3, 250, 250)
        self._test(f, [inp, dict(mod.named_parameters()), dict(mod.named_buffers())])

    def test_varargs(self):
        def f(*args):
            return sum(args)

        self._test(f, [torch.randn(2), torch.randn(2)])

    def test_proxy_tensor(self):
        def f_grad(x):
            val = x.cos().cos().sum()
            return torch.autograd.grad(val, x)

        def f_backward(x):
            val = x.cos().cos().sum()
            val.backward()
            return x.grad

        for f in [f_grad, f_backward]:
            self._test(f, [torch.randn(3, requires_grad=True)])

    def test_pickle_issue89626(self):
        import pickle
        x = torch.randn(2)
        make_fx(lambda x: x * 2, tracing_mode=self.tracing_mode)(x)
        pickle.dumps(x)

    def test_inplace_metadata(self):
        def f(x):
            x = x.clone()
            x.unsqueeze_(-1)
            assert x.shape[-1] == 1
            return x

        self._test(f, [torch.randn(5)])

    def test_mode_tracing_factory_function(self):
        def f(x):
            return x + torch.randn(x.shape)

        # default behavior should trace factory functions
        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))
        self.assertTrue(
            any(
                node.target == aten.randn.default
                for node in traced.graph.nodes
            )
        )

    def test_pre_dispatch_functionalization(self):
        def f(x):
            a = FunctionalTensorMode(pre_dispatch=True, export=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                y = y + x_unwrapped
                y.mul_(5)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        from torch._dispatch.python import enable_python_dispatcher

        with enable_python_dispatcher():
            inp = torch.randn(4, 4)
            gm = make_fx(f, pre_dispatch=True)(inp)

        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1):
    matmul = torch.ops.aten.matmul.default(x_1, x_1)
    add = torch.ops.aten.add.Tensor(matmul, x_1);  matmul = x_1 = None
    mul = torch.ops.aten.mul.Tensor(add, 5);  add = None
    return mul""")

    def test_pre_dispatch_functionalization_view_op(self):
        def f(x):
            a = FunctionalTensorMode(pre_dispatch=True, export=True)
            with a:
                x_unwrapped = FunctionalTensor.to_functional(x)
                y = torch.matmul(x_unwrapped, x_unwrapped)
                x_unwrapped = x_unwrapped.transpose(1, 0)
                y = y + x_unwrapped
                y = y.view(2, 8)
                y_unwrapped = torch._from_functional_tensor(y.elem)
                return y_unwrapped

        from torch._dispatch.python import enable_python_dispatcher

        with enable_python_dispatcher():
            inp = torch.randn(4, 4)
            gm = make_fx(f, pre_dispatch=True)(inp)

        # TODO actually not decompose
        self.assertExpectedInline(gm.code.strip(), """\
def forward(self, x_1):
    matmul = torch.ops.aten.matmul.default(x_1, x_1)
    transpose = torch.ops.aten.transpose.int(x_1, 1, 0);  x_1 = None
    add = torch.ops.aten.add.Tensor(matmul, transpose);  matmul = transpose = None
    view = torch.ops.aten.view.default(add, [2, 8]);  add = None
    return view""")

    def test_val_metadata_mutation(self):
        def f(x):
            y = x.clone()
            y.unsqueeze_(0)
            return y

        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3, requires_grad=True))
        self.assertEqual([
            tuple(node.meta['val'].shape)
            for node in traced.graph.nodes
            if 'val' in node.meta
        ], [(3,), (3,), (1, 3)])

    def test_make_fx_overloads(self):
        def f(x):
            return x.cos() + torch.randn(x.shape)

        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))

        self.assertTrue(all(isinstance(node.target, torch._ops.OpOverload)
                            for node in traced.graph.nodes if node.op == 'call_function'))

    def test_tensor_constants(self):
        def f():
            val = torch.tensor(float('inf'))
            return torch.full((100, 100), val)

        self._test(f, [])

    def test_allclose(self):
        def f(a, b):
            return torch.allclose(a, b)

        def test_f():
            make_fx(f, tracing_mode=self.tracing_mode)(
                torch.zeros(3), torch.zeros(3)
            )

        if self.tracing_mode != "real":
            self.assertRaises(DataDependentOutputException, test_f)
        else:
            self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_constant_proxy_tensor_mut(self):
        def f():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((100, 100), val)

        g = make_fx(f, tracing_mode=self.tracing_mode)()
        self.assertEqual(g(), f())
        # In case we mutated shared state in the g graph!
        self.assertEqual(g(), f())

    def test_constant_unbind(self):
        def f():
            val = torch.tensor([2])
            r, = torch.unbind(val, 0)
            return r.item()

        g = make_fx(f, tracing_mode=self.tracing_mode)()
        self.assertEqual(g(), f())

    def test_constant_blowup(self):
        def f():
            val = torch.tensor([2])
            blowup = val.repeat(1000)
            return bool(blowup.sum().item() == 2)

        def test_f():
            make_fx(f, tracing_mode=self.tracing_mode)()

        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_constant_random(self):
        def f():
            val = torch.tensor([2.0])
            val.normal_()
            return bool(val.item() == 2.1)

        def test_f():
            make_fx(f, tracing_mode=self.tracing_mode)()

        self.assertRaisesRegex(RuntimeError, "data-dependent", test_f)

    def test_decomposition_interpreter(self):
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.rand((4, 4))
        fx_module = make_fx(fn, tracing_mode=self.tracing_mode, decomposition_table=None)(x)

        found_silu = False
        for n in fx_module.graph.nodes:
            if n.target == torch.ops.aten.silu or n.target == torch.ops.aten.silu.default:
                found_silu = True

        self.assertTrue(found_silu)

        new_graph = torch.fx.Graph()
        silu_decomp_table = {torch.ops.aten.silu.default: decomposition_table[torch.ops.aten.silu.default]}
        DecompositionInterpreter(
            fx_module,
            new_graph=new_graph,
            decomposition_table=silu_decomp_table,
        ).run(x)

        decomposed_module = torch.fx.GraphModule(fx_module, new_graph)

        for n in decomposed_module.graph.nodes:
            self.assertTrue(n.target != torch.ops.aten.silu)
            self.assertTrue(n.target != torch.ops.aten.silu.default)

        self.assertEqual(fx_module(x), decomposed_module(x))

    def test_make_fx_model_fwd_bwd(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(x, params):
            out = torch.func.functional_call(model, params, x).sum()
            out.backward()
            return list(params.values())
        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(input, params)
        # fx may change the order of parameters in list, so using set() to compare
        self.assertTrue(
            torch.allclose(fx_f(input, params)[0], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[0], f(input, params)[1])
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params)[1], f(input, params)[0])
            or
            torch.allclose(fx_f(input, params)[1], f(input, params)[1])
        )

    def test_make_fx_model_double_param(self):
        class Emformer(torch.nn.Module):
            def __init__(
                self,
                input_dim: int = 256,
            ) -> None:
                super().__init__()

                self.layer_norm = torch.nn.LayerNorm(input_dim)

            def forward(mod_self, x):  # noqa: B902
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                y = mod_self.layer_norm(x)
                self.assertTrue(isinstance(mod_self.layer_norm.weight, torch.Tensor))
                z = mod_self.layer_norm(y)
                return z


        gm = make_fx(Emformer())(torch.randn(16, 1, 256))
        ops = {n.target for n in gm.graph.nodes if n.op == 'call_function'}
        self.assertEqual(len(ops), 2)


    def test_make_fx_model_fwd_bwd_wgtupdate(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(args, params, buffers):
            for p in params.values():
                p.grad = None
            if not isinstance(args, Iterable):
                args = [args]
            params_and_buffers = {**params, **buffers}
            out = torch.func.functional_call(model, params_and_buffers, args)
            out.sum().backward()
            return [p - 1e-4 * p.grad for p in params.values()]

        input = torch.randn(3, 5, requires_grad=True)
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        fx_f = make_fx(f, tracing_mode=self.tracing_mode)(input, params, buffers)
        # fx may change the order of parameters in list, so using set() to compare
        # also there is a numerical difference in results so changing atol from 1e-08 to 1e-03
        self.assertTrue(
            torch.allclose(fx_f(input, params, buffers)[0], f(input, params, buffers)[0], atol=1e-03)
            or
            torch.allclose(fx_f(input, params, buffers)[0], f(input, params, buffers)[1], atol=1e-03)
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params, buffers)[1], f(input, params, buffers)[0], atol=1e-03)
            or
            torch.allclose(fx_f(input, params, buffers)[1], f(input, params, buffers)[1], atol=1e-03)
        )

    def test_trace_subclasses(self):
        def f1(x):
            x = UnwrapTensor(x)
            y = x * 2
            return y

        def f2(x):
            wrapped = UnwrapTensor(x)
            y = x * wrapped
            return y

        inp = [torch.randn(5)]
        self._test(f1, inp)
        self._test(f2, inp)

    def test_partial_decomp(self):
        def f(a, b, c):
            x = torch.addmm(a, b, c)
            y = torch.addmm(a, b, c, beta=2, alpha=1)
            return x + y
        inps = [torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5)]
        fx_g = make_fx(f)(*inps)

        def addmm(a, b, c, beta=1, alpha=1):
            if beta == 1 and alpha == 1:
                return NotImplemented
            return beta * a + alpha * (b @ c)

        decomposed_fx = make_fx(f, decomposition_table={aten.addmm.default: addmm})(*inps)

        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        self.assertEqual(len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2)
        self.assertEqual(len([n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]), 1)

    def test_decomp_of_capture(self):
        val = torch.randn(5)

        def f(x):
            return x.t() + val.t()

        def nop(x):
            return x.cos()

        traced = make_fx(f, decomposition_table={torch.ops.aten.t.default: nop})(torch.randn(5))
        self.assertEqual(len([n for n in traced.graph.nodes if n.target == torch.ops.aten.t.default]), 0)


    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_amp_cache(self):
        layer = torch.nn.Conv2d(3, 3, 3).cuda()

        def f(x, w):
            return torch.nn.functional.conv2d(x, w, stride=layer.stride)

        inp = torch.randn(4, 3, 10, 10, device='cuda')
        with torch.autocast('cuda'):
            out_graph = make_fx(f)(inp, layer.weight).graph
            out_graph2 = make_fx(f)(inp, layer.weight).graph

        self.assertEqual(len(out_graph.nodes), len(out_graph2.nodes))
        for a, b in zip(out_graph.nodes, out_graph2.nodes):
            self.assertEqual(a.op, b.op)

    def test_strides(self):
        def f(x):
            self.assertTrue(x.is_contiguous())
            self.assertFalse(x.is_contiguous(memory_format=torch.channels_last))
            x = x.permute(0, 3, 1, 2)
            self.assertFalse(x.is_contiguous())
            self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
            return x
        make_fx(f)(torch.randn(2, 3, 4, 5))

        def f(x):
            self.assertTrue(x.is_contiguous())
            y = x[:, 1]
            self.assertFalse(y.is_contiguous())
            y = x[:, ::2]
            self.assertFalse(y.is_contiguous())
            return x.cos()

        make_fx(f)(torch.randn(2, 3, 4, 5))

    def test_pr_86917(self):
        # Tests the issue brought up here https://github.com/pytorch/pytorch/pull/86917#issuecomment-1283155344
        def f(a, b):
            return torch.ops.aten.nll_loss_forward(a, b, None, 1, 10)

        self._test(f, [torch.randn(1, 10), torch.zeros(1, dtype=torch.long)])

class TestGenericProxyTensorReal(TestGenericProxyTensor):
    tracing_mode = "real"


class TestGenericProxyTensorFake(TestGenericProxyTensor):
    tracing_mode = "fake"


class TestGenericProxyTensorSymbolic(TestGenericProxyTensor):
    tracing_mode = "symbolic"


del TestGenericProxyTensor


class TestRealProxyTensor(TestCase):
    def test_error_on_data_dependent_ops(self):
        def f():
            x = torch.randn([])
            y = torch.randn([])
            assert torch.allclose(x * y, y * x)
            z = float(x)
            z2 = float(y)

        # Smoke tests
        make_fx(f, _error_on_data_dependent_ops=False)()
        make_fx(f, pre_dispatch=True, _error_on_data_dependent_ops=False)()

class TestFakeProxyTensor(TestCase):
    def test_issue82547(self):
        x = nn.Parameter(torch.randn(3, 3))

        def f():
            return torch.ops.aten.t.default(x)
        self.assertRaisesRegex(Exception, "Please convert all Tensors", lambda: make_fx(f, tracing_mode="fake")())

        class A(torch.Tensor):
            pass

        x = A(torch.randn(3, 3))
        self.assertRaisesRegex(TypeError, "Multiple dispatch failed", lambda: make_fx(f, tracing_mode="fake")())

    def test_use_fake_and_tensor(self):
        def f(x, y):
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        g = make_fx(f, tracing_mode="fake")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        self.assertEqual(g(x, y), f(x, y))

    def test_free_fake(self):
        def f(x):
            return torch.add(x, y)

        with FakeTensorMode() as fake_mode:
            y = torch.randn(2)
            make_fx(f, tracing_mode="real")(torch.randn(2))

    def test_fused_adam(self):
        # See https://github.com/pytorch/pytorch/issues/99356
        params = [torch.randn(10, 10) for _ in range(10)]
        grads = [torch.randn(10, 10) for _ in range(10)]
        exp_avgs = [torch.randn(10, 10) for _ in range(10)]
        exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        max_exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        state_steps = [torch.tensor(0) for _ in range(10)]

        def fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps):
            (new_params, _, _, _, _) = aten._fused_adam.default(
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                lr=0.1,
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
                eps=1e-8,
                amsgrad=False,
                maximize=False,
            )

            for p, new_p in zip(params, new_params):
                p.copy_(new_p)

            return params

        gm = make_fx(fused_adam, tracing_mode='fake')(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )
        ensure_ops_have_val = [aten._fused_adam.default, operator.getitem]
        for n in gm.graph.nodes:
            if n.op == "call_function" and n.target in ensure_ops_have_val:
                self.assertIn('val', n.meta)

    def test_alias(self):
        def f(x):
            return torch.ops.aten.alias(x)

        r = str(make_fx(f, tracing_mode="fake")(torch.randn(2)).code).strip()
        # NB: this should not have a detach call
        self.assertExpectedInline(r, """\
def forward(self, x_1):
    alias = torch.ops.aten.alias.default(x_1);  x_1 = None
    return alias""")

    def test_meta(self):
        def f(x):
            a = x.cos()
            b = torch.var_mean(a, dim=0)
            c = b * 2
            return c

        out = make_fx(f, tracing_mode="fake")(torch.randn(5, 5))
        for n in out.graph.nodes:
            if n.op == 'output':
                continue
            self.assertTrue('val' in n.meta)

    def test_fake_tensor_mode(self):
        def f(a):
            d = a.cos()
            return d

        from torch._guards import detect_fake_mode

        existing_fake_mode = FakeTensorMode()
        with existing_fake_mode:
            out = make_fx(f, tracing_mode="real")(torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]))

        fake_mode = detect_fake_mode([node.meta.get('val', None) for node in out.graph.nodes])
        self.assertEqual(fake_mode, existing_fake_mode)

def _get_node(fx_g, cond):
    for n in fx_g.graph.nodes:
        if cond(n):
            return n
    raise AssertionError

def _get_free_symbols(shape_env):
    vars = tuple(shape_env.var_to_val.keys())
    return len([var for var in vars if var not in shape_env.replacements])

def _trace(f, *args):
    inps = [torch.randn(arg) for arg in args]
    return make_fx(f, tracing_mode="symbolic")(*inps)

# TODO: Need to test the guards themselves specifically as well
class TestSymbolicTracing(TestCase):
    def _test_dynamic(self, fn, trace_inputs, test_inputs, assert_eq=True):
        """
        Tests fn traced with trace_inputs against test_inputs
        Also returns shape env
        """
        trace_inputs = [torch.randn(shape) for shape in trace_inputs]
        traced_f = make_fx(fn, tracing_mode="symbolic")(*trace_inputs)
        for input in test_inputs:
            input = [torch.randn(shape) for shape in input]
            rx, ry = traced_f(*input), fn(*input)
            if assert_eq:
                self.assertEqual(rx, ry)
        return traced_f


    def test_debug_interpreter(self):
        import torch.library
        from torch.library import Library

        foo = Library("foo", "DEF")  # noqa: TOR901
        foo.define("foo(Tensor self) -> Tensor")

        # Operator where meta and cpu disagree on strides
        @torch.library.impl(foo, "foo", "CPU")
        def foo_cpu(x):
            return x.clone().T

        @torch.library.impl(foo, "foo", "Meta")
        def foo_meta(x):
            return x.clone()

        def f(x):
            return torch.ops.foo.foo.default(x)

        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2, 2))
        from torch._functorch.compilers import DebugInterpreter

        interp = DebugInterpreter(gm)

        # input mismatch is caught (indicates guard problem)
        self.assertRaisesRegex(
            AssertionError, r"3 != 1",
            lambda: interp.run(torch.randn(3, 3).T),
        )

        # Catch the incorrect meta
        self.assertRaisesRegex(
            AssertionError, r"\(3, 1\) != \(1, 3\)",
            lambda: interp.run(torch.randn(3, 3))
        )

    def test_int_input(self):
        def f(x, y):
            return x.view(y)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 4), 12).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    view = torch.ops.aten.view.default(x_1, [y_1]);  x_1 = y_1 = None
    return view""")

    def test_resize_from_zero(self):
        def f(x, y):
            x.resize_(y.size(0))

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(0), torch.empty(2)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    resize_ = torch.ops.aten.resize_.default(x_1, [sym_size_int]);  x_1 = sym_size_int = resize_ = None
    return None""")

    def test_broadcast_shapes(self):
        def f(x, y):
            return torch.functional.broadcast_shapes(x.size(), y.size()[0])

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(3, 1), torch.empty(5)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    return (sym_size_int, sym_size_int_1)""")

    def test_deduped_shape(self):
        def f(s0, s1, x, y):
            return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
            x = fake_mode.from_tensor(x)
            y = fake_mode.from_tensor(y)
            r = str(make_fx(f, tracing_mode="real")(x.shape[0], y.shape[0], x, y).code).strip()
            self.assertExpectedInline(r, """\
def forward(self, s0_1, s1_1, x_1, y_1):
    empty = torch.ops.aten.empty.memory_format([s0_1], device = device(type='cpu'), pin_memory = False)
    return ((s0_1, s1_1), empty)""")

    def test_non_deduped_shape(self):
        def f(x, y):
            return torch.functional.broadcast_shapes(x.size(), y.size()[0]), torch.empty(x.shape[0])

        x = torch.empty(3, 1)
        y = torch.empty(5)
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env = ShapeEnv()

        with FakeTensorMode(shape_env=shape_env, static_shapes=False) as fake_mode:
            x = fake_mode.from_tensor(x)
            y = fake_mode.from_tensor(y)
            r = str(make_fx(f, tracing_mode="real")(x, y).code).strip()
            self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0);  x_1 = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(y_1, 0);  y_1 = None
    empty = torch.ops.aten.empty.memory_format([sym_size_int], device = device(type='cpu'), pin_memory = False)
    return ((sym_size_int, sym_size_int_1), empty)""")

    def test_unary(self):
        def f(x):
            assert x.shape[0] < 20
            return x.cos()
        test_inputs = []
        test_inputs.append([(2, 5)])
        test_inputs.append([(6, 8)])
        gm = self._test_dynamic(f, [(3, 4)], test_inputs)
        self.assertTrue(eval_guards(gm, torch.randn(4, 5)))
        self.assertEqual(repr(bind_symbols(gm, torch.randn(4, 5))), "{s75: 4, s96: 5}")
        self.assertFalse(eval_guards(gm, torch.randn(25, 5)))
        self.assertExpectedInline(show_guards(gm), """L['x'].size()[0] <= 19""")

    def test_repeat_interleave(self):
        def f(src_tokens, beam_size_src):
            return src_tokens.repeat_interleave(beam_size_src.size(0), 0)

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = make_fx(f, tracing_mode="symbolic")(src_tokens, torch.randn(5))
        self.assertEqual(len(gm.shape_env.guards), 0)

    def test_non_symint_size_spec(self):
        # this isn't really a proxy tensor test, but it's the most convenient
        # way to get a fake tensor with symbolic sizes
        def f(x):
            torch._C._non_sym_sizes(x)
            return x + 1

        x = torch.randn(2, 3)
        make_fx(f, tracing_mode="symbolic")(x)

    # https://github.com/pytorch/pytorch/issues/108195
    def test_symbolic_repeat_interleave(self):
        def f(y, x):
            return y.repeat_interleave(x, dim=1)

        y = torch.tensor([[1, 2], [3, 4]])
        x = torch.tensor([2, 3])
        r = str(make_fx(f, tracing_mode="symbolic")(y, x).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, y_1, x_1):
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1);  x_1 = None
    index_select = torch.ops.aten.index_select.default(y_1, 1, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select""")

    def test_mod_gcd_unbacked(self):
        def f(_a, _b, _stride):
            a = _a.item()
            b = _b.item()
            stride = _stride.item()
            torch._check_is_size(a)
            torch._check_is_size(b)
            torch._check_is_size(stride)
            ta = torch.randn(a * stride)
            tb = torch.randn(b * stride)
            r = torch.cat([ta, tb])
            return r.view(a + b, stride)

        _a = torch.tensor(30)
        _b = torch.tensor(20)
        _stride = torch.tensor(10)
        r = str(make_fx(f, tracing_mode="symbolic")(_a, _b, _stride).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, _a_1, _b_1, _stride_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(_a_1);  _a_1 = None
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(_b_1);  _b_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(_stride_1);  _stride_1 = None
    mul = _local_scalar_dense * _local_scalar_dense_2
    randn = torch.ops.aten.randn.default([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    mul_1 = _local_scalar_dense_1 * _local_scalar_dense_2
    randn_1 = torch.ops.aten.randn.default([mul_1], device = device(type='cpu'), pin_memory = False);  mul_1 = None
    cat = torch.ops.aten.cat.default([randn, randn_1]);  randn = randn_1 = None
    add = _local_scalar_dense + _local_scalar_dense_1;  _local_scalar_dense = _local_scalar_dense_1 = None
    view = torch.ops.aten.view.default(cat, [add, _local_scalar_dense_2]);  cat = add = _local_scalar_dense_2 = None
    return view""")

    def test_cumsum_unbacked(self):
        def f(x):
            y = x.item()
            z = torch.randn((3, y, 3))
            return z.cumsum(0)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([5])).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    randn = torch.ops.aten.randn.default([3, _local_scalar_dense, 3], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    cumsum = torch.ops.aten.cumsum.default(randn, 0);  randn = None
    return cumsum"""  # noqa: B950
        )


    def test_repeat_interleave_unbacked_output_size(self):
        def f(x, y):
            s = x.sum().item()
            return y.repeat_interleave(x, dim=0, output_size=s)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor([2, 3]), torch.randn(2)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1, y_1):
    sum_1 = torch.ops.aten.sum.default(x_1)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(sum_1);  sum_1 = None
    repeat_interleave = torch.ops.aten.repeat_interleave.Tensor(x_1, output_size = _local_scalar_dense);  x_1 = _local_scalar_dense = None
    index_select = torch.ops.aten.index_select.default(y_1, 0, repeat_interleave);  y_1 = repeat_interleave = None
    return index_select"""  # noqa: B950
        )

    def test_arange_unbacked_output_size(self):
        def f(x):
            return torch.arange(0, x)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    arange = torch.ops.aten.arange.start(0, _local_scalar_dense, device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return arange"""  # noqa: B950
        )

    def test_adv_index_batch(self):
        def f(src_tokens):
            bsz, src_len = src_tokens.size()[:2]
            start_step = src_tokens.shape[1]
            beam_size = 1
            generate_size = 64
            max_len = src_len + generate_size
            tokens = torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(0)
            tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)
            return tokens

        prompt_size = 64
        vocab_size = 64
        batch_size = 4
        src_tokens = torch.randint(1, vocab_size, (batch_size, prompt_size))
        gm = make_fx(f, tracing_mode="symbolic")(src_tokens)
        # Guards to rule out batch_size == sys.maxsize (wobbling between 2 and
        # 1 ok)
        self.assertEqual(len(gm.shape_env.guards), 0)

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_cpu_scalar_cuda(self):
        # Extracted from wave2vec2
        def f(a, b):
            return (a * b) @ b

        r = str(
            make_fx(f, tracing_mode="symbolic")(
                torch.tensor(1.0), torch.randn(2, 2, device='cuda')
            ).code
        ).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1, b_1):
    mul = torch.ops.aten.mul.Tensor(a_1, b_1);  a_1 = None
    mm = torch.ops.aten.mm.default(mul, b_1);  mul = b_1 = None
    return mm""")

    def test_binary_broadcast(self):
        def f(a, b):
            c = a * b
            return c

        test_inputs = []
        test_inputs.append([(1, 5), (3, 1)])
        test_inputs.append([(1, 4), (4, 1)])
        shape_env = self._test_dynamic(f, [(1, 2), (3, 1)], test_inputs).shape_env
        assert len(shape_env.guards) == 0

    def test_multiply_shape(self):
        def f(a):
            return torch.empty(a.shape[0] * 2)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
    mul = sym_size_int * 2;  sym_size_int = None
    empty = torch.ops.aten.empty.memory_format([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    return empty""")

    def test_item(self):
        def f(a):
            r = a.item()
            return r * a

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(1)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1)
    mul = torch.ops.aten.mul.Tensor(a_1, _local_scalar_dense);  a_1 = _local_scalar_dense = None
    return mul""")

    def test_tensor_symfloat(self):
        def f(a):
            r = torch.tensor(a.size(0) ** 2.0)
            assert r.dtype is torch.float
            return r

        gm = make_fx(f, tracing_mode="symbolic")(torch.randn(2))
        r = str(gm.code).strip()
        # NB: this specializes, which is fine, the point is to make sure the
        # dtype inference is correct
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    return lift_fresh_copy""")
        self.assertEqual(gm._tensor_constant0, torch.tensor(4.0))

    def test_item_to_constructor(self):
        def f(a):
            r = a.item()
            return torch.empty(r)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randint(5, (1,))).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, a_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(a_1);  a_1 = None
    empty = torch.ops.aten.empty.memory_format([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return empty"""  # noqa: B950
        )


    def test_setitem_symint(self):
        # from moco
        # https://github.com/pytorch/pytorch/issues/101939
        def f(x):
            x[0] = x.size(0)
            return x

        r = str(make_fx(f, tracing_mode="symbolic")(torch.randn(10)).code).strip()
        self.assertExpectedInline(
            r, """\
def forward(self, x_1):
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    scalar_tensor = torch.ops.aten.scalar_tensor.default(sym_size_int, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  sym_size_int = None
    select = torch.ops.aten.select.int(x_1, 0, 0)
    copy_ = torch.ops.aten.copy_.default(select, scalar_tensor);  select = scalar_tensor = copy_ = None
    return x_1"""  # noqa: B950
        )

    def test_dynamic_pointwise_scalar(self):
        def f(gravity, mask):
            gravity[mask, 0] = gravity[mask, 0] * -1

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.randn((12, 4)),
            torch.randint(0, 2, (12,), dtype=torch.bool)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, gravity_1, mask_1):
    select = torch.ops.aten.select.int(gravity_1, 1, 0)
    index = torch.ops.aten.index.Tensor(select, [mask_1]);  select = None
    mul = torch.ops.aten.mul.Tensor(index, -1);  index = None
    select_1 = torch.ops.aten.select.int(gravity_1, 1, 0);  gravity_1 = None
    index_put_ = torch.ops.aten.index_put_.default(select_1, [mask_1], mul);  select_1 = mask_1 = mul = index_put_ = None
    return None""")

    def test_reflect_r_over_x(self):
        def reflect_R_over_x(R):
            reflect = torch.eye(3, device=R.device)
            reflect[0, 0] = -1
            return reflect @ R @ reflect

        def f(crop_camera, mask):
            crop_camera[mask] = reflect_R_over_x(crop_camera[mask])

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.randn((12, 3, 3)),
            torch.randint(0, 2, (12,), dtype=torch.bool)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, crop_camera_1, mask_1):
    index = torch.ops.aten.index.Tensor(crop_camera_1, [mask_1])
    eye = torch.ops.aten.eye.default(3, device = device(type='cpu'), pin_memory = False)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    select = torch.ops.aten.select.int(eye, 0, 0)
    select_1 = torch.ops.aten.select.int(select, 0, 0);  select = None
    copy_ = torch.ops.aten.copy_.default(select_1, lift_fresh_copy);  select_1 = lift_fresh_copy = copy_ = None
    sym_size_int = torch.ops.aten.sym_size.int(index, 0)
    expand = torch.ops.aten.expand.default(eye, [sym_size_int, 3, 3])
    view = torch.ops.aten.view.default(expand, [sym_size_int, 3, 3]);  expand = None
    sym_size_int_1 = torch.ops.aten.sym_size.int(crop_camera_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(crop_camera_1, 2)
    expand_1 = torch.ops.aten.expand.default(index, [sym_size_int, sym_size_int_1, sym_size_int_2]);  index = None
    view_1 = torch.ops.aten.view.default(expand_1, [sym_size_int, sym_size_int_1, sym_size_int_2]);  expand_1 = sym_size_int_1 = sym_size_int_2 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [sym_size_int, 3, 3]);  bmm = None
    mul_4 = sym_size_int * 3
    view_3 = torch.ops.aten.view.default(view_2, [mul_4, 3]);  view_2 = mul_4 = None
    mm = torch.ops.aten.mm.default(view_3, eye);  view_3 = eye = None
    _unsafe_view = torch.ops.aten._unsafe_view.default(mm, [sym_size_int, 3, 3]);  mm = sym_size_int = None
    index_put_ = torch.ops.aten.index_put_.default(crop_camera_1, [mask_1], _unsafe_view);  crop_camera_1 = mask_1 = _unsafe_view = index_put_ = None
    return None""")  # noqa: B950

    def test_unbacked_slice(self):
        def f(x, m):
            x = x[m]
            return x[slice(None, None, None), slice(None, None, None), slice(None, 2, None)]

        make_fx(f, tracing_mode="symbolic")(
            torch.randn((12, 3, 3)),
            torch.randint(0, 2, (12,), dtype=torch.bool)
        )

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_unbacked_batch_resnet(self):
        mod = torchvision.models.resnet18()

        def f(x, mask, params, buffers):
            for p in itertools.chain([x, mask], params.values(), buffers.values()):
                for s in p.shape:
                    guard_int(s)
            x = x[mask]
            torch._check(x.shape[0] >= 1)
            for p in params.values():
                p.grad = None
            return torch.func.functional_call(mod, {**params, **buffers}, (x,)).sum()

        make_fx(f, tracing_mode="symbolic")(
            torch.randn(3, 3, 250, 250),
            torch.randint(0, 2, (3,), dtype=torch.bool),
            dict(mod.named_parameters()),
            dict(mod.named_buffers()),
        )

    def test_boolean_index(self):
        def f(images, handedness, valid):
            images = images[valid]
            handedness = handedness[valid]
            right_hand_mask = handedness == 1
            images[right_hand_mask] = images[right_hand_mask].flip(-1)

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.randint(0, 256, (512, 1, 96, 96)),
            torch.randint(0, 1, (512,)),
            torch.randint(0, 2, (512,), dtype=torch.bool)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, images_1, handedness_1, valid_1):
    index = torch.ops.aten.index.Tensor(images_1, [valid_1]);  images_1 = None
    index_1 = torch.ops.aten.index.Tensor(handedness_1, [valid_1]);  handedness_1 = valid_1 = None
    eq = torch.ops.aten.eq.Scalar(index_1, 1);  index_1 = None
    index_2 = torch.ops.aten.index.Tensor(index, [eq])
    flip = torch.ops.aten.flip.default(index_2, [-1]);  index_2 = None
    index_put_ = torch.ops.aten.index_put_.default(index, [eq], flip);  index = eq = flip = index_put_ = None
    return None""")

    def test_neg_shape(self):
        def f(a):
            return torch.empty(-a.shape[0] + 10)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(2)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0);  a_1 = None
    neg = -sym_size_int;  sym_size_int = None
    add = neg + 10;  neg = None
    empty = torch.ops.aten.empty.memory_format([add], device = device(type='cpu'), pin_memory = False);  add = None
    return empty""")

    def test_unbacked_unification(self):
        def f(x, y):
            z = torch.zeros(x.item())
            return z + y

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10), torch.randn(10)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    add = torch.ops.aten.add.Tensor(zeros, y_1);  zeros = y_1 = None
    return add""")  # noqa: B950

    def test_reshape_divisibility_unbacked(self):
        def f(x):
            i0 = x.item()
            r = torch.zeros(i0, 4, 20)
            r = r.transpose(2, 1)
            return r.reshape(-1, 80)
        make_fx(f, tracing_mode="symbolic")(torch.tensor(24))

    def test_view_divisibility_unbacked(self):
        def f(x):
            i0 = x.item()
            r = torch.zeros(i0, 192)
            return r.view(12, -1, 192)
        make_fx(f, tracing_mode="symbolic")(torch.tensor(24))

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_view_divisibility_unbacked_relatively_prime(self):
        # See https://github.com/pytorch/pytorch/issues/123651
        def f(x):
            i0 = x.item()
            torch._check_is_size(i0)
            # To trigger the original issue, the max bound has to
            # be chosen such that 448 / 447 < 2 (which it is.)
            torch._check(i0 <= 448)
            return torch.zeros(256 * i0).view(-1, 447)
        make_fx(f, tracing_mode="symbolic")(torch.tensor(256 * 447, device="cuda"))

    def test_unbacked_unify_guard(self):
        def f(x, y):
            z = torch.zeros(x.item())
            torch._check(z.size(0) == y.size(0))  # refines i0 = s0
            if z.size(0) == 4:
                return y * 2
            else:
                return y + 2

        r = str(make_fx(f, tracing_mode="symbolic")(torch.tensor(10), torch.randn(10)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, x_1, y_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = zeros = None
    add = torch.ops.aten.add.Tensor(y_1, 2);  y_1 = None
    return add""")  # noqa: B950

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    @unittest.expectedFailure
    def test_unbacked_unify_guard_transitivity(self):
        def f(x1, x2, y):
            z1 = torch.zeros(x1.item())
            z2 = torch.zeros(x2.item())
            torch._check(z1.size(0) == z2.size(0))  # refines i0 = i1
            torch._check(z2.size(0) == y.size(0))  # refines i0 = s0
            if z1.size(0) == 4:
                return y * 2
            else:
                return y + 2

        gm = make_fx(f, tracing_mode="symbolic")(
            torch.tensor(10, device="cuda"),
            torch.tensor(10, device="cuda"),
            torch.randn(10, device="cuda")
        )
        insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
        gm.recompile()
        r = str(gm.code).strip()
        # self.assertExpectedInline(
        #     r, """"""  # noqa: B950
        # )

    @unittest.skipIf(not HAS_CUDA, 'CUDA-only test')
    def test_unbacked_unify_dependency_violation(self):
        def f(x1, x2, x3, y):
            z1 = x1.item()
            torch._check(z1 // 9 == 1)
            z2 = x2.item()
            z3 = x3.item()
            torch._check(z1 == z2 + z3)
            return y * 2
        # NB: inputs are done as CUDA to ensure they aren't queried to be
        # backed

        gm = make_fx(f, tracing_mode="symbolic")(
            torch.tensor(10, device="cuda"), torch.tensor(5, device="cuda"),
            torch.tensor(5, device="cuda"), torch.randn(1, device="cuda")
        )
        insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
        gm.recompile()
        self.assertEqual(gm(
            torch.tensor(12, device="cuda"), torch.tensor(6, device="cuda"),
            torch.tensor(6, device="cuda"), torch.tensor([1.0], device="cuda")),
            torch.tensor([2.0], device="cuda")
        )
        with self.assertRaises(RuntimeError):
            gm(
                torch.tensor(20, device="cuda"), torch.tensor(10, device="cuda"),
                torch.tensor(10, device="cuda"), torch.tensor([1.0], device="cuda")
            )


    def test_split_unbacked_sizes(self):
        def f(lengths, values):
            # tolist not directly supported atm
            sizes = [lengths[i].item() for i in range(lengths.size(0))]
            for s in sizes:
                # TODO(avik): no assertion generated with torch._check_is_size?
                torch._constrain_as_size(s)
            return torch.split(values, sizes)

        r = str(make_fx(f, tracing_mode="symbolic")(
            torch.tensor([2, 3, 4]),
            torch.randn(9)
        ).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, lengths_1, values_1):
    select = torch.ops.aten.select.int(lengths_1, 0, 0)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(select);  select = None
    select_1 = torch.ops.aten.select.int(lengths_1, 0, 1)
    _local_scalar_dense_1 = torch.ops.aten._local_scalar_dense.default(select_1);  select_1 = None
    select_2 = torch.ops.aten.select.int(lengths_1, 0, 2);  lengths_1 = None
    _local_scalar_dense_2 = torch.ops.aten._local_scalar_dense.default(select_2);  select_2 = None
    sym_constrain_range_for_size = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense);  sym_constrain_range_for_size = None
    sym_constrain_range_for_size_1 = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense_1);  sym_constrain_range_for_size_1 = None
    sym_constrain_range_for_size_2 = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense_2);  sym_constrain_range_for_size_2 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(values_1, [_local_scalar_dense, _local_scalar_dense_1, _local_scalar_dense_2]);  values_1 = _local_scalar_dense = _local_scalar_dense_1 = _local_scalar_dense_2 = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    return (getitem, getitem_1, getitem_2)""")  # noqa: B950

    def test_invalidate_nonzero(self):
        ok = False

        def f(a):
            nonlocal ok
            b = a.clone()
            x = b.nonzero()
            x1 = b.nonzero()
            x2 = b.nonzero()
            assert x1.shape[0] == x2.shape[0]
            ok = True
            b.normal_()
            y = b.nonzero()
            try:
                bool(x1.shape[0] == y.shape[0])
                self.fail("didn't raise exception")
            except GuardOnDataDependentSymNode:
                pass

        make_fx(f, tracing_mode="symbolic")(torch.randn(4))

    @torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True)
    def test_invalidate_nonzero_propagate_real_tensors(self):
        def f(a):
            b = a.clone()
            x = b.nonzero()
            x1 = b.nonzero()
            x2 = b.nonzero()
            assert x1.shape[0] == x2.shape[0]
            b.normal_()
            y = b.nonzero()
            # Because you're not actually going to generate exactly zero with
            # normal_ lol
            assert x1.shape[0] == y.shape[0]

        make_fx(f, tracing_mode="symbolic")(torch.randn(4))

    def test_sqrt_size(self):
        def f(a):
            return a / a.size(-1) ** 0.5

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_float = torch.sym_float(sym_size_int);  sym_size_int = None
    pow_1 = sym_float ** 0.5;  sym_float = None
    div = torch.ops.aten.div.Tensor(a_1, pow_1);  a_1 = pow_1 = None
    return div""")

    def test_make_fx_with_custom_tracer_preserving_nn_module_stack(self):

        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                return x + 1

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                return x + self.bar(x)

        gm = make_fx(Foo())(torch.randn(4, 4))
        for node in gm.graph.nodes:
            self.assertTrue("nn_module_stack" not in node.meta)

        foo = Foo()

        def functional_call(*args, **kwargs):
            with stateless._reparametrize_module(foo, {}):
                return foo(*args, **kwargs)

        functional_call._orig_mod = foo

        gm_with_stack = make_fx(functional_call, record_module_stack=True)(torch.randn(4, 4))
        found = False
        for node in gm_with_stack.graph.nodes:
            if "nn_module_stack" in node.meta:
                if len(node.meta["nn_module_stack"]) == 1:
                    self.assertTrue("custom_tracer_preserving_nn_module_stack.<locals>.Foo" in str(node.meta["nn_module_stack"]))
                    found = True
                elif len(node.meta["nn_module_stack"]) == 2:
                    self.assertTrue("preserving_nn_module_stack.<locals>.Bar" in str(node.meta["nn_module_stack"]))
                    found = True
                else:
                    # there can be at most 2 level
                    self.assertTrue(False)

        self.assertTrue(found)

        gm_without_stack = make_fx(functional_call)(torch.randn(4, 4))
        for node in gm_without_stack.graph.nodes:
            self.assertTrue("nn_module_stack" not in node.meta)

    def test_symint_to_tensor(self):
        def f(a):
            return a / a.shape[0]

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    div = torch.ops.aten.div.Tensor(a_1, sym_size_int);  a_1 = sym_size_int = None
    return div""")

        r = str(make_fx(f, tracing_mode="symbolic", decomposition_table=decomposition_table)(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size_int = torch.ops.aten.sym_size.int(a_1, 0)
    sym_float = torch.sym_float(sym_size_int);  sym_size_int = None
    div = torch.ops.prims.div.default(a_1, sym_float);  a_1 = sym_float = None
    return div""")

    def test_cat(self):
        def f(a, b):
            val = torch.mul(a, b)
            out = torch.cat([val, val])
            if out.shape[0] * out.shape[1] > 20:
                out = out.cos()
            return out

        test_inputs = []
        test_inputs.append([(1, 5), (6, 1)])
        test_inputs.append([(1, 4), (3, 1)])
        gm = self._test_dynamic(f, [(1, 6), (8, 1)], test_inputs)
        self.assertTrue(eval_guards(gm, torch.randn(1, 10), torch.randn(6, 1)))
        self.assertFalse(eval_guards(gm, torch.randn(1, 2), torch.randn(4, 1)))
        self.assertExpectedInline(show_guards(gm), """2*L['b'].size()[0]*L['a'].size()[1] > 20""")

    def test_new_empty(self):
        def f(a, b):
            return a.new_empty(b.shape[0], b.shape[1] * 2)

        self._test_dynamic(f, [(2, 4), (4, 5)], [[(2, 3), (5, 7)], [(3, 7), (9, 3)]], assert_eq=False).shape_env

    def test_size_with_tensor(self):
        # I think I messed up writing this test case originally, I think
        # I'm supposed to hit an error case, but the code here works in both
        # eager and tracing
        def f(tensor):
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [2] + list(tensor.shape[:-2]) + list(max_size)
            return tensor.new_empty(batch_shape)

        a = torch.randn(3, 800, 1199)
        f(a)
        make_fx(f, tracing_mode="symbolic")(a)

    def test_fake_tensor_as_size(self):
        def f(x):
            r = torch.zeros([x])
            return r

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.tensor(4))
        self.assertExpectedInline(fx_g.code.strip(), """\
def forward(self, x_1):
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(x_1);  x_1 = None
    zeros = torch.ops.aten.zeros.default([_local_scalar_dense], device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return zeros""")  # noqa: B950

    def test_expand(self):
        def f(a):
            b = torch.mul(a, a)
            c = b.expand(a.shape)
            return c

        self._test_dynamic(f, [(3,)], [[(3,)], [(4,)], [(2,)]])
        self._test_dynamic(f, [(5, 1)], [[(4, 1)], [(3, 1)], [(6, 1)]])

    def test_metadata(self):
        def f(a, b):
            d = a.new_empty(a.shape[0] + b.shape[0])
            return d
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(5), torch.randn(4))
        meta_c = _get_node(fx_g, lambda x: x.target == aten.new_empty.default)
        meta_d = _get_node(fx_g, lambda x: x.target == operator.add)
        self.assertTrue(meta_c.meta['val'].shape[0].node.expr == meta_d.meta['val'].node.expr)

    def test_metadata_fresh(self):
        def f(x):
            assert x.shape[0] == 3
            return x.cos()

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(3))
        meta_cos = _get_node(fx_g, lambda x: x.target == aten.cos.default)
        meta_inp = _get_node(fx_g, lambda x: x.op == 'placeholder')
        self.assertTrue(meta_cos.meta['val'].shape[0] == 3)
        # Checks if the input expr has been updated even though the constraint
        # happened afterwards
        self.assertTrue(meta_inp.meta['val'].shape[0] == 3)

    def test_elementwise_meta_with_sym_numbers(self):
        def f(x, offset, as_sym_float=False):
            x0 = x.size()[0]
            if as_sym_float:
                x0 = torch.sym_float(x0)
            return torch.add(x0, offset)

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2.0, False)
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        self.assertEqual(meta_add.meta['val'].shape, ())
        self.assertEqual(meta_add.meta['val'].dtype, torch.float32)

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2, False)
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        self.assertEqual(meta_add.meta['val'].shape, ())
        self.assertEqual(meta_add.meta['val'].dtype, torch.int64)

        fx_g = make_fx(f, tracing_mode="symbolic")(torch.rand(2, 3), 2, True)
        meta_add = _get_node(fx_g, lambda x: x.target == aten.add.Tensor)
        self.assertEqual(meta_add.meta['val'].shape, ())
        self.assertEqual(meta_add.meta['val'].dtype, torch.float32)

    def test_return_symint(self):
        def f(x):
            return x.shape[0], x.cos(), x.shape[0] / 5
        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

        def f(x):
            return x.shape
        self._test_dynamic(f, [(5, 3)], [[(4, 6)]])

    def test_rmethod(self):
        def f(x):
            return x.size(0) + x
        self._test_dynamic(f, [(5,)], [[(4,)], [(12,)]])

    def test_mega_guard(self):
        def f(a, b):
            assert a.shape[0] == b.shape[0] * 2
            return a.cos()
        fx_g = make_fx(f, tracing_mode="symbolic")(torch.randn(16), torch.randn(8))
        from torch._dynamo.source import LocalSource
        self.assertExpectedInline(
            str(fx_g.shape_env.produce_guards(fx_placeholder_vals(fx_g), [LocalSource("a"), LocalSource("b")], ignore_static=False)),  # noqa: B950
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "L['a'].stride()[0] == 1", "L['a'].storage_offset() == 0", "L['b'].stride()[0] == 1", "L['b'].storage_offset() == 0", "2 <= L['b'].size()[0]"]"""  # noqa: B950
        )
        self.assertExpectedInline(
            str(fx_g.shape_env.produce_guards(fx_placeholder_vals(fx_g), [LocalSource("a"), LocalSource("b")], ignore_static=True)),  # noqa: B950
            """["L['a'].size()[0] == 2*L['b'].size()[0]", "2 <= L['b'].size()[0]"]"""  # noqa: B950
        )

    def test_guard_upperbound_range_refinement(self):
        def f(a):
            assert a.shape[0] > 5 and a.shape[0] > 12
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(15))
        self.assertExpectedInline(show_guards(tensor), """13 <= L['a'].size()[0]""")

    def test_guard_lowerbound_range_refinement(self):
        def f(a):
            assert a.shape[0] < 20 and a.shape[0] < 30
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(15))
        self.assertExpectedInline(show_guards(tensor), """L['a'].size()[0] <= 19""")

    def test_guard_upperbound_range_refinement_multivariate(self):
        def f(a):
            assert a.shape[0] > 5 and a.shape[0] > 12
            assert a.shape[1] > 5 and a.shape[1] > a.shape[0]
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn((15, 20)))
        self.assertExpectedInline(show_guards(tensor), """\
L['a'].size()[1] > L['a'].size()[0]
13 <= L['a'].size()[0]
14 <= L['a'].size()[1]""")

    def test_guard_lowerbound_range_refinement_multivariate(self):
        def f(a):
            assert a.shape[0] < 20 and a.shape[0] < 30
            assert a.shape[1] < 30 and a.shape[1] < a.shape[0]
            return a.cos()
        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn((15, 5)))
        self.assertExpectedInline(
            show_guards(tensor),
            """\
L['a'].size()[1] < L['a'].size()[0]
L['a'].size()[0] <= 19
L['a'].size()[1] <= 18""")

    def test_sym_storage_offset(self):
        def f(x, y):
            return x + y

        inp = (torch.randn(8)[3:], torch.randn(5))
        fx_g = make_fx(f, tracing_mode="symbolic")(*inp)
        inp = (torch.randn(8)[3:], torch.randn(5))
        self.assertEqual(fx_g(*inp), f(*inp))

    def _assert_no_guards(self, fx_g, free_symbols):
        assert _get_free_symbols(fx_g.shape_env) == free_symbols, fx_g.shape_env.var_to_val
        assert len(fx_g.shape_env.get_nontrivial_guards()) == 0, fx_g.shape_env.format_guards()

    def test_guards_equal(self):
        def f(a, b):
            return a * b

        # NB: Numbers are carefully chosen to avoid duck shaping from applying

        fx_g = _trace(f, (5, 6), (5, 6))
        self._assert_no_guards(fx_g, 2)

        fx_g = _trace(f, (5, 6, 7), (5, 6, 7))
        self._assert_no_guards(fx_g, 3)

        fx_g = _trace(f, (5, 1), (1, 6))
        self._assert_no_guards(fx_g, 2)

        def f(a, b, c, d):
            a = a + b
            cat = torch.cat([c, d])
            return a + cat

        fx_g = _trace(f, 7, 7, 4, 3)
        self._assert_no_guards(fx_g, 2)

        def f(a, b, c, d, e):
            vals = [a, b, c, d, e]
            x = a
            for idx in range(len(vals) - 1):
                x = torch.cat([x, vals[idx]]) + vals[idx + 1]
            return x

        fx_g = _trace(f, 2, 4, 8, 16, 32)
        self._assert_no_guards(fx_g, 1)

        def f(a, b):
            a = a.view(b.shape[0])
            return a + b.sum()

        fx_g = _trace(f, (4, 2), 8)
        self._assert_no_guards(fx_g, 2)

        fx_g = _trace(f, (4, 2), (8, 5))
        self._assert_no_guards(fx_g, 3)

        fx_g = _trace(f, (2, 3, 4), 24)
        self._assert_no_guards(fx_g, 3)

    def test_nonidentity_transitive_guards(self):
        def f(a, b, c, d, e):
            vals = [a, b, c, d, e]
            cat_vals = []
            for idx in range(len(vals) - 1):
                cat_vals.append(torch.cat([vals[idx], vals[idx]]))
            final_vals = []
            for a, b in reversed(list(zip(cat_vals, vals[1:]))):
                final_vals.append(a + b)
            return final_vals

        fx_g = _trace(f, 2, 4, 8, 16, 32)
        self.assertExpectedInline(show_guards(fx_g), """""")

    @torch.fx.experimental._config.patch(translation_validation=True)
    def test_constant_specialization(self):
        def f(t):
            assert t.shape[0] == 10
            return t

        tensor = make_fx(f, tracing_mode="symbolic")(torch.randn(10))
        self.assertExpectedInline(show_guards(tensor), """""")


make_fx_failures = {
    # unknown
    xfail('allclose'),
    xfail('equal'),
    # empty
    skip('new_empty'),
    skip('empty_like'),
    skip('empty'),
    skip('empty_permuted'),
    # flaky
    skip('linalg.lstsq', 'grad_oriented'),
    skip('nn.functional.max_unpool1d', '', device_type='cpu'),
    skip('nn.functional.max_unpool2d', '', device_type='cpu'),
    skip('nn.functional.max_unpool3d', '', device_type='cpu'),
    skip('linalg.lstsq'),  # flaky, probably just a precision issue

    # data-dependent control flow
    skip('item'),
    xfail('cov'),
    xfail('nn.functional.gaussian_nll_loss'),
    xfail('tensor_split'),
    xfail('corrcoef'),
    xfail('quantile'),
    xfail('nanquantile'),

    # Seems like it's creating a sparse tensor that isn't captured by tensor.is_sparse
    xfail('sparse.sampled_addmm'),
    xfail('sparse.mm', 'reduce'),

    # proxy tensor doesn't support sparse correctly right now
    skip('to_sparse'),
    # segfaults
    skip('block_diag'),

    # AssertionError: Tensor-likes are not close!
    skip('empty_strided', '', device_type='cpu'),
}

only_real_tensor_failures = {
    xfail('narrow'),
}

only_fake_tensor_failures = {
    xfail('narrow'),
}

fake_tensor_failures = set()

symbolic_tensor_failures = {
    xfail('combinations', ''),
    xfail('geqrf', ''),  # aten.geqrf.default - couldn't find symbolic meta function/decomposition
    xfail('histogram', ''),  # Could not run 'aten::histogram.bin_ct' with arguments from the 'Meta' backend. This c...
    xfail('histogramdd', ''),  # aten._histogramdd_bin_edges.default - couldn't find symbolic meta function/decomposition
    xfail('nanquantile', ''),  # Could not run 'aten::equal' with arguments from the 'Meta' backend.
    xfail('nn.functional.binary_cross_entropy', ''),  # aten.new_empty.default - couldn't find symbolic meta function/decom...
    xfail('nn.functional.cross_entropy', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.ctc_loss'),  # aten._ctc_loss.Tensor - couldn't find symbolic meta function/decomposition
    xfail('quantile', ''),  # Could not run 'aten::equal' with arguments from the 'Meta' backend.

    xfail('max_pool2d_with_indices_backward', ''),  # Expected a value of type 'List[int]' for argument 'kernel_size' but...
}
symbolic_tensor_segfaults = {
    skip('nn.functional.batch_norm')  # Segfault??
}

symbolic_tensor_failures.update(symbolic_tensor_segfaults)

inplace_symbolic_tensor_failures = {
    # bugs
    xfail('float_power', ''),  # base given to float_power_ has dtype Float but the operation's result requires dtype Double
}

out_symbolic_tensor_failures = {
    # Cast error details: Unable to cast (...) to Tensor
    #
    # This happens because the test is set up to call the out variant using the `out` kwarg:
    #   torch._some_op(arg1, arg2, out=(out1, out2, out3))
    #
    # However, this only works on torch ops, not aten ops. For `_batch_norm_with_update`,
    # this fails because the op has no python bindings, so it doesn't support the `out` kwarg
    # way of calling its out variant.
    xfail('_batch_norm_with_update', ''),
    xfail('_native_batch_norm_legit', ''),
    xfail('angle', ''),
    xfail('argmax', ''),
    xfail('argmin', ''),
    xfail('gather', ''),
    xfail('linalg.pinv', ''),
    xfail('linalg.pinv', 'hermitian'),
    xfail('scatter_add', ''),
    xfail('scatter', ''),
    xfail('take_along_dim', ''),

    # SymIntArrayRef expected to contain only concrete
    xfail('randn', ''),

    # RuntimeError: Cannot call numel() on tensor with symbolic sizes/strides
    xfail('index_reduce', 'prod'),
    xfail('index_reduce', 'mean'),
    xfail('index_reduce', 'amax'),
    xfail('index_reduce', 'amin'),
}

out_symbolic_tensor_segfaults = {
    skip('nanmean', ''),
}

out_symbolic_tensor_failures.update(out_symbolic_tensor_segfaults)

# Copies inputs to inplace operations to avoid inplace modifications
#   to leaves requiring gradient
def _get_safe_inplace(inplace_variant):
    @functools.wraps(inplace_variant)
    def _fn(t, *args, **kwargs):
        return inplace_variant(t.clone(), *args, **kwargs)

    return _fn

def _test_make_fx_helper(self, device, dtype, op, tracing_mode, inplace=False, out=False):
    fn = _get_safe_inplace(op.get_inplace()) if inplace else op.op
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    # Limit ourselves to first 100 inputs so symbolic tracing tests don't take too long
    count = 100
    if out:
        count = 5
    for sample_input in itertools.islice(sample_inputs_itr, count):
        if inplace and sample_input.broadcasts_input:
            continue
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs
        if out:
            expected = fn(*args, **kwargs)
            kwargs['out'] = expected

        try:
            optests.make_fx_check(fn, args, kwargs, tracing_mode, self.assertEqual,
                                  randomize_data=True)
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")


def skipIfNameMatches(pattern):
    """
    Decorator to skip a test if its name matches the given pattern.
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if re.match(pattern, test_func.__name__):
                raise unittest.SkipTest(f"Test '{test_func.__name__}' skipped because its name matches the pattern '{pattern}'")
            return test_func(*args, **kwargs)
        return wrapper
    return decorator

# Auto functionalize shouldn't work with make_fx directly
filtered_hop_db = [op for op in hop_db if op.name != "auto_functionalize"]

@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "Cond requires dynamo")
class TestProxyTensorOpInfo(TestCase):
    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_exhaustive', make_fx_failures.union(only_real_tensor_failures))
    def test_make_fx_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "real")

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_fake_exhaustive',
             make_fx_failures.union(fake_tensor_failures, only_fake_tensor_failures))
    def test_make_fx_fake_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "fake")

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive',
             make_fx_failures | fake_tensor_failures | symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "symbolic")

    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive_inplace',
             make_fx_failures | fake_tensor_failures | symbolic_tensor_failures | inplace_symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive_inplace(self, device, dtype, op):
        if not op.get_inplace():
            self.skipTest("No inplace variable for this op")
        _test_make_fx_helper(self, device, dtype, op, "symbolic", inplace=True)

    @ops(op_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive_out',
             make_fx_failures | fake_tensor_failures | symbolic_tensor_failures | out_symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive_out(self, device, dtype, op):
        if not op.supports_out:
            self.skipTest("Op doesn't support out")
        _test_make_fx_helper(self, device, dtype, op, "symbolic", out=True)


only_for = ("cpu")
instantiate_device_type_tests(TestProxyTensorOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
