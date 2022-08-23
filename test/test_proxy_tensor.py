# Owner(s): ["module: ProxyTensor"]

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import unittest
import warnings
import torch.nn.utils._stateless as stateless
from collections.abc import Iterable
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_methods_invocations import op_db, wrapper_set_seed
from torch._subclasses.fake_tensor import DynamicOutputShapeException

from torch._decomp import decomposition_table
from torch.testing._internal.common_device_type import ops
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import make_fx, DecompositionInterpreter, get_isolated_graphmodule
from torch.utils._pytree import tree_map
from torch import nn
import re

import types
import functools

aten = torch.ops.aten

try:
    import sympy  # noqa: F401
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
skipIfNoSympy = unittest.skipIf(not HAS_SYMPY, "no sympy")


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


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


# Copied from functorch
def xfail(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)


def skip(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)


def skipOps(test_case_name, base_test_name, to_skip):
    all_opinfos = op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [o for o in all_opinfos
                            if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(unittest.expectedFailure,
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(unittest.skip("Skipped!"),
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn
    return wrapped


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
        # into the outer graph
        def f2(x):
            gm = make_fx(f1)(x)
            self.assertFalse(is_any_sum(gm))
            self.assertTrue(is_any_sigmoid(gm))
            return torch.digamma(x)

        traced = make_fx(f2)(torch.randn(3))
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
    copy__default = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    return copy__default
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
        # because free fake tensors are not supported.  Fortunately stateless
        # does precisely this for us.
        def f(x, params, buffers):
            for p in params.values():
                p.grad = None
            loss = stateless.functional_call(mod, {**params, **buffers}, (x,)).sum()
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

    def test_make_fx_overloads(self):
        def f(x):
            return x.cos() + torch.randn(x.shape)

        traced = make_fx(f, tracing_mode=self.tracing_mode)(torch.randn(3))

        self.assertTrue(all([isinstance(node.target, torch._ops.OpOverload)
                             for node in traced.graph.nodes if node.op == 'call_function']))

    def test_tensor_constants(self):
        def f():
            val = torch.tensor(float('inf'))
            return torch.full((100, 100), val)

        self._test(f, [])

    def test_allclose(self):
        def f(a, b):
            return torch.allclose(a, b)

        self.assertRaisesRegex(
            RuntimeError, "data-dependent",
            lambda: make_fx(f, tracing_mode=self.tracing_mode)(
                torch.zeros(3), torch.zeros(3)
            )
        )

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
            return blowup.sum().item()

        self.assertRaisesRegex(
            RuntimeError, "data-dependent",
            lambda: make_fx(f, tracing_mode=self.tracing_mode)()
        )

    def test_constant_random(self):
        def f():
            val = torch.tensor([2.0])
            val.normal_()
            return val.item()

        self.assertRaisesRegex(
            RuntimeError, "data-dependent",
            lambda: make_fx(f, tracing_mode=self.tracing_mode)()
        )

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
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.linear(x).relu()

        model = Foo()

        def f(x, params):
            out = stateless.functional_call(model, params, x).sum()
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

    def test_make_fx_model_fwd_bwd_wgtupdate(self):
        class Foo(torch.nn.Module):
            def __init__(self):
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
            out = stateless.functional_call(model, params_and_buffers, args)
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
        def f(x):
            x = UnwrapTensor(x)
            y = x * 2
            return y

        inp = [torch.randn(5)]
        self._test(f, inp)

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

        decomposed_fx = make_fx(f, {aten.addmm.default: addmm})(*inps)

        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        self.assertEqual(len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2)
        self.assertEqual(len([n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]), 1)




class TestGenericProxyTensorReal(TestGenericProxyTensor):
    tracing_mode = "real"


class TestGenericProxyTensorFake(TestGenericProxyTensor):
    tracing_mode = "fake"


def xfail_inherited_tests(tests):
    """
    Given a list of test names which are defined by a superclass of the
    class this decorates, mark them as expected failure.  This is useful
    if you are doing poor man's parameterized tests by subclassing a generic
    test class.
    """
    def deco(cls):
        for t in tests:
            # NB: expectedFailure operates by mutating the method in question,
            # which is why you have to copy the function first
            setattr(cls, t, unittest.expectedFailure(copy_func(getattr(cls, t))))
        return cls
    return deco


@skipIfNoSympy
@xfail_inherited_tests([
    "test_inplace_metadata",
    "test_mode_tracing_factory_function",
    "test_make_fx_overloads",
    "test_make_fx_model_fwd_bwd_wgtupdate",
    "test_make_fx_model_fwd_bwd",
    "test_proxy_tensor",
    "test_resnet18_backward_trace",
    "test_trace_subclasses",
])
class TestGenericProxyTensorSymbolic(TestGenericProxyTensor):
    tracing_mode = "symbolic"


del TestGenericProxyTensor


class TestRealProxyTensor(TestCase):
    pass

class TestFakeProxyTensor(TestCase):
    def test_issue82547(self):
        x = nn.Parameter(torch.randn(3, 3))

        def f():
            return torch.ops.aten.t.default(x)
        self.assertRaisesRegex(Exception, "non-Fake Tensor", lambda: make_fx(f, tracing_mode="fake")())

        class A(torch.Tensor):
            pass

        x = A(torch.randn(3, 3))
        self.assertRaisesRegex(TypeError, "no implementation found", lambda: make_fx(f, tracing_mode="fake")())

    def test_use_fake_and_tensor(self):
        def f(x, y):
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        g = make_fx(f, tracing_mode="fake")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        self.assertEqual(g(x, y), f(x, y))

# TODO: Need to test the guards themselves specifically as well
@skipIfNoSympy
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
        return traced_f.shape_env


    def test_unary(self):
        def f(x):
            assert x.shape[0] < 20
            return x.cos()
        test_inputs = []
        test_inputs.append([(2, 5)])
        test_inputs.append([(6, 8)])
        shape_env = self._test_dynamic(f, [(3, 4)], test_inputs)
        self.assertTrue(shape_env.evaluate_guards_for_args(torch.randn(4, 5)))
        self.assertFalse(shape_env.evaluate_guards_for_args(torch.randn(25, 5)))
        assert len(shape_env.guards) == 1

    def test_binary_broadcast(self):
        def f(a, b):
            c = a * b
            return c

        test_inputs = []
        test_inputs.append([(1, 5), (3, 1)])
        test_inputs.append([(1, 4), (4, 1)])
        shape_env = self._test_dynamic(f, [(1, 2), (3, 1)], test_inputs)
        assert len(shape_env.guards) == 0

    def test_multiply_shape(self):
        def f(a):
            return torch.empty(a.shape[0] * 2)

        r = str(make_fx(f, tracing_mode="symbolic")(torch.empty(4)).code).strip()
        self.assertExpectedInline(r, """\
def forward(self, a_1):
    sym_size = torch.ops.aten.sym_size(a_1, 0);  a_1 = None
    mul = sym_size * 2;  sym_size = None
    empty = torch.ops.aten.empty.memory_format([mul], device = device(type='cpu'), pin_memory = False);  mul = None
    sym_size_1 = torch.ops.aten.sym_size(empty, 0)
    return empty""")

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
        shape_env = self._test_dynamic(f, [(1, 6), (8, 1)], test_inputs)
        self.assertTrue(shape_env.evaluate_guards_for_args(torch.randn(1, 10), torch.randn(6, 1)))
        self.assertFalse(shape_env.evaluate_guards_for_args(torch.randn(1, 2), torch.randn(4, 1)))
        assert len(shape_env.guards) == 1

    def test_new_empty(self):
        def f(a, b):
            return a.new_empty(b.shape[0], b.shape[1] * 2)

        self._test_dynamic(f, [(2, 4), (4, 5)], [[(2, 3), (5, 7)], [(3, 7), (9, 3)]], assert_eq=False)


    def test_expand(self):
        def f(a):
            b = torch.mul(a, a)
            c = b.expand(a.shape)
            return c

        self._test_dynamic(f, [(3,)], [[(3,)], [(4,)], [(2,)]])
        self._test_dynamic(f, [(5, 1)], [[(4, 1)], [(3, 1)], [(6, 1)]])



make_fx_failures = {
    # unknown
    xfail('allclose'),
    xfail('equal'),
    xfail('linalg.eigvals'),
    xfail('nn.functional.max_pool1d', device_type='cpu'),
    # empty
    skip('new_empty'),
    skip('empty_like'),
    skip('empty'),
    # flaky
    skip('linalg.lstsq', 'grad_oriented'),
    skip('nn.functional.max_unpool1d', '', device_type='cpu'),
    skip('nn.functional.max_unpool2d', '', device_type='cpu'),
    skip('nn.functional.max_unpool3d', '', device_type='cpu'),
    skip('linalg.lstsq'),  # flaky, probably just a precision issue

    # data-dependent control flow
    xfail('cov'),
    xfail('istft'),
    xfail('nn.functional.gaussian_nll_loss'),
    xfail('tensor_split'),
    xfail('corrcoef'),
    xfail('quantile'),
    xfail('nanquantile'),

    # Seems like it's creating a sparse tensor that isn't captured by tensor.is_sparse
    xfail('sparse.sampled_addmm'),

    # ???
    xfail('nn.functional.ctc_loss'),
    # proxy tensor doesn't support sparse correctly right now
    skip('to_sparse'),
    # segfaults
    skip('block_diag'),
}

fake_tensor_failures = {
    # FakeTensor fallback doesn't work
    xfail('segment_reduce', 'lengths'),
    xfail('multinomial'),
    xfail('mvlgamma', 'mvlgamma_p_1'),
    xfail('mvlgamma', 'mvlgamma_p_3'),
    xfail('mvlgamma', 'mvlgamma_p_5'),
    xfail('cholesky'),
    xfail('cholesky_inverse'),
    # ASAN failures due to divide by 0
    skip('nn.functional.nll_loss'),
}

symbolic_tensor_failures = {
    # Needs complex-value support
    xfail('polar'),
    xfail('complex'),
    xfail('linalg.eig'),
    xfail('__getitem__', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('__rmatmul__', ''),  # aten.new_empty.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.amax', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.amin', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.argmax', ''),  # aten.argmax.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.argmin', ''),  # aten.argmin.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.cumprod', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.cumsum', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.log_softmax', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.logaddexp', ''),  # aten.logaddexp.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.mean', ''),  # ones() received an invalid combination of arguments - got (torch.Size, device=torch.device, ...
    xfail('_masked.median', ''),  # aten.nanmedian.dim - couldn't find symbolic meta function/decomposition
    xfail('_masked.norm', ''),  # aten.linalg_vector_norm.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.normalize', ''),  # aten.linalg_vector_norm.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.prod', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.softmax', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.softmin', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.std', ''),  # ones() received an invalid combination of arguments - got (torch.Size, device=torch.device, d...
    xfail('_masked.sum', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('_masked.var', ''),  # ones() received an invalid combination of arguments - got (torch.Size, device=torch.device, d...
    xfail('addmm', ''),  # aten.mm.default - couldn't find symbolic meta function/decomposition
    xfail('addmm', 'decomposed'),  # aten.mm.default - couldn't find symbolic meta function/decomposition
    xfail('addmv', ''),  # aten.addmv.default - couldn't find symbolic meta function/decomposition
    xfail('addr', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('all', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type promotion!
    xfail('aminmax', ''),  # aten.aminmax.default - couldn't find symbolic meta function/decomposition
    xfail('argmax', ''),  # aten.argmax.default - couldn't find symbolic meta function/decomposition
    xfail('argmin', ''),  # aten.argmin.default - couldn't find symbolic meta function/decomposition
    xfail('argsort', ''),  # aten.sort.default - couldn't find symbolic meta function/decomposition
    xfail('argwhere', ''),  # aten.nonzero.default - couldn't find symbolic meta function/decomposition
    xfail('as_strided', ''),  # aten.as_strided.default - couldn't find symbolic meta function/decomposition
    xfail('as_strided_scatter', ''),  # aten.as_strided_scatter.default - couldn't find symbolic meta function/decomposition
    xfail('baddbmm', ''),  # aten.baddbmm.default - couldn't find symbolic meta function/decomposition
    xfail('bernoulli', ''),  # aten.bernoulli.default - couldn't find symbolic meta function/decomposition
    xfail('bfloat16', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('bmm', ''),  # aten.bmm.default - couldn't find symbolic meta function/decomposition
    xfail('bool', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('broadcast_tensors', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('bucketize', ''),  # aten.bucketize.Tensor - couldn't find symbolic meta function/decomposition
    xfail('byte', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('cartesian_prod', ''),  # Tensors of type TensorImpl do not have numel
    xfail('cdist', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('chalf', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('char', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('cholesky_solve', ''),  # Could not run 'aten::_cholesky_solve_helper' with arguments from the 'Meta' back...
    xfail('chunk', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('clone', ''),  # aten.clone.default - couldn't find symbolic meta function/decomposition
    xfail('column_stack', ''),  # Tensors of type TensorImpl do not have numel
    xfail('constant_pad_nd', ''),  # aten.fill.Scalar - couldn't find symbolic meta function/decomposition
    xfail('count_nonzero', ''),  # Could not run 'aten::count_nonzero.dim_IntList' with arguments from the 'Meta' ba...
    xfail('cross', ''),  # aten.linalg_cross.default - couldn't find symbolic meta function/decomposition
    xfail('cummax', ''),  # aten.cummax.default - couldn't find symbolic meta function/decomposition
    xfail('cummin', ''),  # aten.cummin.default - couldn't find symbolic meta function/decomposition
    xfail('cumprod', ''),  # aten.cumprod.default - couldn't find symbolic meta function/decomposition
    xfail('cumsum', ''),  # aten.cumsum.default - couldn't find symbolic meta function/decomposition
    xfail('cumulative_trapezoid', ''),  # aten.slice.Tensor - couldn't find symbolic meta function/decomposition
    xfail('deg2rad', ''),  # aten.deg2rad.default - couldn't find symbolic meta function/decomposition
    xfail('diag_embed', ''),  # aten.diag_embed.default - couldn't find symbolic meta function/decomposition
    xfail('diagflat', ''),  # Tensors of type TensorImpl do not have numel
    xfail('diagonal', ''),  # aten.diagonal.default - couldn't find symbolic meta function/decomposition
    xfail('diagonal_scatter', ''),  # aten.diagonal_scatter.default - couldn't find symbolic meta function/decomposition
    xfail('diff', ''),  # aten.empty_like.default - couldn't find symbolic meta function/decomposition
    xfail('dist', ''),  # aten.dist.default - couldn't find symbolic meta function/decomposition
    xfail('double', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('dsplit', ''),  # aten.slice.Tensor - couldn't find symbolic meta function/decomposition
    xfail('eig', ''),  # aten.eig.default - couldn't find symbolic meta function/decomposition
    xfail('einsum', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('expand_as', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.fft2', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.fft', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.fftn', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.fftshift', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.hfft2', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.hfft', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('fft.hfftn', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.ifft2', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.ifft', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.ifftn', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.ifftshift', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.ihfft2', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.ihfft', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.ihfftn', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.irfft2', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.irfft', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('fft.irfftn', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.rfft2', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.rfft', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fft.rfftn', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('fill', ''),  # The underlying op of 'aten.stride' has no overload name '_schema'
    xfail('flatten', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('unflatten', ''),  # RuntimeError: Trying to call aten.size on a tensor with symbolic shapes...
    xfail('float', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('frexp', ''),  # aten.frexp.Tensor - couldn't find symbolic meta function/decomposition
    xfail('full_like', ''),  # aten.full_like.default - couldn't find symbolic meta function/decomposition
    xfail('gather', ''),  # aten.gather.default - couldn't find symbolic meta function/decomposition
    xfail('geqrf', ''),  # aten.geqrf.default - couldn't find symbolic meta function/decomposition
    xfail('gradient', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('half', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('histc', ''),  # Could not run 'aten::histc' with arguments from the 'Meta' backend. This could be because...
    xfail('histogram', ''),  # Could not run 'aten::histogram.bin_ct' with arguments from the 'Meta' backend. This c...
    xfail('histogramdd', ''),  # aten._histogramdd_bin_edges.default - couldn't find symbolic meta function/decomposition
    xfail('hsplit', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('i0', ''),  # aten.i0.default - couldn't find symbolic meta function/decomposition
    xfail('index_add', ''),  # Float
    xfail('index_copy', ''),  # Expected a long tensor for index, but got Float
    xfail('index_fill', ''),  # aten.index_fill.int_Scalar - couldn't find symbolic meta function/decomposition
    xfail('index_put', ''),  # aten.index_put.default - couldn't find symbolic meta function/decomposition
    xfail('index_reduce', ''),  # Float
    xfail('inner', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('int', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('inverse', ''),  # Tensors of type TensorImpl do not have numel
    xfail('isclose', ''),  # The underlying op of 'aten.stride' has no overload name '_schema'
    xfail('isin', ''),  # aten.isin.Tensor_Tensor - couldn't find symbolic meta function/decomposition
    xfail('isreal', ''),  # aten.empty_like.default - couldn't find symbolic meta function/decomposition
    xfail('kron', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('kthvalue', ''),  # aten.kthvalue.default - couldn't find symbolic meta function/decomposition
    xfail('lerp', ''),  # aten.lerp.Scalar - couldn't find symbolic meta function/decomposition
    xfail('linalg.cholesky', ''),  # aten.linalg_cholesky_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.cholesky_ex', ''),  # aten.linalg_cholesky_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.cond', ''),  # Tensors of type TensorImpl do not have numel
    xfail('linalg.cross', ''),  # aten.linalg_cross.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.det', ''),  # aten._linalg_det.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.eigh', ''),  # aten._linalg_eigh.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.eigvalsh', ''),  # aten._linalg_eigh.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.householder_product', ''),  # aten.linalg_householder_product.default - couldn't find symbolic meta funct...
    xfail('linalg.inv', ''),  # aten.linalg_inv_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.inv_ex', ''),  # aten.linalg_inv_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.ldl_factor', ''),  # aten.linalg_ldl_factor_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.ldl_factor_ex', ''),  # aten.linalg_ldl_factor_ex.default - couldn't find symbolic meta function/decompos...
    xfail('linalg.ldl_solve', ''),  # aten.linalg_ldl_solve.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.lu', ''),  # aten.linalg_lu.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.lu_factor', ''),  # aten.linalg_lu_factor_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.lu_factor_ex', ''),  # aten.linalg_lu_factor_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.lu_solve', ''),  # aten.linalg_lu_solve.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.matrix_power'),  # RuntimeError: Trying to call aten.size on a tensor with symbolic shape
    xfail('linalg.matrix_norm', ''),  # aten.linalg_vector_norm.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.matrix_rank', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.matrix_rank', 'hermitian'),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.multi_dot', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.norm', ''),  # TensorImpl do not have numel
    xfail('linalg.norm', 'subgradients_at_zero'),  # TensorImpl do not have numel
    xfail('linalg.pinv', ''),  # aten.linalg_pinv.atol_rtol_tensor - couldn't find symbolic meta function/decomposition
    xfail('linalg.pinv', 'singular'),  # aten.linalg_cholesky_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.pinv', 'hermitian'),  # aten.linalg_pinv.atol_rtol_tensor - couldn't find symbolic meta function/decompo...
    xfail('linalg.qr', ''),  # aten.linalg_qr.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.slogdet', ''),  # aten._linalg_slogdet.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.solve', ''),  # aten._linalg_solve_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.solve_ex', ''),  # aten._linalg_solve_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.solve_triangular', ''),  # aten.linalg_solve_triangular.default - couldn't find symbolic meta function/de...
    xfail('linalg.svd', ''),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.svdvals', ''),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.tensorinv', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.tensorsolve', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.vander', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.vecdot', ''),  # Could not run 'aten::vdot' with arguments from the 'Meta' backend. This could be ...
    xfail('linalg.vector_norm', ''),  # TensorImpl do not have numel
    xfail('logaddexp2', ''),  # aten.logaddexp2.default - couldn't find symbolic meta function/decomposition
    xfail('logaddexp', ''),  # aten.logaddexp.default - couldn't find symbolic meta function/decomposition
    xfail('logcumsumexp', ''),  # aten.logcumsumexp.default - couldn't find symbolic meta function/decomposition
    xfail('logdet', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('long', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('lu', ''),  # aten.linalg_lu_factor_ex.default - couldn't find symbolic meta function/decomposition
    xfail('lu_solve', ''),  # aten.linalg_lu_solve.default - couldn't find symbolic meta function/decomposition
    xfail('lu_unpack', ''),  # aten.lu_unpack.default - couldn't find symbolic meta function/decomposition
    xfail('masked_fill', ''),  # expected predicate to be bool, got torch.float32
    xfail('masked_scatter', ''),  # aten.masked_scatter.default - couldn't find symbolic meta function/decomposition
    xfail('masked_select', ''),  # aten.masked_select.default - couldn't find symbolic meta function/decomposition
    xfail('matmul', ''),  # aten.new_empty.default - couldn't find symbolic meta function/decomposition
    xfail('matrix_exp', ''),  # aten.linalg_matrix_exp.default - couldn't find symbolic meta function/decomposition
    xfail('max', 'reduction_with_dim'),  # aten.max.dim - couldn't find symbolic meta function/decomposition
    xfail('mean', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type promotion!
    xfail('median', ''),  # Could not run 'aten::median' with arguments from the 'Meta' backend. This could be becau...
    xfail('meshgrid', 'list_of_tensors'),  # Tensors of type TensorImpl do not have numel
    xfail('meshgrid', 'variadic_tensors'),  # Tensors of type TensorImpl do not have numel
    xfail('min', 'reduction_with_dim'),  # aten.min.dim - couldn't find symbolic meta function/decomposition
    xfail('mm', ''),  # aten.mm.default - couldn't find symbolic meta function/decomposition
    xfail('mode', ''),  # aten.mode.default - couldn't find symbolic meta function/decomposition
    xfail('msort', ''),  # aten.sort.default - couldn't find symbolic meta function/decomposition
    xfail('mv', ''),  # aten.mv.default - couldn't find symbolic meta function/decomposition
    xfail('nanmean', ''),  # The underlying op of 'aten.stride' has no overload name '_schema'
    xfail('nanquantile', ''),  # Could not run 'aten::equal' with arguments from the 'Meta' backend.
    xfail('narrow', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('native_layer_norm', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type promot...
    xfail('nn.functional.adaptive_avg_pool1d', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.adaptive_avg_pool2d', ''),  # argument 'size' must be tuple of ints, but found element o...
    xfail('nn.functional.adaptive_avg_pool3d', ''),  # aten._adaptive_avg_pool3d.default - couldn't find symbolic meta func...
    xfail('nn.functional.adaptive_max_pool1d', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.adaptive_max_pool2d', ''),  # aten.adaptive_max_pool2d.default - couldn't find symbolic meta funct...
    xfail('nn.functional.adaptive_max_pool3d', ''),  # argument 'output_size' (position 2) must be tupl...
    xfail('nn.functional.avg_pool1d', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.avg_pool2d', ''),  # aten.avg_pool2d.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.avg_pool3d', ''),  # aten.avg_pool3d.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.batch_norm', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.bilinear', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.binary_cross_entropy', ''),  # aten.new_empty.default - couldn't find symbolic meta function/decom...
    xfail('nn.functional.binary_cross_entropy_with_logits', ''),  # aten.binary_cross_entropy_with_logits.default - couldn'...
    xfail('nn.functional.conv1d', ''),  # aten.convolution.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.conv2d', ''),  # aten.convolution.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.conv_transpose1d', ''),  # aten.convolution.default - couldn't find symbolic meta function/decompo...
    xfail('nn.functional.conv_transpose2d', ''),  # aten.convolution.default - couldn't find symbolic meta function/decompo...
    xfail('nn.functional.conv_transpose3d', ''),  # aten.convolution.default - couldn't find symbolic meta function/decompo...
    xfail('nn.functional.cosine_embedding_loss', ''),  # The underlying op of 'aten.stride' has no overload name '_schema'
    xfail('nn.functional.cosine_similarity', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.cross_entropy', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.dropout2d', ''),  # Tensors of type TensorImpl do not have numel
    xfail('nn.functional.dropout3d', ''),  # Tensors of type TensorImpl do not have numel
    xfail('nn.functional.dropout', ''),  # Tensors of type TensorImpl do not have numel
    xfail('nn.functional.embedding_bag', ''),  # aten._embedding_bag_forward_only.default - couldn't find symbolic meta fun...
    xfail('nn.functional.embedding', ''),  # argument 'size' must be tuple of ints, but found element of type tor...
    xfail('nn.functional.feature_alpha_dropout', 'with_train'),  # Tensors of type TensorImpl do not have numel
    xfail('nn.functional.fractional_max_pool2d', ''),  # argument 'size' must be tuple of ints, but found element of t...
    xfail('nn.functional.fractional_max_pool3d', ''),  # argument 'size' must be tuple of ints, but found element of t...
    xfail('nn.functional.glu', ''),  # aten.glu.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.grid_sample', ''),  # aten.grid_sampler_2d.default - couldn't find symbolic meta function/decompos...
    xfail('nn.functional.group_norm', ''),  # 'torch._C.SymIntNode' and 'int'
    xfail('nn.functional.hinge_embedding_loss', ''),  # aten.empty_like.default - couldn't find symbolic meta function/deco...
    xfail('nn.functional.huber_loss', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.instance_norm', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.interpolate', 'area'),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.interpolate', 'bicubic'),  # aten.upsample_bicubic2d.vec - couldn't find symbolic meta function/d...
    xfail('nn.functional.interpolate', 'bilinear'),  # aten.upsample_bilinear2d.vec - couldn't find symbolic meta function...
    xfail('nn.functional.interpolate', 'linear'),  # aten.upsample_linear1d.vec - couldn't find symbolic meta function/dec...
    xfail('nn.functional.interpolate', 'nearest'),  # aten.upsample_nearest1d.vec - couldn't find symbolic meta function/d...
    xfail('nn.functional.interpolate', 'trilinear'),  # aten.upsample_trilinear3d.vec - couldn't find symbolic meta functi...
    xfail('nn.functional.kl_div', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type pro...
    xfail('nn.functional.l1_loss', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.layer_norm', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type...
    xfail('nn.functional.linear', ''),  # aten.mv.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.local_response_norm', ''),  # Tensors of type TensorImpl do not have numel
    xfail('nn.functional.margin_ranking_loss', ''),  # The underlying op of 'aten.stride' has no overload name '_schema'
    xfail('nn.functional.max_pool2d', ''),  # aten.max_pool2d_with_indices.default - couldn't find symbolic meta function/d...
    xfail('nn.functional.max_pool3d', ''),  # aten.max_pool3d_with_indices.default - couldn't find symbolic meta function/d...
    xfail('nn.functional.max_unpool1d', 'grad'),  # aten.max_unpool2d.default - couldn't find symbolic meta function/decom...
    xfail('nn.functional.max_unpool2d', 'grad'),  # aten.max_unpool2d.default - couldn't find symbolic meta function/decom...
    xfail('nn.functional.max_unpool3d', 'grad'),  # aten.max_unpool3d.default - couldn't find symbolic meta function/decom...
    xfail('nn.functional.mse_loss', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.multi_margin_loss', ''),  # Could not run 'aten::multi_margin_loss' with arguments from the...
    xfail('nn.functional.multilabel_margin_loss', ''),  # Could not run 'aten::multilabel_margin_loss_forward' with ...
    xfail('nn.functional.multilabel_soft_margin_loss', ''),  # aten.new_empty.default - couldn't find symbolic meta functio...
    xfail('nn.functional.normalize', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.pad', 'circular'),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.pad', 'constant'),  # aten.fill.Scalar - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.pad', 'reflect'),  # aten.reflection_pad1d.default - couldn't find symbolic meta function/decompo...
    xfail('nn.functional.pad', 'replicate'),  # aten.replication_pad1d.default - couldn't find symbolic meta function/deco...
    xfail('nn.functional.pdist', ''),  # Could not run 'aten::_pdist_forward' with arguments from the 'Meta' backend...
    xfail('nn.functional.pixel_shuffle', ''),  # aten.pixel_shuffle.default - couldn't find symbolic meta function/decompos...
    xfail('nn.functional.pixel_unshuffle', ''),  # aten.pixel_unshuffle.default - couldn't find symbolic meta function/deco...
    xfail('nn.functional.poisson_nll_loss', ''),  # The underlying op of 'aten.stride' has no overload name '_schema'
    xfail('nn.functional.rrelu', ''),  # aten.empty_like.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.smooth_l1_loss', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.soft_margin_loss', ''),  # aten.soft_margin_loss.default - couldn't find symbolic meta function/de...
    xfail('nn.functional.triplet_margin_loss', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing element...
    xfail('nn.functional.triplet_margin_with_distance_loss', ''),  # Unexpected type <class 'torch.SymIntNode'> when com...
    xfail('nn.functional.unfold', ''),  # aten.im2col.default - couldn't find symbolic meta function/decomposition
    xfail('nn.functional.upsample_bilinear', ''),  # aten.upsample_bilinear2d.vec - couldn't find symbolic meta function/de...
    xfail('nn.functional.upsample_nearest', ''),  # aten.upsample_nearest1d.vec - couldn't find symbolic meta function/deco...
    xfail('norm', ''),  # TensorImpl does not have numel
    xfail('norm', 'nuc'),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('normal', ''),  # aten.normal.Tensor_Tensor - couldn't find symbolic meta function/decomposition
    xfail('normal', 'number_mean'),  # aten.normal.float_Tensor - couldn't find symbolic meta function/decomposition
    xfail('ones_like', ''),  # aten.ones_like.default - couldn't find symbolic meta function/decomposition
    xfail('ormqr', ''),  # aten.ormqr.default - couldn't find symbolic meta function/decomposition
    xfail('outer', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('pca_lowrank', ''),  # aten.mm.default - couldn't find symbolic meta function/decomposition
    xfail('pinverse', ''),  # aten.linalg_pinv.atol_rtol_tensor - couldn't find symbolic meta function/decomposition
    xfail('polygamma', 'polygamma_n_0'),  # aten.polygamma.default - couldn't find symbolic meta function/decomposition
    xfail('polygamma', 'polygamma_n_1'),  # aten.polygamma.default - couldn't find symbolic meta function/decomposition
    xfail('polygamma', 'polygamma_n_2'),  # aten.polygamma.default - couldn't find symbolic meta function/decomposition
    xfail('polygamma', 'polygamma_n_3'),  # aten.polygamma.default - couldn't find symbolic meta function/decomposition
    xfail('polygamma', 'polygamma_n_4'),  # aten.polygamma.default - couldn't find symbolic meta function/decomposition
    xfail('put', ''),  # aten.clone.default - couldn't find symbolic meta function/decomposition
    xfail('quantile', ''),  # Could not run 'aten::equal' with arguments from the 'Meta' backend.
    xfail('qr', ''),  # aten.linalg_qr.default - couldn't find symbolic meta function/decomposition
    xfail('rad2deg', ''),  # aten.rad2deg.default - couldn't find symbolic meta function/decomposition
    xfail('rand_like', ''),  # aten.randn_like.default - couldn't find symbolic meta function/decomposition
    xfail('randint_like', ''),  # aten.randint_like.default - couldn't find symbolic meta function/decomposition
    xfail('randn_like', ''),  # aten.randn_like.default - couldn't find symbolic meta function/decomposition
    xfail('ravel', ''),  # Tensors of type TensorImpl do not have numel
    xfail('renorm', ''),  # aten.renorm.default - couldn't find symbolic meta function/decomposition
    xfail('repeat', ''),  # aten.repeat.default - couldn't find symbolic meta function/decomposition
    xfail('reshape_as', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('reshape', ''),  # Tensors of type TensorImpl do not have numel
    xfail('resize_', ''),  # aten.clone.default - couldn't find symbolic meta function/decomposition
    xfail('resize_as_', ''),  # aten.clone.default - couldn't find symbolic meta function/decomposition
    xfail('roll', ''),  # Tensors of type TensorImpl do not have numel
    xfail('rot90', ''),  # aten.empty_like.default - couldn't find symbolic meta function/decomposition
    xfail('round', ''),  # aten.round.default - couldn't find symbolic meta function/decomposition
    xfail('round', 'decimals_0'),  # aten.round.decimals - couldn't find symbolic meta function/decomposition
    xfail('round', 'decimals_3'),  # aten.round.decimals - couldn't find symbolic meta function/decomposition
    xfail('round', 'decimals_neg_3'),  # aten.round.decimals - couldn't find symbolic meta function/decomposition
    xfail('scatter_add', ''),  # aten.scatter_add.default - couldn't find symbolic meta function/decomposition
    xfail('scatter', ''),  # aten.scatter.src - couldn't find symbolic meta function/decomposition
    xfail('scatter_reduce', 'amax'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decomposition
    xfail('scatter_reduce', 'amin'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decomposition
    xfail('scatter_reduce', 'mean'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decomposition
    xfail('scatter_reduce', 'prod'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decomposition
    xfail('scatter_reduce', 'sum'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decomposition
    xfail('searchsorted', ''),  # Could not run 'aten::searchsorted.Tensor' with arguments from the 'Meta' backend. ...
    xfail('segment_reduce', 'offsets'),  # aten.segment_reduce.default - couldn't find symbolic meta function/decomposition
    xfail('select', ''),  # aten.select.int - couldn't find symbolic meta function/decomposition
    xfail('select_scatter', ''),  # aten.select_scatter.default - couldn't find symbolic meta function/decomposition
    xfail('sgn', ''),  # aten.sgn.default - couldn't find symbolic meta function/decomposition
    xfail('short', ''),  # aten._to_copy.default - couldn't find symbolic meta function/decomposition
    xfail('sinc', ''),  # aten.sinc.default - couldn't find symbolic meta function/decomposition
    xfail('slice_scatter', ''),  # aten.slice_scatter.default - couldn't find symbolic meta function/decomposition
    xfail('sort', ''),  # aten.sort.default - couldn't find symbolic meta function/decomposition
    xfail('special.airy_ai', ''),  # aten.special_airy_ai.default - couldn't find symbolic meta function/decomposition
    xfail('special.bessel_j0', ''),  # aten.special_bessel_j0.default - couldn't find symbolic meta function/decomposition
    xfail('special.bessel_j1', ''),  # aten.special_bessel_j1.default - couldn't find symbolic meta function/decomposition
    xfail('special.bessel_y0', ''),  # aten.special_bessel_y0.default - couldn't find symbolic meta function/decomposition
    xfail('special.bessel_y1', ''),  # aten.special_bessel_y1.default - couldn't find symbolic meta function/decomposition
    xfail('special.chebyshev_polynomial_t', ''),  # aten.special_chebyshev_polynomial_t.default - couldn't find symbolic me...
    xfail('special.chebyshev_polynomial_u', ''),  # aten.special_chebyshev_polynomial_u.default - couldn't find symbolic me...
    xfail('special.entr', ''),  # aten.special_entr.default - couldn't find symbolic meta function/decomposition
    xfail('special.erfcx', ''),  # aten.special_erfcx.default - couldn't find symbolic meta function/decomposition
    xfail('special.hermite_polynomial_h', ''),  # aten.special_hermite_polynomial_h.default - couldn't find symbolic meta f...
    xfail('special.hermite_polynomial_he', ''),  # aten.special_hermite_polynomial_he.default - couldn't find symbolic meta...
    xfail('special.laguerre_polynomial_l', ''),  # aten.special_laguerre_polynomial_l.default - couldn't find symbolic meta...
    xfail('special.log_ndtr', ''),  # aten.special_log_ndtr.default - couldn't find symbolic meta function/decomposition
    xfail('special.modified_bessel_i0', ''),  # aten.special_modified_bessel_i0.default - couldn't find symbolic meta funct...
    xfail('special.modified_bessel_i1', ''),  # aten.special_modified_bessel_i1.default - couldn't find symbolic meta funct...
    xfail('special.modified_bessel_k0', ''),  # aten.special_modified_bessel_k0.default - couldn't find symbolic meta funct...
    xfail('special.modified_bessel_k1', ''),  # aten.special_modified_bessel_k1.default - couldn't find symbolic meta funct...
    xfail('special.ndtri', ''),  # aten.special_ndtri.default - couldn't find symbolic meta function/decomposition
    xfail('special.polygamma', 'special_polygamma_n_0'),  # aten.polygamma.default - couldn't find symbolic meta function/...
    xfail('special.scaled_modified_bessel_k0', ''),  # aten.special_scaled_modified_bessel_k0.default - couldn't find symbo...
    xfail('special.scaled_modified_bessel_k1', ''),  # aten.special_scaled_modified_bessel_k1.default - couldn't find symbo...
    xfail('special.spherical_bessel_j0', ''),  # aten.special_spherical_bessel_j0.default - couldn't find symbolic meta fun...
    xfail('special.xlog1py', ''),  # aten.special_xlog1py.default - couldn't find symbolic meta function/decomposition
    xfail('split', ''),  # 'torch._C.SymIntNode' and 'int'
    xfail('split', 'list_args'),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('split_with_sizes', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('std', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type promotion!
    xfail('std_mean', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type promotion!
    xfail('stft', ''),  # argument 'size' must be tuple of ints, but found element of type torch._C.SymIntNode at...
    xfail('sum_to_size', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('svd', ''),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('svd_lowrank', ''),  # aten.mm.default - couldn't find symbolic meta function/decomposition
    xfail('symeig', ''),  # aten.symeig.default - couldn't find symbolic meta function/decomposition
    xfail('take_along_dim', ''),  # dtype of indices should be Long but got Float
    xfail('take', ''),  # aten.take.default - couldn't find symbolic meta function/decomposition
    xfail('tensordot', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('tile', ''),  # aten.repeat.default - couldn't find symbolic meta function/decomposition
    xfail('topk', ''),  # aten.topk.default - couldn't find symbolic meta function/decomposition
    xfail('trapz', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('trapezoid', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('triangular_solve', ''),  # aten.triangular_solve.default - couldn't find symbolic meta function/decomposition
    xfail('tril', ''),  # aten.tril.default - couldn't find symbolic meta function/decomposition
    xfail('triu', ''),  # aten.triu.default - couldn't find symbolic meta function/decomposition
    xfail('unfold', ''),  # aten.unfold.default - couldn't find symbolic meta function/decomposition
    xfail('var_mean', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type promotion!
    xfail('var', ''),  # Unexpected type <class 'torch.SymIntNode'> when computing elementwise type promotion!
    xfail('vdot', ''),  # aten.vdot.default - couldn't find symbolic meta function/decomposition
    xfail('view_as_complex', ''),  # aten.view_as_complex.default - couldn't find symbolic meta function/decomposition
    xfail('view_as', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('view', ''),  # Tensors of type TensorImpl do not have numel
    xfail('vsplit', ''),  # aten.size.default - couldn't find symbolic meta function/decomposition
    xfail('where', ''),  # expected predicate to be bool, got torch.float32
    xfail('zero_', ''),  # aten.clone.default - couldn't find symbolic meta function/decomposition
    xfail('zeros_like', ''),  # aten.zeros_like.default - couldn't find symbolic meta function/decomposition
    xfail('unbind', ''),  # aten.unbind.int - couldn't find symbolic meta function/decomposition
}
symbolic_tensor_segfaults = {
    skip('_masked.logsumexp', ''),  # Tensors of type TensorImpl do not have numel
}

symbolic_tensor_failures.update(symbolic_tensor_segfaults)

def _test_make_fx_helper(self, device, dtype, op, tracing_mode):
    def f(args, kwargs):
        return op.op(*args, **kwargs)
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
    new_f = None
    for sample_input in sample_inputs_itr:
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs

        try:
            new_f = make_fx(f, tracing_mode=tracing_mode)(args, kwargs)
        except DynamicOutputShapeException as e:
            self.skipTest("Dynamic output shape operation in trace")

        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
                arg.uniform_(0, 1)
        try:
            old_out = f(args, kwargs)
        except Exception:
            continue
        new_out = wrapper_set_seed(new_f, args, kwargs)
        self.assertEqual(new_out, old_out)

class TestProxyTensorOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_exhaustive', make_fx_failures)
    def test_make_fx_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "real")

    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_fake_exhaustive', make_fx_failures.union(fake_tensor_failures))
    def test_make_fx_fake_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "fake")

    @skipIfNoSympy
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive',
             make_fx_failures | fake_tensor_failures | symbolic_tensor_failures)
    def test_make_fx_symbolic_exhaustive(self, device, dtype, op):
        _test_make_fx_helper(self, device, dtype, op, "symbolic")


only_for = ("cpu")
instantiate_device_type_tests(TestProxyTensorOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
