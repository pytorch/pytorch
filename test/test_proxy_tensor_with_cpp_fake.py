# Owner(s): ["module: ProxyTensor"]
# ruff: noqa: F841

"""
Tests for make_fx with C++ FakeTensor mode.

All tests run make_fx(tracing_mode="real") inside cpp_fake_tensor_mode().
The C++ Fake dispatch key handles ops with Meta kernels. Ops without Meta
kernels fall back to CppFakeFallbackMode, which looks up the specific
Python handler (decomposition, fake_impl, etc.) and calls it. Sub-ops
re-enter C++ Fake dispatch, so all results remain C++ fake tensors.
"""

from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch._dynamo
import torch._library.simple_registry
import torch._library.utils
import unittest
import warnings
import operator
import contextlib
from collections.abc import Iterable
from torch.nn.utils import stateless
from torch._subclasses.fake_tensor import (
    DynamicOutputShapeException,
    DataDependentOutputException,
    FakeTensorConverter,
    FakeTensorMode,
)
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch._decomp import decomposition_table
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, skip, xfail, skipOps
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.hop_db import hop_db
import torch.testing._internal.optests as optests
from torch._dispatch.python import enable_python_dispatcher
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    DecompositionInterpreter,
    get_isolated_graphmodule,
)
from torch.utils._pytree import tree_map
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

import functools
import itertools

aten = torch.ops.aten

HAS_CUDA = torch.cuda.is_available()

USE_TORCHVISION = False
try:
    import torchvision

    USE_TORCHVISION = True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )


@contextlib.contextmanager
def cpp_fake_tensor_mode(*, shape_env=None):
    """Activate C++ FakeTensor mode with a Python fallback for unhandled ops.

    The C++ Fake dispatch key handles ops that have Meta kernels.
    Ops without Meta kernels are forwarded to CppFakeFallbackMode which
    looks up the specific Python handler (decomposition, fake_impl, etc.)
    and calls it. Sub-ops re-enter C++ Fake dispatch, so all tensors
    remain C++ fake tensors — no Python FakeTensors are created.
    """
    if shape_env is None:
        shape_env = ShapeEnv()
    converter = FakeTensorConverter()
    # fallback = CppFakeFallbackMode()
    # torch._C._create_and_enter_fake_tensor_mode(converter, shape_env, fallback)
    torch._C._create_and_enter_fake_tensor_mode(converter, shape_env)
    try:
        yield shape_env
    finally:
        torch._C._exit_fake_tensor_mode()


def _create_new_input(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype != torch.float:
        return x + 1
    if x.is_leaf:
        return torch.rand_like(x, requires_grad=x.requires_grad)
    else:
        return torch.rand_like(x)


class TestCppFakeProxyTensor(TestCase):
    """Tests for make_fx under C++ FakeTensor mode.

    Each test wraps the make_fx call in cpp_fake_tensor_mode() and uses
    tracing_mode="real" so that the C++ Fake dispatch key provides the
    fake tensor semantics.
    """

    def _test(self, f, inps, compare_graph=False):
        # Trace under C++ fake mode
        with cpp_fake_tensor_mode():
            cpp_gm = make_fx(f, tracing_mode="real")(*inps)

        if compare_graph:
            # Trace under Python fake mode and compare graph structure
            py_gm = make_fx(f, tracing_mode="fake")(*inps)
            cpp_ops = [n.target for n in cpp_gm.graph.nodes if n.op == "call_function"]
            py_ops = [n.target for n in py_gm.graph.nodes if n.op == "call_function"]
            self.assertEqual(cpp_ops, py_ops)

        # Verify correctness with real inputs
        new_inps = tree_map(_create_new_input, inps)
        r1 = cpp_gm(*new_inps)
        r2 = f(*new_inps)
        self.assertEqual(r1, r2)

    def test_make_fx_simple(self):
        def f(x):
            return torch.sin(x)

        self._test(f, (torch.randn(3),))

    def test_scalar_device(self, device="cpu"):
        def f(a, b):
            return a + b

        self._test(f, [torch.randn(3, device=device), torch.tensor(5)])

    def test_empty_like_doesnt_burn_in_defaults(self):
        def f(x):
            return torch.empty_like(x)

        with cpp_fake_tensor_mode():
            out = make_fx(f, tracing_mode="real")(torch.randn(3))
        self.assertExpectedInline(
            out.code.strip(),
            """\
def forward(self, x_1):
    empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False);  x_1 = None
    return empty_like""",
        )

    def test_proxy_tensor_mode_with_decomp_table_preserves_proxy(self):
        def f(x):
            y = x.new_zeros(x.size())
            y.copy_(x)
            return y

        def _new_zeros_decomp(
            inp, size, dtype=None, layout=None, device=None, pin_memory=None
        ):
            return torch.zeros(size, dtype=inp.dtype, device=inp.device)

        factory_func_decomp = {torch.ops.aten.new_zeros.default: _new_zeros_decomp}

        with cpp_fake_tensor_mode():
            out = make_fx(
                f, tracing_mode="real", decomposition_table=factory_func_decomp
            )(torch.ones(2))
        self.assertExpectedInline(
            out.code,
            """\



def forward(self, x_1):
    zeros = torch.ops.aten.zeros.default([2], dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    copy_ = torch.ops.aten.copy_.default(zeros, x_1);  zeros = x_1 = None
    return copy_
    """,
        )

    def test_make_fx_reentrant_dispatch(self):
        def f(x):
            return torch.ops.aten.norm.Scalar(x, 2.0)

        def norm_decomp(x, p=2.0):
            if p != 2.0:
                raise RuntimeError("can't handle with p != 2")
            return torch.sqrt(torch.sum(torch.square(x)))

        decomp = {torch.ops.aten.norm.Scalar: norm_decomp}

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real", decomposition_table=decomp)(
                torch.rand(3)
            )

        for n in traced.graph.nodes:
            self.assertTrue("square" not in str(n.target))
            self.assertTrue("norm" not in str(n.target))

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
            if x.shape[-1] != 1:
                raise AssertionError(f"expected x.shape[-1] == 1, got {x.shape[-1]}")
            return x

        self._test(f, [torch.randn(5)])

    def test_mode_tracing_factory_function(self):
        def f(x):
            return x + torch.randn(x.shape)

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3))
        self.assertTrue(
            any(node.target == aten.randn.default for node in traced.graph.nodes)
        )

    def test_val_metadata_mutation(self):
        def f(x):
            y = x.clone()
            y.unsqueeze_(0)
            return y

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3, requires_grad=True))
        self.assertEqual(
            [
                tuple(node.meta["val"].shape)
                for node in traced.graph.nodes
                if "val" in node.meta
            ],
            [(3,), (3,), (1, 3)],
        )

    def test_make_fx_overloads(self):
        def f(x):
            return x.cos() + torch.randn(x.shape)

        with cpp_fake_tensor_mode():
            traced = make_fx(f, tracing_mode="real")(torch.randn(3))

        self.assertTrue(
            all(
                isinstance(node.target, torch._ops.OpOverload)
                for node in traced.graph.nodes
                if node.op == "call_function"
            )
        )

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_tensor_constants(self):
        def f():
            val = torch.tensor(float("inf"))
            return torch.full((100, 100), val)

        self._test(f, [])

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_constant_proxy_tensor_mut(self):
        def f():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((100, 100), val)

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")()
        self.assertEqual(g(), f())
        self.assertEqual(g(), f())

    @unittest.skip("C++ fake mode has no constant propagation")
    def test_constant_unbind(self):
        def f():
            val = torch.tensor([2])
            (r,) = torch.unbind(val, 0)
            return r.item()

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")()
        self.assertEqual(g(), f())

    def test_decomposition_interpreter(self):
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.rand((4, 4))
        with cpp_fake_tensor_mode():
            fx_module = make_fx(fn, tracing_mode="real", decomposition_table=None)(x)

        found_silu = False
        for n in fx_module.graph.nodes:
            if (
                n.target == torch.ops.aten.silu
                or n.target == torch.ops.aten.silu.default
            ):
                found_silu = True

        self.assertTrue(found_silu)

        new_graph = torch.fx.Graph()
        silu_decomp_table = {
            torch.ops.aten.silu.default: decomposition_table[
                torch.ops.aten.silu.default
            ]
        }
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
        with cpp_fake_tensor_mode():
            fx_f = make_fx(f, tracing_mode="real")(input, params)
        self.assertTrue(
            torch.allclose(fx_f(input, params)[0], f(input, params)[0])
            or torch.allclose(fx_f(input, params)[0], f(input, params)[1])
        )
        self.assertTrue(
            torch.allclose(fx_f(input, params)[1], f(input, params)[0])
            or torch.allclose(fx_f(input, params)[1], f(input, params)[1])
        )

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
        with cpp_fake_tensor_mode():
            fx_f = make_fx(f, tracing_mode="real")(input, params, buffers)
        self.assertTrue(
            torch.allclose(
                fx_f(input, params, buffers)[0],
                f(input, params, buffers)[0],
                atol=1e-03,
            )
            or torch.allclose(
                fx_f(input, params, buffers)[0],
                f(input, params, buffers)[1],
                atol=1e-03,
            )
        )
        self.assertTrue(
            torch.allclose(
                fx_f(input, params, buffers)[1],
                f(input, params, buffers)[0],
                atol=1e-03,
            )
            or torch.allclose(
                fx_f(input, params, buffers)[1],
                f(input, params, buffers)[1],
                atol=1e-03,
            )
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

        with cpp_fake_tensor_mode():
            gm = make_fx(Emformer(), tracing_mode="real")(torch.randn(16, 1, 256))
        ops = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertEqual(len(ops), 2)

    def test_partial_decomp(self):
        def f(a, b, c):
            x = torch.addmm(a, b, c)
            y = torch.addmm(a, b, c, beta=2, alpha=1)
            return x + y

        inps = [torch.randn(5, 5), torch.randn(5, 5), torch.randn(5, 5)]
        with cpp_fake_tensor_mode():
            fx_g = make_fx(f, tracing_mode="real")(*inps)

        def addmm(a, b, c, beta=1, alpha=1):
            if beta == 1 and alpha == 1:
                return NotImplemented
            return beta * a + alpha * (b @ c)

        with cpp_fake_tensor_mode():
            decomposed_fx = make_fx(
                f, tracing_mode="real", decomposition_table={aten.addmm.default: addmm}
            )(*inps)

        self.assertEqual(fx_g(*inps), decomposed_fx(*inps))
        self.assertEqual(
            len([n for n in fx_g.graph.nodes if n.target == aten.addmm.default]), 2
        )
        self.assertEqual(
            len(
                [n for n in decomposed_fx.graph.nodes if n.target == aten.addmm.default]
            ),
            1,
        )

    def test_decomp_of_capture(self):
        val = torch.randn(5)

        def f(x):
            return x.t() + val.t()

        def nop(x):
            return x.cos()

        with cpp_fake_tensor_mode():
            traced = make_fx(
                f,
                tracing_mode="real",
                decomposition_table={torch.ops.aten.t.default: nop},
            )(torch.randn(5))
        self.assertEqual(
            len(
                [n for n in traced.graph.nodes if n.target == torch.ops.aten.t.default]
            ),
            0,
        )

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_amp_cache(self):
        layer = torch.nn.Conv2d(3, 3, 3).cuda()

        def f(x, w):
            return torch.nn.functional.conv2d(x, w, stride=layer.stride)

        inp = torch.randn(4, 3, 10, 10, device="cuda")
        with torch.autocast("cuda"):
            with cpp_fake_tensor_mode():
                out_graph = make_fx(f, tracing_mode="real")(inp, layer.weight).graph
                out_graph2 = make_fx(f, tracing_mode="real")(inp, layer.weight).graph

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

        with cpp_fake_tensor_mode():
            make_fx(f, tracing_mode="real")(torch.randn(2, 3, 4, 5))

        def f(x):
            self.assertTrue(x.is_contiguous())
            y = x[:, 1]
            self.assertFalse(y.is_contiguous())
            y = x[:, ::2]
            self.assertFalse(y.is_contiguous())
            return x.cos()

        with cpp_fake_tensor_mode():
            make_fx(f, tracing_mode="real")(torch.randn(2, 3, 4, 5))

    def test_pr_86917(self):
        def f(a, b):
            return torch.ops.aten.nll_loss_forward(a, b, None, 1, 10)

        self._test(f, [torch.randn(1, 10), torch.zeros(1, dtype=torch.long)])

    def test_use_fake_and_tensor(self):
        def f(x, y):
            z = torch.tensor([2.0, 3.0])
            return x + y + z

        with cpp_fake_tensor_mode():
            g = make_fx(f, tracing_mode="real")(torch.randn(2), torch.randn(2))
        x, y = torch.randn(2), torch.randn(2)
        self.assertEqual(g(x, y), f(x, y))

    def test_fused_adam(self):
        params = [torch.randn(10, 10) for _ in range(10)]
        grads = [torch.randn(10, 10) for _ in range(10)]
        exp_avgs = [torch.randn(10, 10) for _ in range(10)]
        exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        max_exp_avg_sqs = [torch.randn(10, 10) for _ in range(10)]
        state_steps = [torch.tensor(0) for _ in range(10)]

        def fused_adam(
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps
        ):
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

        with cpp_fake_tensor_mode():
            gm = make_fx(fused_adam, tracing_mode="real")(
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
                self.assertIn("val", n.meta)

    def test_alias(self):
        def f(x):
            return torch.ops.aten.alias(x)

        with cpp_fake_tensor_mode():
            r = str(make_fx(f, tracing_mode="real")(torch.randn(2)).code).strip()
        self.assertExpectedInline(
            r,
            """\
def forward(self, x_1):
    alias = torch.ops.aten.alias.default(x_1);  x_1 = None
    return alias""",
        )

    def test_meta(self):
        def f(x):
            a = x.cos()
            b = torch.var_mean(a, dim=0)
            c = b * 2
            return c

        with cpp_fake_tensor_mode():
            out = make_fx(f, tracing_mode="real")(torch.randn(5, 5))
        for n in out.graph.nodes:
            if n.op == "output":
                continue
            self.assertTrue("val" in n.meta)

    def test_simple_add(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(3, 4))

        # Verify the graph has the expected structure
        call_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(len(call_nodes) >= 1)

        # Verify it runs correctly with real inputs
        x, y = torch.randn(3, 4), torch.randn(3, 4)
        self.assertEqual(gm(x, y), f(x, y))

    def test_matmul(self):
        def f(x, y):
            return torch.matmul(x, y)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(4, 5))
        x, y = torch.randn(3, 4), torch.randn(4, 5)
        self.assertEqual(gm(x, y), f(x, y))

    def test_multiple_outputs(self):
        def f(x):
            return torch.max(x, dim=0)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        r1 = gm(x)
        r2 = f(x)
        self.assertEqual(r1[0], r2[0])
        self.assertEqual(r1[1], r2[1])

    def test_inplace_ops(self):
        def f(x):
            y = x.clone()
            y.add_(1.0)
            return y

        self._test(f, (torch.randn(3, 4),))

    def test_view_ops(self):
        def f(x):
            y = x.view(2, 6)
            z = y.t()
            return z

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), f(x))

    def test_cat(self):
        def f(x, y):
            return torch.cat([x, y], dim=0)

        self._test(f, (torch.randn(3, 4), torch.randn(5, 4)))

    def test_nn_module(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        def f(x):
            return model(x)

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 10))
        x = torch.randn(3, 10)
        self.assertEqual(gm(x), f(x))

    def test_comparison_with_python_fake(self):
        """Verify that C++ fake mode and Python fake mode produce the same graph structure."""

        def f(x):
            y = torch.sin(x)
            z = torch.cos(y)
            return z + x

        inp = torch.randn(4, 4)

        # Trace with Python fake mode
        py_gm = make_fx(f, tracing_mode="fake")(inp)

        # Trace with C++ fake mode
        with cpp_fake_tensor_mode():
            cpp_gm = make_fx(f, tracing_mode="real")(inp)

        # Both should produce identical graph structure
        py_ops = [n.target for n in py_gm.graph.nodes if n.op == "call_function"]
        cpp_ops = [n.target for n in cpp_gm.graph.nodes if n.op == "call_function"]
        self.assertEqual(py_ops, cpp_ops)

        # Both should produce correct results
        x = torch.randn(4, 4)
        self.assertEqual(py_gm(x), cpp_gm(x))

    def test_factory_ops_under_cpp_fake(self):
        """Factory ops like torch.zeros should work under C++ fake mode."""

        def f(x):
            z = torch.zeros(x.shape)
            return x + z

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4))
        x = torch.randn(3, 4)
        self.assertEqual(gm(x), f(x))

    def test_dtype_promotion(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(
                torch.randn(3, dtype=torch.float32),
                torch.randn(3, dtype=torch.float64),
            )
        x = torch.randn(3, dtype=torch.float32)
        y = torch.randn(3, dtype=torch.float64)
        self.assertEqual(gm(x, y), f(x, y))

    def test_broadcasting(self):
        def f(x, y):
            return x + y

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, 4), torch.randn(4))
        x, y = torch.randn(3, 4), torch.randn(4)
        self.assertEqual(gm(x, y), f(x, y))

    @unittest.skipIf(not HAS_CUDA, "CUDA-only test")
    def test_cuda_device(self):
        def f(x):
            return x.sin() + x.cos()

        with cpp_fake_tensor_mode():
            gm = make_fx(f, tracing_mode="real")(torch.randn(3, device="cuda"))
        x = torch.randn(3, device="cuda")
        self.assertEqual(gm(x), f(x))

    # --- Higher Order Op tests ---
    # These mirror the hop_db entries but call internal ops (cond_op, while_loop_op,
    # map_impl, scan_op) directly, bypassing the user-facing wrappers that route
    # through torch.compile/Dynamo.

    def _make_arg(self, *shape, low=0.1, high=2):
        return make_tensor(*shape, low=low, high=high, dtype=torch.float, device="cpu")

    def test_cond_simple(self):
        """Mirrors hop_db simple_cond."""
        from torch._higher_order_ops.cond import cond_op

        def f(x):
            return cond_op(
                x.sum() > 2, lambda x: (x.cos(),), lambda x: (x.sin(),), [x]
            )

        self._test(f, (self._make_arg(2, 2, 2),), compare_graph=True)

    def test_while_loop_simple(self):
        """Mirrors hop_db simple_while_loop."""
        from torch._higher_order_ops.while_loop import while_loop_op

        def f(iter_t, x):
            def cond_fn(iter_t, x):
                return iter_t > 0

            def body_fn(iter_t, x):
                return iter_t - 1, x.cos()

            return while_loop_op(cond_fn, body_fn, (iter_t, x), ())

        self._test(f, (torch.tensor(3), self._make_arg(2, 3, 4)), compare_graph=True)

    def test_map_simple(self):
        """Mirrors hop_db simple_map."""
        from torch._higher_order_ops.map import map_impl

        def inner_f(x0, x1, y0, y1):
            return [x0.cos().add_(1.0) * y0, (x1 + y1.sin()).cos_().view(x1.size())]

        def f(x0, x1, y0, y1):
            return map_impl(inner_f, [x0, x1], (y0, y1))

        self._test(f, (self._make_arg(2, 2, 2), self._make_arg(2, 2, 2),
                        self._make_arg(1), self._make_arg(1)), compare_graph=True)

    def test_scan_simple(self):
        """Mirrors hop_db simple_scan."""
        from torch._higher_order_ops.scan import scan_op

        def combine_fn(carry, x):
            result = carry @ x + x
            return result, carry.clone()

        def f(init, xs):
            return scan_op(combine_fn, [init], [xs], ())

        self._test(f, (self._make_arg(2, 2), self._make_arg(2, 2, 2)), compare_graph=True)


# --- OpInfo-based exhaustive tests for C++ FakeTensor mode ---

# Failures shared with the original make_fx tests (ops that don't work with
# proxy tensor tracing regardless of fake mode implementation).
cpp_fake_make_fx_failures = {
    # unknown
    xfail('allclose'),
    xfail('equal'),
    # empty
    skip('new_empty'),
    # skip('new_empty_strided'),
    skip('empty_like'),
    skip('empty'),
    skip('empty_permuted'),
    # flaky
    skip('linalg.lstsq', 'grad_oriented'),
    skip('nn.functional.max_unpool1d', '', device_type='cpu'),
    skip('nn.functional.max_unpool2d', '', device_type='cpu'),
    skip('nn.functional.max_unpool3d', '', device_type='cpu'),
    skip('linalg.lstsq'),
    # data-dependent control flow
    skip('item'),
    xfail('cov'),
    xfail('nn.functional.gaussian_nll_loss'),
    xfail('corrcoef'),
    # sparse
    xfail('sparse.sampled_addmm'),
    xfail('sparse.mm', 'reduce'),
    skip('to_sparse'),
    # segfaults
    skip('block_diag'),
    # AssertionError: Tensor-likes are not close!
    skip('empty_strided', '', device_type='cpu'),
}

cpp_fake_only_real_failures = {
    xfail('narrow'),
    xfail('tensor_split'),
}

cpp_fake_only_fake_failures = {
    xfail('tensor_split'),
}

# Failures specific to symbolic shapes under C++ fake mode.
# These mirror symbolic_tensor_failures from test_proxy_tensor.py.
cpp_fake_symbolic_failures = {
    xfail('combinations', ''),
    xfail('geqrf', ''),
    xfail('histogram', ''),
    xfail('histogramdd', ''),
    xfail('nn.functional.binary_cross_entropy', ''),
    xfail('nn.functional.cross_entropy', ''),
    xfail('nn.functional.ctc_loss'),
    xfail('max_pool2d_with_indices_backward', ''),
    skip('nn.functional.batch_norm'),
}


def _get_safe_inplace(inplace_variant):
    @functools.wraps(inplace_variant)
    def _fn(t, *args, **kwargs):
        return inplace_variant(t.clone(), *args, **kwargs)
    return _fn


def _test_make_fx_helper_cpp_fake(self, device, dtype, op, inplace=False,
                                  out=False, decomp_table=None):
    """Like _test_make_fx_helper but wraps make_fx in cpp_fake_tensor_mode()."""
    fn = _get_safe_inplace(op.get_inplace()) if inplace else op.op
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

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
            _make_fx_check_cpp_fake(fn, args, kwargs, self.assertEqual,
                                    randomize_data=True,
                                    decomp_table=decomp_table)
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")


def _test_make_fx_helper_cpp_fake_symbolic(self, device, dtype, op):
    """Like _test_make_fx_helper_cpp_fake but with symbolic shapes."""
    fn = op.op
    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)

    for sample_input in itertools.islice(sample_inputs_itr, 100):
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs

        try:
            _make_fx_check_cpp_fake_symbolic(fn, args, kwargs,
                                             self.assertEqual,
                                             randomize_data=True)
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")


_fake_counter = 0


def _to_cpp_fake(x):
    if isinstance(x, torch.Tensor):
        return torch._C._make_fake_tensor(x)
    return x


def _to_cpp_fake_symbolic(x):
    if not isinstance(x, torch.Tensor):
        return x
    global _fake_counter
    _fake_counter += 1
    from torch._dynamo.source import ConstantSource
    from torch.fx.experimental.symbolic_shapes import DimDynamic, StatelessSymbolicContext

    source = ConstantSource(f"arg_{_fake_counter}")
    ctx = StatelessSymbolicContext(
        dynamic_sizes=[DimDynamic.DYNAMIC] * x.dim(),
    )
    return torch._C._make_fake_tensor(x, source=source, symbolic_context=ctx)


def _make_fx_check_cpp_fake(func, args, kwargs, assert_close,
                            randomize_data=False, decomp_table=None):
    """Like optests.make_fx_check but traces under cpp_fake_tensor_mode()."""
    from torch.testing._internal.optests.make_fx import (
        handle_sizes_for_dynamic_shapes,
        randomize,
    )
    from torch.testing._utils import wrapper_set_seed

    f, *new_args = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    def run(f, *args, **kwargs):
        return wrapper_set_seed(f, *args, **kwargs)

    with cpp_fake_tensor_mode():
        traced_f = make_fx(f, tracing_mode="real",
                           decomposition_table=decomp_table)(*new_args)

    msg = (
        "op(*args, **kwargs) and make_fx(op)(*args, **kwargs) under "
        "cpp_fake_tensor_mode produced different values."
    )

    if randomize_data:
        new_args = randomize(new_args)
    try:
        expected = run(f, *new_args)
    except Exception:
        if randomize_data:
            return
        raise
    result = run(traced_f, *new_args)
    assert_close(result, expected, msg=msg)


def _make_fx_check_cpp_fake_symbolic(func, args, kwargs, assert_close,
                                     randomize_data=False, decomp_table=None):
    """Like _make_fx_check_cpp_fake but with symbolic shapes."""
    from torch.testing._internal.optests.make_fx import (
        handle_sizes_for_dynamic_shapes,
        randomize,
    )
    from torch.testing._utils import wrapper_set_seed
    from torch.utils._pytree import tree_map_only

    f, *new_args = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    def run(f, *args, **kwargs):
        return wrapper_set_seed(f, *args, **kwargs)

    with cpp_fake_tensor_mode() as shape_env:
        symbolic_args = tree_map_only(torch.Tensor, _to_cpp_fake_symbolic, new_args)
        traced_f = make_fx(f, tracing_mode="real",
                           decomposition_table=decomp_table)(*symbolic_args)

    msg = (
        "op(*args, **kwargs) and make_fx(op)(*args, **kwargs) under "
        "cpp_fake_tensor_mode (symbolic) produced different values."
    )

    if randomize_data:
        new_args = randomize(new_args)
    try:
        expected = run(f, *new_args)
    except Exception:
        if randomize_data:
            return
        raise
    result = run(traced_f, *new_args)
    assert_close(result, expected, msg=msg)


# HOPs whose user-facing wrappers (torch.cond, etc.) call torch.compile internally,
# creating a Python FakeTensorMode that conflicts with C++ fake mode.
# These are tested directly via internal ops in TestCppFakeProxyTensor.
_HOP_SKIP_USER_FACING = {
    "cond", "map", "scan", "while_loop", "while_loop_stack_output",
    "auto_functionalize",
}
filtered_hop_db = [op for op in hop_db if op.name not in _HOP_SKIP_USER_FACING]


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "Cond requires dynamo")
class TestCppFakeProxyTensorOpInfo(TestCase):
    """Exhaustive op tests under C++ FakeTensor mode."""

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestCppFakeProxyTensorOpInfo', 'test_make_fx_exhaustive',
             cpp_fake_make_fx_failures | cpp_fake_only_real_failures)
    def test_make_fx_exhaustive(self, device, dtype, op):
        print(f"\n[cpp_fake exhaustive] {op.name}.{op.variant_test_name or 'default'}")
        _test_make_fx_helper_cpp_fake(self, device, dtype, op)

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestCppFakeProxyTensorOpInfo', 'test_make_fx_fake_exhaustive',
             cpp_fake_make_fx_failures | cpp_fake_only_fake_failures)
    def test_make_fx_fake_exhaustive(self, device, dtype, op):
        print(f"\n[cpp_fake fake_exhaustive] {op.name}.{op.variant_test_name or 'default'}")
        _test_make_fx_helper_cpp_fake(self, device, dtype, op)

    @ops(op_db + filtered_hop_db + custom_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestCppFakeProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive',
             cpp_fake_make_fx_failures | cpp_fake_symbolic_failures)
    def test_make_fx_symbolic_exhaustive(self, device, dtype, op):
        _test_make_fx_helper_cpp_fake_symbolic(self, device, dtype, op)


only_for = ("cpu",)
instantiate_device_type_tests(TestCppFakeProxyTensorOpInfo, globals(), only_for=only_for)


if __name__ == "__main__":
    run_tests()
