# Owner(s): ["oncall: pt2"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import unittest
import warnings
from collections.abc import Callable
from contextlib import ContextDecorator, ExitStack, nullcontext
from functools import partial, wraps
from typing import Any, Optional, Union
from unittest.mock import patch

from common_utils import (
    decorate,
    decorateForModules,
    saved_tensors_hooks_to_gm,
    skip,
    skipOps,
    xfail,
)

import torch
import torch._dynamo as torchdynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
from functorch import grad, jacrev, make_fx, vjp, vmap
from functorch.compile import (
    aot_function,
    aot_module,
    aot_module_simplified,
    compiled_function,
    compiled_module,
    default_decompositions,
    default_partition,
    get_aot_compilation_context,
    make_boxed_compiler,
    make_boxed_func,
    memory_efficient_fusion,
    min_cut_rematerialization_partition,
    nnc_jit,
    nop,
)
from functorch.experimental import control_flow
from torch._decomp import decomposition_table
from torch._dynamo.testing import normalize_gm
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._functorch.aot_autograd import (
    _aot_export_function,
    aot_export_joint_simple,
    aot_export_module,
    SerializableAOTDispatchCompiler,
)
from torch._higher_order_ops.out_dtype import out_dtype
from torch._inductor.codecache import compiled_fx_graph_hash
from torch._inductor.custom_graph_pass import CustomPartitionerFn
from torch._inductor.output_code import MockFXGraphCacheOutput
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensorMode
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, ShapeEnv
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.utils.rnn import PackedSequence
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    compare_equal_outs_and_grads,
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_MACOS,
    IS_WINDOWS,
    IS_X86,
    outs_and_grads,
    parametrize,
    run_tests,
    skipIfRocm,
    TEST_MKL,
    TestCase,
    xfail_inherited_tests,
    xfailIfTorchDynamo,
)
from torch.testing._internal.custom_tensor import ConstantExtraMetadataTensor
from torch.testing._internal.hop_db import hop_db
from torch.testing._internal.optests import (
    _test_aot_autograd_forwards_backwards_helper,
    aot_autograd_check,
)
from torch.testing._internal.subclasses import WrapperSubclass
from torch.testing._internal.two_tensor import TwoTensor, TwoTensorMode
from torch.utils._python_dispatch import TorchDispatchMode


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

USE_NETWORKX = False
try:
    import networkx  # noqa: F401

    USE_NETWORKX = True
except ImportError:
    warnings.warn("Some tests use networkx but it was not installed", UserWarning)

# NB: numpy is a testing dependency!


def amax_to_scale(
    amax: torch.Tensor,
    float8_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
):
    amax = amax.to(torch.float64)
    res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=1e-12)
    res = res.to(torch.float32)
    return res


# Must be at module level to use fx.wrap
@torch.fx.wrap
def _pack_fp8_with_scale_wrap(x):
    if not x.dtype.is_floating_point:
        return x

    amax = torch.max(torch.abs(x))
    scale = amax_to_scale(amax, torch.float8_e5m2)
    x_scaled = x.to(torch.float32) * scale
    x_fp8 = x_scaled.to(torch.float8_e5m2)
    return x.dtype, scale, x_fp8


@torch.fx.wrap
def _unpack_fp8_with_scale_wrap(x):
    if isinstance(x, torch.Tensor):
        return x

    dtype, scale, x_fp8 = x
    y = x_fp8.to(torch.float32) / scale
    return y.to(dtype)


@torch.fx.wrap
def _pack_fp8_wrap(x):
    if not x.dtype.is_floating_point:
        return x

    if type(x) is not torch.Tensor:
        # Check only during compilation
        # Test calls hooks to get reference output
        ctx = torch._functorch._aot_autograd.graph_compile._get_saved_tensor_hook_context()
        assert ctx["_fw_graph"] is not None
        assert ctx["_bw_graph"] is not None
        assert ctx["_node"] is not None

    return (x.dtype, x.to(torch.float8_e5m2))


@torch.fx.wrap
def _unpack_fp8_wrap(x):
    if isinstance(x, torch.Tensor):
        return x

    dtype, tensor = x
    if type(tensor) is not torch.Tensor:
        # Check only during compilation
        # Test calls hooks to get reference output
        ctx = torch._functorch._aot_autograd.graph_compile._get_saved_tensor_hook_context()
        assert ctx["_fw_graph"] is not None
        assert ctx["_bw_graph"] is not None
        assert ctx["_node"] is not None
    return tensor.to(dtype)


def pack_fp8(x):
    return _pack_fp8_wrap(x)


def unpack_fp8(packed):
    return _unpack_fp8_wrap(packed)


def pack_fp8_with_scale(x):
    return _pack_fp8_with_scale_wrap(x)


def unpack_fp8_with_scale(packed):
    return _unpack_fp8_with_scale_wrap(packed)


class AOTTestCase(TestCase):
    pass


class TestPythonKey(AOTTestCase):
    def test_make_fx(self, device):
        def f(x):
            return torch.sin(x)

        inp = torch.randn(3)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_grad(self, device):
        def f(x):
            return torch.sin(x).sum()

        inp = torch.randn(3)
        f = grad(f)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_scalar_device(self, device):
        def f(a, b):
            return a + b

        inps = [torch.randn(3, device=device), torch.tensor(5)]
        fx_f = make_fx(f)(*inps)
        self.assertEqual(fx_f(*inps), f(*inps))

    def test_make_fx_vmap(self, device):
        def f(x):
            return torch.sin(x)

        inp = torch.randn(5, 3)
        f = vmap(f)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(5, 3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_jacrev(self, device):
        def f(x):
            return x.sin().sum()

        inp = torch.randn(3)
        f = jacrev(jacrev(f))
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_vjp(self, device):
        def f(x):
            return torch.sin(x).sum()

        primals = torch.randn(3)
        _, vjp_fn = vjp(f, primals)
        cotangent = torch.randn(())
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    def test_make_fx_functionalize(self, device):
        from functorch.experimental import functionalize

        def fn(a):
            a = a * 2
            a.relu_()
            return a

        a = torch.randn(3, device=device)
        symbolic_gm = torch.fx.symbolic_trace(fn)
        includes_method_relu_ = any(
            str(n.target) == "relu_" for n in symbolic_gm.graph.nodes
        )
        self.assertTrue(includes_method_relu_)
        # Also verifies fix for https://github.com/pytorch/pytorch/issues/84570
        gm = make_fx(functionalize(symbolic_gm))(a)
        includes_aten_relu = any(
            n.target == torch.ops.aten.relu.default for n in gm.graph.nodes
        )
        self.assertTrue(includes_aten_relu)

    def test_make_fx_no_decompose(self, device):
        # FIXME
        return self.skipTest("error: maximum recursion reached")

        def f(x):
            return torch.tanh(x).sum()

        fx_f = make_fx(grad(f))(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}

        self.assertEqual(torch.ops.aten.tanh_backward in ops, True)

        fx_f = make_fx(grad(f), decomposition_table)(torch.randn(5))
        ops = {i.target for i in fx_f.graph.nodes}
        self.assertEqual(torch.ops.aten.tanh_backward in ops, False)

    def test_nnc_jit(self, device):
        def f(x):
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(3)
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_scalar(self, device):
        def f(x):
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(())
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_pytrees(self, device):
        def f(x):
            return [torch.sin(x[0])]

        jit_f = nnc_jit(f)

        inp = [torch.randn(3)]
        self.assertEqual(jit_f(inp), f(inp))

    def test_external_calls(self, device):
        def f(a, b):
            return torch.mv(a, b)

        jit_f = nnc_jit(f)
        inp = [torch.randn(3, 3), torch.randn(3)]
        self.assertEqual(jit_f(*inp), f(*inp))

    def test_nnc_passthrough(self, device):
        def f(x, y):
            return x + y, y

        inp = (torch.randn(3), torch.randn(3))
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

        def f(x):
            x["a"] = x["a"] * 2
            return x

        inp = ({"a": torch.randn(3), "b": torch.randn(3)},)
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_backward_trace(self, device):
        mod = torchvision.models.resnet18()

        def f(x):
            out = mod(x)
            out.sum().backward()
            return [a.grad for a in mod.parameters()]

        inp = torch.randn(3, 3, 250, 250, requires_grad=True)
        grads = f(inp)

        mod.zero_grad()
        mod(inp).sum().backward()
        grads2 = [a.grad for a in mod.parameters()]
        self.assertEqual(grads, grads2)


def get_base(t):
    return t._base if t._is_view() else t


def is_in_base(t, maybe_tensors):
    t_base = get_base(t)
    for maybe_tensor in maybe_tensors:
        if isinstance(maybe_tensor, torch.Tensor):
            if t_base is get_base(maybe_tensor):
                return True
    return False


def skipIfDynamoInput(reason):
    """
    Skip TestAOTAutograd if running with dynamo input
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if isinstance(self, TestAOTAutogradWithDynamo):
                self.skipTest(
                    f"Skipping {self._testMethodName} in TestAOTAutogradWithDynamo because {reason}"
                )
            else:
                func(self, *args, **kwargs)

        return wrapper

    return decorator


class TestAOTAutograd(AOTTestCase):
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: list[Optional[Callable]],
        decompositions: Optional[dict],
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        """
        Runs aot_autograd with the specified settings on f.
        """
        if isinstance(f, nn.Module):
            compiled_f = aot_module(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        else:
            compiled_f = aot_function(
                f,
                fw_compiler=make_boxed_compiler(
                    partial(extract_graph, graph_cell=fw_graph_cell)
                ),
                bw_compiler=nop,
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                dynamic=dynamic,
            )
        return compiled_f

    # test_mutation will:
    # - Ensure that inputs are non-leaves, so our graphs can mutate them
    # - try to mutate outputs of the graph (to ensure that autograd meta is set properly on outputs)
    @patch("functorch.compile.config.debug_assert", True)
    def verify_aot_autograd(
        self,
        f,
        inp_: Union[Callable, list[Any]],
        *,
        test_mutation: bool = False,
        keep_inp_mutations: bool = False,
        decompositions: Optional[dict] = None,
        dynamic: bool = False,
        # Only active when inp_ is Callable.
        # TODO: probably consolidate all tests to make inp a Callable.
        make_inputs_subclasses: bool = False,
    ):
        def make_inputs(inp_):
            # Some tests pass in a callable for inp, to generate the inputs
            # (useful if we want to generate complicated aliasing inputs)
            if isinstance(inp_, Callable):
                inp_callable = inp_
                # The callable should return a tuple of f_inputs, f_graph_inputs
                # (The idea is that we might want to compile a function with the graph inputs,
                # but test autograd backprop all the way through the actual inputs)
                with TwoTensorMode() if make_inputs_subclasses else nullcontext():
                    inp, graph_inps = inp_callable()
            else:
                inp = []
                # Our input clones need to mimic when inputs are duplicates of one another
                dupes_map = {}
                for i, x in enumerate(inp_):
                    if x in dupes_map:
                        x_dupe_idx = dupes_map[x]
                        inp.append(inp[x_dupe_idx])
                    else:
                        dupes_map[x] = i
                        if not isinstance(x, torch.Tensor):
                            x_copy = x
                        else:
                            x_copy = x.detach().clone().requires_grad_(x.requires_grad)
                            if x.requires_grad and not x.is_leaf:
                                x_copy = x_copy.clone()

                        inp.append(x_copy)

                if test_mutation:
                    # For graphs where we mutate inputs, need our test to make sure inputs aren't leaves
                    graph_inps = [x.add(1) for x in inp]
                else:
                    graph_inps = inp

            return inp, graph_inps

        def check_results(
            ref_results,
            test_results,
            ref_graph_inps,
            test_graph_inps,
            ref_inp,
            test_inp,
        ):
            ref_out, ref_grad = ref_results
            test_out, test_grad = test_results
            self.assertEqual(ref_grad, test_grad)
            if isinstance(ref_out, torch.Tensor):
                self.assertTrue(isinstance(test_out, torch.Tensor))
                ref_out, test_out = [ref_out], [test_out]
            for ref_o, test_o in zip(ref_out, test_out):
                if isinstance(ref_o, torch.Tensor):
                    self.assertEqual(ref_o.requires_grad, test_o.requires_grad)
                    self.assertEqual(ref_o.is_leaf, test_o.is_leaf)
                    ref_is_view_of_non_interm = is_in_base(
                        ref_o, ref_graph_inps
                    ) or is_in_base(ref_o, ref_out)
                    test_is_view_of_non_interm = is_in_base(
                        test_o, test_graph_inps
                    ) or is_in_base(test_o, test_out)
                    self.assertEqual(
                        ref_is_view_of_non_interm, test_is_view_of_non_interm
                    )
                    self.assertEqual(ref_o, test_o)
                    if test_mutation:
                        # This tests that autograd meta is set properly on the output we can
                        # mutate it.
                        ref_o.add_(2)
                        test_o.add_(2)
                        self.assertEqual(ref_o, test_o)
                        # Reverse the modification
                        ref_o.sub_(2)
                        test_o.sub_(2)
                        self.assertEqual(ref_o, test_o)
            for ref_i, test_i in zip(ref_inp, test_inp):
                if isinstance(ref_i, torch.Tensor):
                    self.assertEqual(ref_i.requires_grad, test_i.requires_grad)
                self.assertEqual(ref_i, test_i)

        for keep_input_mutations in [True] if keep_inp_mutations else [True, False]:
            inp, graph_inps = make_inputs(inp_)
            test_inp, test_graph_inps = make_inputs(inp_)
            fw_graph_cell = [None]
            compiled_f = self.run_autograd(
                f, fw_graph_cell, decompositions, keep_input_mutations, dynamic
            )
            ref_results = outs_and_grads(f, graph_inps, inp)
            test_results = outs_and_grads(compiled_f, test_graph_inps, test_inp)

            check_results(
                ref_results, test_results, graph_inps, test_graph_inps, inp, test_inp
            )
            if isinstance(self, TestAOTAutogradWithCache):
                # When testing with cache, run compiled_f a second time
                cached_inp, cached_graph_inps = make_inputs(inp_)
                cached_results = outs_and_grads(
                    compiled_f, cached_graph_inps, cached_inp
                )
                check_results(
                    ref_results,
                    cached_results,
                    graph_inps,
                    cached_graph_inps,
                    inp,
                    cached_inp,
                )

        return fw_graph_cell[0]

    def test_non_tensor_and_none_inputs(self):
        # int, None, Tensor
        def f(a, b, c):
            return a * c

        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=True)]
        self.verify_aot_autograd(f, inp)
        inp = [2, None, torch.ones(3, 3, dtype=torch.float32, requires_grad=False)]
        self.verify_aot_autograd(f, inp)

    def test_single_output(self):
        def f(a, b):
            return a + b

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output(self):
        def f(a, b):
            return a + b, a - b

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output_list(self):
        def f(a, b):
            return [a + b, a - b]

        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)
        inp = [torch.randn(3, 3, requires_grad=False), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    # Test for bug occurring at the intersection of fake tensors & functionalization.
    def test_squeeze_mutation(self):
        def f(a):
            b = a.clone().squeeze(-1)
            b.add_(1.0)
            return a + b

        inp = [torch.randn(3, 1, requires_grad=True)]
        self.verify_aot_autograd(f, inp, dynamic=True)
        inp = [torch.randn(3, 1, requires_grad=False)]
        self.verify_aot_autograd(f, inp, dynamic=True)

    def test_complex_linear(self):
        # https://github.com/pytorch/pytorch/issues/93424
        inp = [torch.randn(1, 10, 10, dtype=torch.complex64)]

        class F(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 10, dtype=torch.complex64)

            def forward(self, x):
                return self.linear(x).sum().abs()

        self.verify_aot_autograd(F(), inp)

    def test_embedding_bag_view_dynamic(self):
        # Backwards pass tries to wrap a sparse tensor in a FunctionalTensorWrapper;
        # test that this works even though the sparse tensor has no storage.

        class F(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(100, 8, sparse=True)

            def forward(self, x, y):
                return self.emb(x, y).view(-1)

        x = torch.arange(3)
        y = torch.arange(3)
        self.verify_aot_autograd(F(), [x, y], dynamic=False)
        self.verify_aot_autograd(F(), [x, y], dynamic=True)

    def test_input_mutation_simple(self):
        def f(a):
            a.mul_(2)
            return a * 3

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        # Things to note:
        # - the extra clone is because we need to pass the pre-mutated input to grad(),
        #   but autograd operates above functionalization so we need to manually clone.
        #   Hopefully backends can optimize this easily.
        # - The extra return arg is because the compiled forward returns (mutated inputs + outputs)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, 3)
    return (mul, mul_1)""",
        )

    def test_input_mutation_set__input_mutation(self):
        def f(a):
            b = torch.arange(9, dtype=a.dtype).reshape(3, 3)
            with torch.no_grad():
                a.set_(b)
            return a * b

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)

    def test_set__steals_view_chain(self):
        def f(a, b):
            a_ = a.mul(2)
            b_ = b.mul(2)
            b_slice = b_[1].view(3, 3)
            # a_clone should inherit the view chain from b_slice
            a_.set_(b_slice)
            # Also mutates b_,
            a_.view(-1).mul_(2)
            return a_ * b_slice

        inp = [
            torch.ones(3, 3, requires_grad=False),
            torch.zeros(3, 9, requires_grad=False),
        ]
        self.verify_aot_autograd(f, inp, keep_inp_mutations=True)

    def _compile_autocast(self, device, *, forward_autocast):
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x) -> Tensor")
            m.impl("foo", torch.clone, "CompositeExplicitAutograd")

            def autocast(x):
                return x + 1

            m.impl("foo", autocast, "AutocastCPU")
            m.impl("foo", autocast, "AutocastCUDA")

            foo = torch.ops.mylib.foo.default

            class Foo(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return foo(x)

                @staticmethod
                def backward(ctx, grad):
                    (x,) = ctx.saved_tensors
                    return grad * foo(x)

            def fn(x):
                with torch.amp.autocast(device, enabled=False):
                    return Foo.apply(x)

            x = torch.tensor(0.0, device=device, requires_grad=True)
            if forward_autocast:
                with (
                    torch.amp.autocast(device),
                    torch._dynamo.config.patch(recompile_limit=999),
                ):
                    out = torch.compile(fn, fullgraph=True, backend="aot_eager")(x)
            else:
                with torch._dynamo.config.patch(recompile_limit=999):
                    out = torch.compile(fn, fullgraph=True, backend="aot_eager")(x)
            (grad,) = torch.autograd.grad(out, x)
            return out, grad

    @torch._functorch.config.patch(backward_pass_autocast="same_as_forward")
    def test_backward_pass_autocast_on(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            out, grad = self._compile_autocast(device, forward_autocast=True)
            self.assertEqual(out, torch.zeros_like(out))
            self.assertEqual(grad, torch.ones_like(grad))

    @torch._functorch.config.patch(backward_pass_autocast="off")
    def test_backward_pass_autocast_off(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            out, grad = self._compile_autocast(device, forward_autocast=True)
            self.assertEqual(out, torch.zeros_like(out))
            self.assertEqual(grad, torch.zeros_like(grad))

    @torch._functorch.config.patch(backward_pass_autocast="off")
    def test_backward_pass_autocast_custom(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            with torch._functorch.config.patch(
                backward_pass_autocast=[{"device_type": device}]
            ):
                out, grad = self._compile_autocast(device, forward_autocast=False)
                self.assertEqual(out, torch.zeros_like(out))
                self.assertEqual(grad, torch.ones_like(grad))

    @skipIfDynamoInput(
        "Test doesn't make sense with dynamo, which changes order of mutations"
    )
    def test_set__and_data_mutation_good(self):
        def f(a, b):
            # The data mutation happens *after* the set_(). This is ok (see the graph below)
            with torch.no_grad():
                a.set_(b)
                b.mul_(2)
            return a + b

        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        inp = [
            torch.ones(3, 3, requires_grad=False),
            torch.zeros(3, 3, requires_grad=False),
        ]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        # Important things to note:
        # - "return a.set_(b)" desugars into "return b"
        # - Both a and b are recorded as experiencing mutations,
        #   which is why we see "b_updated" (output of the mul) twice in the graph outputs.
        #   a is recorded as both a data mutation and a metadata mutation (due to set_ swapping its storage).
        # - the runtime epilogue for a is "a.set_(mul)"
        # - the runtime epilogue for b is "b.copy_(mul)"
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_2, 2)
    add = torch.ops.aten.add.Tensor(mul, mul)
    set_ = torch.ops.aten.set_.source_Tensor(primals_1, mul);  primals_1 = set_ = None
    copy_ = torch.ops.aten.copy_.default(primals_2, mul);  primals_2 = mul = copy_ = None
    return (add,)""",
        )

    # This is a (hopefully) extremely rare case that is difficult to handle,
    # so we ban it.
    # https://github.com/pytorch/pytorch/issues/126236
    # https://github.com/pytorch/pytorch/pull/126113
    @xfailIfTorchDynamo
    def test_set__and_data_mutation_bad(self):
        def f(a):
            a_view = a.view(-1)
            tmp = torch.ones(3, 3, requires_grad=True)
            # Now, any mutations on either tmp
            # will be tracked as graph input mutations.
            with torch.no_grad():
                a.set_(tmp)
                # BAD: a_view is now detached from every graph input,
                # so we won't recognize that this caused an input mutation!
                a_view.mul_(2)
            return a + tmp

        inp = [torch.ones(3, 3, requires_grad=True)]
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    @skipIfDynamoInput(
        "Test doesn't make sense with dynamo, which changes order of mutations"
    )
    def test_set__not_allowed(self):
        def f(a, b):
            with torch.no_grad():
                a.set_(b)
            # Mutating a will change a's grad_fn, which requires us to replay the mutation outside of the graph.
            # We currently ban this today, when the input also received a set_() input mutation.
            a.mul_(2)
            return a + b

        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        with self.assertRaisesRegex(
            AssertionError, "but the input has other mutations that we cannot"
        ):
            self.verify_aot_autograd(
                f, inp, test_mutation=True, keep_inp_mutations=True
            )

    def test_input_mutation_set__nop(self):
        def f(a):
            b = torch.arange(9, dtype=a.dtype)
            a_old = torch.ops.aten.alias.default(a)
            with torch.no_grad():
                a.set_(b)
                a.set_(a_old)
            return a + b.reshape(3, 3)

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True, keep_inp_mutations=True)
        # Things to note:
        # - There are no set_() calls in the graph (we functionalize a.set_(b) into "b")
        # - There is only **1** graph output. We properly realized that the two set_() calls
        #   undo each other, and so effectively no inputs are mutated.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    arange = torch.ops.aten.arange.default(9, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    alias = torch.ops.aten.alias.default(primals_1);  primals_1 = None
    view = torch.ops.aten.view.default(arange, [3, 3]);  arange = None
    add = torch.ops.aten.add.Tensor(alias, view);  alias = view = None
    return (add,)""",
        )

    def test_input_mutation_simple_with_none_and_nontensor(self):
        # Tensor, None, int
        def f(a, b, c):
            return a * c

        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = [torch.ones(3, 3, requires_grad=req_grad), None, 3]
            out_ref = f(*inp)
            out_test = f_compiled(*inp)
            self.assertEqual(out_ref, out_test)

    # https://github.com/pytorch/pytorch/issues/93363
    def test_mutates_input_noncontiguous(self):
        def f(a):
            a.add_(1)
            return ()

        f_compiled = aot_function(f, nop)
        ref = torch.ones(4, requires_grad=True) + 0
        ref_view = ref[0::2]

        test = torch.ones(4, requires_grad=True) + 0
        test_view = test[0::2]

        out_ref = f(ref_view)  # noqa: F841
        out_test = f_compiled(test_view)  # noqa: F841
        self.assertEqual(ref, test)

    def test_input_mutation_modifies_autograd_meta_of_aliases(self):
        def f(a):
            a.mul_(2)
            out = a + 1
            return out.detach()

        x_ref = torch.ones(3, 3, requires_grad=True).clone()
        x_ref_view = x_ref.view(3, 3)

        x_test = torch.ones(3, 3, requires_grad=True).clone()
        x_test_view = x_test.view(3, 3)

        f_compiled = aot_function(f, nop, keep_inference_input_mutations=True)
        f(x_ref)
        f_compiled(x_test)
        # f will mutate aliases of the input, including its autograd metadata!
        # y.grad_fn is AsStridedBackward
        self.assertEqual(x_ref_view, x_test_view)
        self.assertEqual(x_ref_view._version, x_test_view._version)
        self.assertEqual(x_ref_view.grad_fn.__class__, x_test_view.grad_fn.__class__)
        # Test the actual gradients are correct
        (x_ref * x_ref_view).sum().backward()
        (x_test * x_test_view).sum().backward()
        self.assertEqual(x_ref.grad, x_test.grad)
        self.assertEqual(x_ref_view.grad, x_test_view.grad)

    def test_nested_subclasses(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)
        aaaa = TwoTensor(aa, aa2)
        out = f(aaaa)
        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, TwoTensor))
        self.assertTrue(isinstance(out.b, TwoTensor))
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        self.assertTrue(isinstance(out.b.b, torch.Tensor))

        out.sum().backward()
        self.assertTrue(isinstance(aaaa.grad, TwoTensor))
        self.assertTrue(isinstance(aaaa.grad.a, TwoTensor))
        self.assertTrue(isinstance(aaaa.grad.b, TwoTensor))

    def test_nested_subclasses_non_nested_grad(self):
        @torch.compile(backend="aot_eager")
        def f(x):
            return x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        new_aa = TwoTensor(a3, a4)
        aa = TwoTensor(a, a2)

        aa2 = aa.detach().clone().requires_grad_()
        aaaa = TwoTensor(aa, aa2)
        out = f(new_aa)
        new_out = out + aaaa
        with self.assertRaisesRegex(
            RuntimeError,
            """
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
""",  # noqa: F541
        ):
            new_out.sum().backward()

    def test_nested_subclasses_non_homogenous(self):
        def f(x):
            x_elem = x.elem
            x_metadata = x.constant_attribute
            return x_metadata * x_elem * x.sin().cos()

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)
        custom_aa = ConstantExtraMetadataTensor(aa)
        custom_aa.constant_attribute = 6
        custom_aa2 = ConstantExtraMetadataTensor(aa2)
        custom_aa2.constant_attribute = 6

        out_eager = f(custom_aa)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_aa2)

        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.b, ConstantExtraMetadataTensor))
        self.assertTrue(torch.allclose(out_eager, out))

        out_eager.sum().backward()
        out.sum().backward()

        self.assertTrue(torch.allclose(custom_aa.grad, custom_aa2.grad))
        self.assertTrue(isinstance(custom_aa2.grad, TwoTensor))
        self.assertTrue(isinstance(custom_aa2.grad.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(custom_aa2.grad.b, ConstantExtraMetadataTensor))

    def test_subclasses_mixed(self):
        def f(x, y):
            x_metadata = x.constant_attribute
            out_a = x_metadata * x * y.a
            out_b = x * y.a * y.b
            return TwoTensor(out_a, out_b)

        a = torch.ones(4, requires_grad=False)
        a2 = a.clone()
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 5
        custom_a2 = ConstantExtraMetadataTensor(a2)
        custom_a2.constant_attribute = 5

        b = torch.ones(4, requires_grad=False)
        b2 = b.clone()
        b3 = b.clone()
        b4 = b.clone()
        bb = TwoTensor(b, b2)
        bb2 = TwoTensor(b3, b4)

        out_eager = f(custom_a, bb)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_a2, bb2)

        self.assertTrue(torch.allclose(out_eager, out))
        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.b, ConstantExtraMetadataTensor))

    def test_subclasses_mixed_mode(self):
        def f(x):
            return x.sin().cos()

        class AddConstantMetadataMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if ConstantExtraMetadataTensor not in types:
                    out = ConstantExtraMetadataTensor(out)
                    out.constant_attribute = 5
                return out

        a = torch.ones(4, requires_grad=True)
        a2 = a.detach().clone().requires_grad_()
        a3 = a.detach().clone().requires_grad_()
        a4 = a.detach().clone().requires_grad_()
        aa = TwoTensor(a, a2)
        aa2 = TwoTensor(a3, a4)

        with AddConstantMetadataMode():
            out_eager = f(aa)

        compiled_f = torch.compile(f, backend="aot_eager")

        with AddConstantMetadataMode():
            out = compiled_f(aa2)

        self.assertTrue(isinstance(out, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(out.elem, TwoTensor))
        self.assertTrue(torch.allclose(out_eager, out))

        out_eager.sum().backward()
        out.sum().backward()

        self.assertTrue(torch.allclose(aa.grad, aa2.grad))
        self.assertTrue(isinstance(aa2.grad, ConstantExtraMetadataTensor))
        self.assertTrue(isinstance(aa2.grad.elem, TwoTensor))

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    def test_custom_tensor_metadata(self):
        def f(x):
            x_elem = x.elem
            x_elem_elem = x_elem.elem
            x_elem_metadata = x_elem.constant_attribute
            return x * x_elem * x_elem_elem * x_elem_metadata

        a = torch.ones(4, requires_grad=True)
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 6
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        custom_aa.constant_attribute = 4

        custom_aa_compile = custom_aa.detach().clone().requires_grad_()
        custom_aa_compile.elem.constant_attribute = 6
        out_eager = f(custom_aa)

        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(custom_aa_compile)

        self.assertTrue(torch.allclose(out_eager, out))

        out.sum().backward()

        self.assertTrue(isinstance(custom_aa_compile.grad, ConstantExtraMetadataTensor))
        self.assertTrue(
            isinstance(custom_aa_compile.grad.elem, ConstantExtraMetadataTensor)
        )

    def test_nested_subclasses_complicated_inps(self):
        def f(x, y, z):
            temp = x + y
            temp_plain = x.a + y.b
            res = temp.sum() + temp_plain.sum()
            return x.sin().cos() + res

        x = torch.ones(4, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        xx = TwoTensor(x, x2)
        xx2 = xx.detach().clone().requires_grad_()

        x_nested = TwoTensor(xx, xx2)
        x_nested_compile = x_nested.detach().clone().requires_grad_()

        y_nested = x_nested.detach().clone().requires_grad_()
        y_nested_compile = y_nested.detach().clone().requires_grad_()

        z = x.detach().clone().requires_grad_()
        z_compile = z.detach().clone().requires_grad_()

        out_eager = f(x_nested, y_nested, z)
        compiled_f = torch.compile(f, backend="aot_eager")
        out = compiled_f(x_nested_compile, y_nested_compile, z_compile)
        self.assertTrue(torch.allclose(out_eager, out))

        self.assertTrue(isinstance(out, TwoTensor))
        self.assertTrue(isinstance(out.a, TwoTensor))
        self.assertTrue(isinstance(out.b, TwoTensor))
        self.assertTrue(isinstance(out.a.a, torch.Tensor))
        self.assertTrue(isinstance(out.a.b, torch.Tensor))
        self.assertTrue(isinstance(out.b.a, torch.Tensor))
        self.assertTrue(isinstance(out.b.b, torch.Tensor))

        out.sum().backward()
        out_eager.sum().backward()

        self.assertTrue(isinstance(x_nested_compile.grad, TwoTensor))
        self.assertTrue(isinstance(x_nested_compile.grad.a, TwoTensor))
        self.assertTrue(isinstance(x_nested_compile.grad.b, TwoTensor))

        self.assertTrue(isinstance(y_nested_compile.grad, TwoTensor))
        self.assertTrue(isinstance(y_nested_compile.grad.a, TwoTensor))
        self.assertTrue(isinstance(y_nested_compile.grad.b, TwoTensor))

        self.assertTrue(torch.allclose(x_nested_compile.grad.a.a, x_nested.grad.a.a))
        self.assertTrue(torch.allclose(x_nested_compile.grad.a.b, x_nested.grad.a.b))
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.a, y_nested.grad.a.a))
        self.assertTrue(torch.allclose(y_nested_compile.grad.a.b, y_nested.grad.a.b))

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    def test_nested_subclasses_complicated_inps_mixed(self):
        def f(x, y):
            y_elem = y.elem
            y_elem_elem = y_elem.elem
            y_elem_metadata = y_elem.constant_attribute
            return y * y_elem * y_elem_elem * y_elem_metadata + x

        x = torch.ones(4, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        xx = TwoTensor(x, x2)
        xx2 = xx.detach().clone().requires_grad_()

        x_nested = TwoTensor(xx, xx2)
        x_nested_compile = x_nested.detach().clone().requires_grad_()

        a = torch.ones(4, requires_grad=True)
        custom_a = ConstantExtraMetadataTensor(a)
        custom_a.constant_attribute = 6
        custom_aa = ConstantExtraMetadataTensor(custom_a)
        custom_aa.constant_attribute = 4

        custom_aa_compile = custom_aa.detach().clone().requires_grad_()
        custom_aa_compile.constant_attribute = 4
        custom_aa_compile.elem.constant_attribute = 6

        compiled_f = torch.compile(f, backend="aot_eager")
        out_eager = f(x_nested, custom_aa)
        out = compiled_f(x_nested_compile, custom_aa_compile)
        self.assertTrue(torch.allclose(out_eager, out))

        out.sum().backward()
        out_eager.sum().backward()

        self.assertTrue(torch.allclose(x_nested_compile.grad, x_nested.grad))
        self.assertTrue(torch.allclose(custom_aa_compile.grad, custom_aa.grad))

    def test_composite_impl_compile(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, a):
                return self.linear(a)

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(Foo(), inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    t = torch.ops.aten.t.default(primals_1);  primals_1 = None
    addmm = torch.ops.aten.addmm.default(primals_2, primals_3, t);  primals_2 = None
    return (addmm, primals_3, t)""",
        )

        with torch.inference_mode():
            fw_graph = self.verify_aot_autograd(Foo(), inp, test_mutation=True)
            inp = [torch.ones(3, 3, requires_grad=False)]
            self.assertExpectedInline(
                fw_graph.code.strip(),
                """\
def forward(self, arg0_1, arg1_1, arg2_1):
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    addmm = torch.ops.aten.addmm.default(arg1_1, arg2_1, t);  arg1_1 = arg2_1 = t = None
    return (addmm,)""",
            )

    def test_outputs_are_aliased(self):
        # Tensor, None, int
        def f(a):
            b = a.mul(2)
            c = b.view(-1)
            return b, c

        f_compiled = aot_function(f, nop)
        for req_grad in [True, False]:
            inp = torch.ones(3, requires_grad=req_grad)
            out_ref = f(inp)
            out_test = f_compiled(inp)
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])
            # Try mutating one of the outputs, which is aliased.
            out_ref[0].mul_(3)
            out_test[0].mul_(3)
            # Assert that the aliasing relationship was preserved
            self.assertEqual(out_ref[0], out_test[0])
            self.assertEqual(out_ref[1], out_test[1])

    def test_input_mutation_is_output(self):
        def f(a):
            a.mul_(2)
            return a

        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    return (mul, mul)""",
        )

    def test_input_mutation_multiple(self):
        def f(a, b, c):
            a.mul_(2)
            c.mul_(2)
            return a + b + c

        def create_inp(req_grad):
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None
    add = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
    add_1 = torch.ops.aten.add.Tensor(add, mul_1);  add = None
    return (mul, mul_1, add_1)""",
        )

    def test_input_mutation_return(self):
        def f(a, b):
            return torch.sin(a, out=b)

        inp = [torch.randn(3, 3), torch.ones(3, 3)]

        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    copy_ = torch.ops.aten.copy_.default(arg1_1, sin);  arg1_1 = sin = None
    return (copy_,)""",
        )

    def test_input_mutation_metadata(self):
        def f(a, b):
            a.transpose_(1, 0)
            return a + b

        def create_inp(req_grad):
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)

    @parametrize("backend", ["aot_eager", "inductor"])
    @parametrize("view_replay_for_aliased_outputs", [False, True])
    @parametrize("dynamic_shapes", [False, True])
    def test_alias_of_intermediate_detach(
        self, backend, view_replay_for_aliased_outputs, dynamic_shapes
    ):
        with patch(
            "torch._functorch.config.view_replay_for_aliased_outputs",
            view_replay_for_aliased_outputs,
        ):

            def fn(x):
                x = x + 1
                a = x.transpose(0, 1)
                return a.detach(), a

            def inp_fn():
                t = torch.ones(3, 3, requires_grad=True)
                if dynamic_shapes:
                    torch._dynamo.mark_dynamic(t, 0)
                    torch._dynamo.mark_dynamic(t, 1)
                return t

            x_ref = inp_fn()
            y_ref = fn(x_ref)

            x = inp_fn()
            y = torch.compile(fn, backend=backend, fullgraph=True)(x)
            self.assertEqual(y_ref, y)
            y0, y1 = y
            self.assertFalse(y0.requires_grad)
            self.assertTrue(y1.requires_grad)
            # Check that detach and diff view points to the same intermediate tensor storage
            self.assertEqual(y0.data_ptr(), y1.data_ptr())
            self.assertTrue(y1._is_view())

            sum(y_ref).sum().backward()
            sum(y).sum().backward()
            self.assertEqual(x_ref.grad, x.grad)

    def test_input_mutation_storage_resize_up(self):
        def f(a):
            torch.ops.inductor.resize_storage_bytes_(a, 32)
            # float32, 4 bytes per element, 32 bytes == 8 elements
            with torch.no_grad():
                a.copy_(torch.ones(8))
            return a + 1

        inp = torch.zeros(8, requires_grad=True)
        # Input starts with zero-size-storage
        inp.untyped_storage().resize_(0)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)
        # Final functionalized graph has two mutation ops:
        # (1) a resize_() to resize input tensor up
        # (2) a copy_() to fill in the resized input with valid data
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1):
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 32);  resize_storage_bytes_ = None
    ones = torch.ops.aten.ones.default([8], device = device(type='cpu'), pin_memory = False)
    copy = torch.ops.aten.copy.default(primals_1, ones);  ones = None
    add = torch.ops.aten.add.Tensor(copy, 1)
    copy_ = torch.ops.aten.copy_.default(primals_1, copy);  primals_1 = copy = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_storage_resize_down(self):
        def f(a):
            out = a.sin()
            torch.ops.inductor.resize_storage_bytes_(a, 0)
            return out

        inp = torch.zeros(8, requires_grad=True)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)
        # Final functionalized graph has one mutation ops:
        # (1) a resize_() to resize input tensor down
        # Even though there was technically a "data mutation" on the input (from a.copy_()),
        # We don't include it in the graph since the final input size has zero storage
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1):
    sin = torch.ops.aten.sin.default(primals_1)
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(primals_1, 0);  resize_storage_bytes_ = None
    return (sin, primals_1)""",
        )

    #     def test_input_mutation_storage_resize_up_down(self):
    #         def f(a):
    #             torch.ops.inductor.resize_storage_bytes_(a, 32)
    #             # float32, 4 bytes per element, 32 bytes == 8 elements
    #             with torch.no_grad():
    #                 a.copy_(torch.ones(8))
    #             out = a.sin()
    #             torch.ops.inductor.resize_storage_bytes_(a, 0)
    #             return out

    #         inp = torch.zeros(8, requires_grad=True)
    #         # Input starts with zero-size-storage
    #         inp.untyped_storage().resize_(0)

    #         fw_graph_cell = [None]
    #         compiled_f = aot_function(
    #             f,
    #             fw_compiler=make_boxed_compiler(
    #                 partial(extract_graph, graph_cell=fw_graph_cell)
    #             ),
    #             bw_compiler=nop,
    #             decompositions={},
    #             keep_inference_input_mutations=True,
    #             dynamic=False,
    #         )
    #         out = compiled_f(inp)
    #         # Final graph has two interesting properties:
    #         # (1) no resizes in the functional graph, since the two resizes cancel out
    #         #     and the final size is zero
    #         # (2) no copy_ in the functional graph, even though we copied data into the input,
    #         #     because the input has no storage at the end of graph execution (so no data to copy)
    #         self.assertExpectedInline(
    #             fw_graph_cell[0].code.strip(),
    #             """\
    # def forward(self, primals_1):
    #     ones = torch.ops.aten.ones.default([8], device = device(type='cpu'), pin_memory = False)
    #     copy = torch.ops.aten.copy.default(primals_1, ones);  primals_1 = ones = None
    #     sin = torch.ops.aten.sin.default(copy)
    #     return [sin, copy]""",
    #         )

    # skipped after confirming with @yf225 and @bdhirsh
    @unittest.skipIf(
        True,
        "using set_ unsafely and PT2 FSDP2 no longer uses set_ as used in this test",
    )
    def test_input_mutation_storage_resize_down_and_set_(self):
        # Meant to mimic ppFSDP
        class TracableCreateParameter(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor, placeholder):
                assert not tensor.requires_grad
                return placeholder.set_(tensor)

            @staticmethod
            def backward(ctx, grad):
                return None, grad  # grad flows to placeholder

        def f(dummy_param, param_shard):
            # simulate allgather
            with torch.no_grad():
                allgather_param = torch.cat([param_shard, param_shard])
            # simulate propagating grad state through dummy param, using data of allgather param
            dummy_param_with_grad_state = TracableCreateParameter.apply(  # noqa: F841
                allgather_param, dummy_param
            )
            out = dummy_param.sin()
            # Resize out dummy param, which now has the allgather data
            torch.ops.inductor.resize_storage_bytes_(dummy_param, 0)
            return out

        # Simulates the local shard of our param
        param_shard = torch.zeros(8, requires_grad=True)
        # The dummy, zero-sized allgathered param that autograd will actually compute gradients on
        dummy_param = torch.zeros(16, requires_grad=True)
        dummy_param.untyped_storage().resize_(0)

        fw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(dummy_param, param_shard)
        # Important stuff to point out:
        # (1) We save cat for backward (input to the sin()).
        #     While the original code was dummy_param.sin(),
        #     dummy_param actually contains the `cat` tensor due to the set_() call
        # (2) We emit a cat.resize_storage_(0) in the graph.
        #     After the set_(), cat is the actually data of dummy_param, which is what we call resize_() on
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    cat = torch.ops.aten.cat.default([primals_2, primals_2]);  primals_2 = None
    sin = torch.ops.aten.sin.default(cat)
    resize_storage_bytes_ = torch.ops.inductor.resize_storage_bytes_.default(cat, 0);  resize_storage_bytes_ = None
    set_ = torch.ops.aten.set_.source_Tensor(primals_1, cat);  primals_1 = set_ = None
    return (sin, cat)""",
        )

    def test_input_mutation_storage_resize_before_set_(self):
        def f(a):
            with torch.no_grad():
                torch.ops.inductor.resize_storage_bytes_(a, 0)
                a.set_(torch.ones(2))

        inp = torch.zeros(8, requires_grad=True)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            decompositions={},
            keep_inference_input_mutations=True,
            dynamic=False,
        )
        compiled_f(inp)

    # def test_input_mutation_storage_resize_not_supported(self):
    #     def f(a):
    #         a.mul_(2)
    #         torch.ops.inductor.resize_storage_bytes_(a, 0)
    #         return a

    #     inp = torch.zeros(8, requires_grad=True)

    #     with self.assertRaisesRegex(
    #         AssertionError, "the input has other mutations that we cannot"
    #     ):
    #         compiled_f = aot_function(
    #             f,
    #             fw_compiler=nop,
    #             bw_compiler=nop,
    #             decompositions={},
    #             keep_inference_input_mutations=True,
    #             dynamic=False,
    #         )
    #         out = compiled_f(inp)

    def test_input_output_aliase_custom_autograd_function(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gx):
                return gx * 0.5

        def f(x):
            return Foo.apply(x)

        inp = [torch.ones(2, 2, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)

    def test_input_mutation_requires_grad_detach(self):
        # Here, "a" requires grad, and gets mutated, so we append a copy_() to the end of the graph.
        # Its mutation doesn't take part in autograd though, because we mutated a detach'd view.
        # Need to make sure that this copy_() doesn't error, and doesn't participate in autograd either.
        def f(a):
            a.detach().mul_(2)
            return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=False)
        inp = [torch.ones(4, requires_grad=True)]
        # test_mutation=True will first do some compute on inp, so it is no longer an autograd leaf
        # by the time it becomes a graph input. Good to test both cases.
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_hidden_from_autograd_aliasing(self):
        def f(a):
            a_alias = a.view(-1)
            with torch.no_grad():
                a_alias.mul_(2)
            return a + 1

        inp = [torch.ones(4, requires_grad=True)]
        # The important bit: we detected that the input mutation is safe
        # to include **inside** the graph, since it was under no_grad
        # (so all we need to do is use mark_dirty() on the input to bump the VC)
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    view = torch.ops.aten.view.default(primals_1, [-1])
    mul = torch.ops.aten.mul.Tensor(view, 2);  view = None
    view_1 = torch.ops.aten.view.default(mul, [4]);  mul = None
    add = torch.ops.aten.add.Tensor(view_1, 1)
    copy_ = torch.ops.aten.copy_.default(primals_1, view_1);  primals_1 = view_1 = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad(self):
        def f(a):
            with torch.no_grad():
                a.mul_(2)
            return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )
        # Even though the input requires_grad, we expect the keep the input mutation in the graph
        # (Even though this is a training graph!)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 3)
    copy_ = torch.ops.aten.copy_.default(primals_1, mul);  primals_1 = mul = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad_inference_graph(self):
        def f(a):
            with torch.no_grad():
                a.mul_(2)
                return a + 3

        inp = [torch.ones(4, requires_grad=True)]
        # Even though the input requires_grad, we expect the keep the input mutation in the graph
        fw_graph = self.verify_aot_autograd(
            f, inp, test_mutation=True, keep_inp_mutations=True
        )

        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 2)
    add = torch.ops.aten.add.Tensor(mul, 3)
    copy_ = torch.ops.aten.copy_.default(arg0_1, mul);  arg0_1 = mul = copy_ = None
    return (add,)""",
        )

    def test_input_mutation_requires_grad_no_grad_detach_mixed(self):
        # Perform a mix of mutations on a:
        # 1 normal, 1 in no_grad, 1 on a detach'd tensor.
        # Only the first should participate in gradient computation.
        def f(a):
            a.detach().mul_(2)
            a.mul_(3)
            with torch.no_grad():
                a.mul_(4)
            return a + 5

        inp = [torch.ones(4, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_metadata2(self):
        def f(a):
            a.transpose_(1, 0)
            a.mul_(2)
            return a + 1

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_mutation_batchnorm(self):
        def f(inpt, weight, bias, running_mean, running_var):
            # This is additionally a good test, because the input tensors that we mutate
            # are *also* saved for backwards.
            # This tests that what we save for the backward is actually cloned inputs,
            # and not the original inputs that got mutated.
            return torch._native_batch_norm_legit(
                inpt, weight, bias, running_mean, running_var, True, 0.5, 1e-5
            )

        def create_inp(req_grad):
            return [
                torch.ones(2, 5, 5, 5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5, requires_grad=req_grad),
                torch.ones(5),
                torch.ones(5),
            ]

        from torch._decomp import get_decompositions

        # This simulates what inductor does (running the fw + bw decompositions)
        decompositions = get_decompositions(
            [
                torch.ops.aten._native_batch_norm_legit_functional,
                torch.ops.aten.native_batch_norm_backward,
            ]
        )
        self.verify_aot_autograd(
            f, create_inp(True), test_mutation=True, decompositions=decompositions
        )
        self.verify_aot_autograd(
            f, create_inp(False), test_mutation=True, decompositions=decompositions
        )

    def test_batchnorm_inference(self):
        inp = [
            torch.ones(2, 5, 5, 5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5),
            torch.ones(5),
        ]

        m = torch.nn.BatchNorm2d(4, 4)
        m.eval()
        fw_graph_cell = [None]
        inp = torch.ones(4, 4, 4, 4)
        fw_graph_cell = [None]
        compiled_m = aot_module(
            m,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp = torch.ones(4, 4, 4, 4)
        with torch.no_grad():
            compiled_m(inp)
        # expectation: there are no copy_() calls in the decomposed batch norm when running under training=False (eval mode)
        code = fw_graph_cell[0].code.strip()
        self.assertTrue("copy_" not in str(code))

    def test_input_output_view_simple(self):
        def f(a):
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # Outputs that alias inputs are pulled out of the graph entirely, so we don't compile anything here
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1):
    view = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
    return (view,)""",
        )

    def test_input_output_view_mutate_multiple(self):
        def f(a, b, c):
            a.mul_(2)
            c.mul_(3)
            return b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # The original function returned two outputs, both of which aliased inputs.
        # We expect two outputs in the functional graph, a_updated and c_updated.
        # The actual aliased outputs themselves aren't in the compiled forward graph;
        # Instead, they're generated outside of  the graph.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 3);  clone_1 = None
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    view_2 = torch.ops.aten.view.default(mul_1, [2, 2])
    return (mul, mul_1, view, view_2)""",
        )

    def test_input_output_view_metadata_mutate_multiple(self):
        def f(a, b, c):
            b.mul_(3)
            c.t_()
            return a.view(2, 2), b.view(2, 2), c.view(2, 2)

        def create_inp(req_grad):
            return [
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        # Important thing to check here: of the three inputs:
        # Only the b.mul_(3) should show up in the graph (we functionalize it and return it).
        # Everything else that does not show up in the graph includes:
        # - The metadata mutation on c (we do it outside the graph)
        # - All 3 original fw outputs, which are aliases of inputs (we regenerate them outside of the graph)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_2);  primals_2 = None
    view = torch.ops.aten.view.default(primals_3, [2, 2]);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 3);  clone = None
    t = torch.ops.aten.t.default(view);  view = None
    view_1 = torch.ops.aten.view.default(primals_1, [2, 2]);  primals_1 = None
    view_3 = torch.ops.aten.view.default(t, [2, 2])
    view_4 = torch.ops.aten.view.default(mul, [2, 2])
    return (mul, t, view_1, view_4, view_3)""",
        )

    def test_input_mutation_and_output_view(self):
        def f(a):
            a.add_(1)
            return a.view(-1)

        inp = [torch.ones(2, 2, requires_grad=False).add(1)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 2, requires_grad=True).add(1)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # Here, total # of outputs is 1 because:
        # - num_mutated_inps = 1 (a_updated)
        # - num_fw_outputs = 0 (the output is an alias of the input, so we move it outside the compiled fw)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    view_1 = torch.ops.aten.view.default(add, [-1])
    return (add, view_1)""",
        )

    def test_input_mutation_output_view_multiple(self):
        def f(a, b, c, d):
            b.transpose_(1, 0)
            c.add_(1)
            return d + 1, b.diagonal(), a + c

        def create_inp(req_grad):
            return [
                torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
                .view(2, 2)
                .add(1),
                torch.arange(4, requires_grad=req_grad, dtype=torch.float32)
                .view(2, 2)
                .add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
                torch.ones(2, 2, requires_grad=req_grad).add(1),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    clone = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(primals_4, 1);  primals_4 = None
    diagonal = torch.ops.aten.diagonal.default(transpose)
    add_2 = torch.ops.aten.add.Tensor(primals_1, add);  primals_1 = None
    return (transpose, add, add_1, diagonal, add_2)""",
        )

    def test_output_aliases_intermediate_single(self):
        def f(a):
            out = torch.mul(a, 3)
            return out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # In AOTAutograd, we are obligated to make the compiled forward directly return `out`,
        # and reconstruct `out.view(-1)` as a fresh output.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1]);  mul = None
    return (view,)""",
        )

    def test_output_aliases_input_multi_output_view_should_raise_autograd_error(self):
        def f1(a):
            return list(a.unbind(0))

        f1_compiled = aot_function(f1, nop)

        inp1 = torch.ones(3, 3, requires_grad=True).clone()
        inp2 = torch.ones(3, 3, requires_grad=True).clone()
        inp3 = torch.ones(3, 3, requires_grad=True).clone()

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            out_test1 = f1_compiled(inp1)
            # This raises a runtime error from autograd in eager mode
            out_test1[0].mul_(2)

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            out_test2 = f1_compiled(inp2)
            inp2.mul_(2)
            # In eager mode, if we mutate a tensor, any multi-output-view aliases
            # get their grad_fn replaced with error nodes, so accessing grad_fn should error
            out_test2[0].grad_fn

        with self.assertRaisesRegex(
            RuntimeError, "Such functions do not allow the output views"
        ):
            f1_compiled(inp3)
            out_test1[0].detach().mul_(2)
            # The above case also applies to detached aliases (they turn the multi-output-view
            # alias's grad_fns into error nodes)
            out_test2[0].grad_fn

    def test_output_aliases_input_multi_output_view(self):
        # All aliased outs are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f1(a):
            return list(a.unbind(0))

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)

        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # Several of the outputs are from multi-output views.
        # However: they are part of the same alias set as "a", and "a.view(out.shape)",
        # which are both user-visible.
        # AOTAutograd will not try to be smart here and hide the aliasing relationships from autograd.
        # Instead, it will perform its "output aliases input" logic, and regenerate all aliases.
        def f3(a):
            return *list(a.unbind(0)), a.view(a.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)

        inp_ref_clone = inp_ref.clone()
        inp_clone = inp.clone()
        out_ref = f3(inp_ref_clone)
        out_test = f3_compiled(inp_clone)
        self.assertTrue(all("UnbindBackward" in str(o.grad_fn) for o in out_test[:3]))

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        # Also mutate the input, which should affect the aliased output.
        inp_ref_clone.view(-1).mul_(3)
        inp_clone.view(-1).mul_(3)
        # Do backward
        (inp_ref + out_ref[-1]).sum().backward()
        (inp + out_test[-1]).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_multi_output_view(self):
        # All aliased outs are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f1(a):
            out = torch.mul(a, 3)
            return list(out.unbind(0))

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f1_compiled = aot_function(f1, nop)

        out_ref = f1(inp_ref)
        out_test = f1_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        sum(out_ref).sum().backward()
        sum(out_test).sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # All aliased outs but one are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f2(a):
            out = torch.mul(a, 3)
            return *list(out.unbind(0)), out

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f2_compiled = aot_function(f2, nop)

        out_ref = f2(inp_ref)
        out_test = f2_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # All aliased outs but one are from multi-output views, so AOTAutograd will hide the aliasing from autograd.
        def f3(a):
            out = torch.mul(a, 3)
            return *list(out.unbind(0)), out.view(out.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f3_compiled = aot_function(f3, nop)

        out_ref = f3(inp_ref)
        out_test = f3_compiled(inp)
        # Assert that we get CompiledFunctionBackward in the backward graph,
        # and not AsStridedBackward. No view-regeneration necessary for this mult-output view case.
        # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
        self.assertTrue(
            all("CompiledFunctionBackward" in str(o.grad_fn) for o in out_test)
        )

        # The last output is not from a multi-output view, so autograd will let us mutate it.
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)
        out_ref[-1].sum().backward()
        out_test[-1].sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

        # There are 5 outputs that all alias each other.
        # 3 of them come from multi-output views, but the other 3 are "ordinary" aliases.
        # Therefore, AOTAutograd will not attempt the multi-output-view optimization,
        # and apply the intermediate_base logic to all aliases.
        # (In theory we could probably get AOTAutograd to only apply the intermediate base
        # logic to the last 2 outputs and not the first 3. We should probably
        # just do the graph partitioning defined in this doc instead though).
        # https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit
        def f4(a):
            out = torch.mul(a, 3)
            # also return the graph intermediate directly,
            # which will force AOTAutograd to do the "intermediate base" logic.
            # (Why? The user can mutate "out", which should change the autograd metadata
            #  of the other aliased outputs)
            return *list(out.unbind(0)), out, out.view(out.shape)

        inp = torch.ones(3, 3, requires_grad=True)
        inp_ref = torch.ones(3, 3, requires_grad=True)
        f4_compiled = aot_function(f4, nop)

        out_ref = f4(inp_ref)
        out_test = f4_compiled(inp)
        # Mutate the last output of f4 (autograd will allow this, since it is not a multi-output view,
        # as long as *only* the non-multi-output views participate in the backward)
        # Note: We could probably try to hide **only** the multi-output views from autograd here
        # and only do the intermediate base logic for the last two aliases.
        # Longer term solution of graph partitioning is probably cleaner though (see the note).
        out_ref[-1].mul_(2)
        out_test[-1].mul_(2)

        out_ref_sum = out_ref[-1] + out_ref[-2]
        out_test_sum = out_test[-1] + out_test[-2]
        out_ref_sum.sum().backward()
        out_test_sum.sum().backward()
        self.assertEqual(inp_ref.grad, inp.grad)

    def test_output_aliases_intermediate_mutation_linear(self):
        def f(x):
            return (x + 1).view(-1)

        inp = [torch.ones(3, 3, requires_grad=True)]
        # use inductor's decomps (which will e.g. turn _unsafe_view() into view())
        from torch._inductor.decomposition import decompositions

        f_compiled = aot_function(f, nop, decompositions=decompositions)

        out_ref = f(*inp)
        out_test = f_compiled(*inp)

        out_ref.mul_(2)
        out_test.mul_(2)
        self.assertEqual(out_ref, out_test)

    def test_output_aliases_intermediate_no_grad(self):
        def f(a, b):
            out = torch.mul(a, 3)
            # First output is an alias of an intermediate that doesn't require grad
            return out.view(-1), b.add(1)

        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3), torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        # important bit: we don't bother generating an intermediate base as an output in the graph,
        # because the intermediate base itself didn't require gradients.
        # (the only problematic case is when both the base and the aliasesed output require gradients).
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1]);  mul = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    return (view, add)""",
        )

    def test_output_aliases_intermediate_returned_multiple_times(self):
        def f(a):
            out = torch.mul(a, 3)
            out_view = out.view(-1)
            return out, out_view, out

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_multiple(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate these two output views in the epilogue.
            return out.view(-1), out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    view_1 = torch.ops.aten.view.default(mul, [-1])
    return (view, view_1, mul)""",
        )

    def test_output_aliases_intermediate_and_returned(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out.view(-1), out

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    return (view, mul)""",
        )

    def test_output_aliases_intermediate_and_returned_flipped(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out, out.view(-1)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    return (mul, view)""",
        )

    def test_output_aliases_intermediate_and_returned_different_grad(self):
        def f(a):
            out = torch.mul(a, 3)
            # AOTAutograd should manually generate the first output (a view of an intermediate)
            # but not the second (which is itself the intermediate for the first)
            return out.view(-1), out, out[0].detach()

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    select = torch.ops.aten.select.int(mul, 0, 0)
    detach = torch.ops.aten.detach.default(select);  select = None
    return (view, mul, detach)""",
        )

    def test_output_aliases_intermediate_inplace_view(self):
        def f(a):
            out = torch.mul(a, 3)
            out.t_()
            return out

        # TODO: fix this test.
        # See https://github.com/pytorch/pytorch/issues/90507
        # self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_inplace_view_with_detach(self):
        def f(a):
            out = torch.mul(a, 3)
            out.t_()
            out.detach_()
            # Thanks to the detach_() AOT Autograd doesn't need to do anything.
            # `out` will show up as having OutputType.non_alias,
            # and ._is_view() == False
            return out, a + 1

        inp = [torch.ones(2, 4, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(2, 4, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3)
    t = torch.ops.aten.t.default(mul);  mul = None
    add = torch.ops.aten.add.Tensor(primals_1, 1);  primals_1 = None
    return (t, add)""",
        )

    def test_output_aliases_intermediate_inplace_view_and_view(self):
        def f(a):
            out = torch.mul(a, 3)
            out_view = out.unsqueeze(0)
            out.t_()
            out_view2 = out.unsqueeze(0)
            return out_view, out, out_view2

        inp = [torch.ones(2, 4, requires_grad=True)]  # noqa: F841

        # TODO: fix this test.
        # See <github issue link>
        # self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_output_aliases_intermediate_multiple_mixed(self):
        def f(a):
            out1 = torch.mul(a, 3)
            out2 = torch.mul(a, 4)
            # AOTAutograd should manually generate these two output views in the epilogue.
            return out1.view(-1), out2.transpose(1, 0), out1.transpose(1, 0)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3)
    mul_1 = torch.ops.aten.mul.Tensor(primals_1, 4);  primals_1 = None
    view = torch.ops.aten.view.default(mul, [-1])
    transpose = torch.ops.aten.transpose.int(mul_1, 1, 0);  mul_1 = None
    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)
    return (view, transpose, transpose_1, mul)""",
        )

    def test_output_all_alias_types(self):
        # There are 3 types of aliasing that require us to return metadata in the compiled fw:
        # (1) outputs that are views of inputs
        # (2) outputs that are views of intermediates
        # (3) inputs that get metadata mutations
        # test all 3 of them here
        def f(a):
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.squeeze(), tmp.transpose(1, 0), a.unsqueeze(0)

        def inp_callable(req_grad):
            x = torch.ones(1, 2, 4, requires_grad=req_grad).clone()
            return [(x,), (x,)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # TODO: make this test run with dynamic shapes so it is more meaningful
        # metadata output order: (a_updated_meta, out1_meta, out2_meta, out3_meta)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    view = torch.ops.aten.view.default(primals_1, [1, 2, 4]);  primals_1 = None
    transpose = torch.ops.aten.transpose.int(view, 1, 0);  view = None
    mul = torch.ops.aten.mul.Tensor(transpose, 2)
    squeeze = torch.ops.aten.squeeze.default(mul)
    transpose_1 = torch.ops.aten.transpose.int(mul, 1, 0)
    unsqueeze = torch.ops.aten.unsqueeze.default(transpose, 0)
    return (transpose, squeeze, transpose_1, unsqueeze, mul)""",
        )

    @parametrize("req_grad", [False, True])
    def test_subclass_metadata_mutation(self, req_grad):
        def f(a):
            a.transpose_(1, 0)
            tmp = a.mul(2)
            return tmp.transpose(1, 0)

        def inp_callable(req_grad):
            x = torch.ones(1, 2, 4, requires_grad=req_grad).clone()
            return [(x,), (x,)]

        # See https://github.com/pytorch/pytorch/issues/114975
        with self.assertRaisesRegex(
            RuntimeError,
            "Metadata mutations are currently not allowed on tensor subclasses",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=req_grad),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

    def test_input_data_and_metadata_mutation(self):
        def f(a):
            a.t_()
            a[0].mul_(2)
            return a.view(a.shape)

        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=True)]
        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    t = torch.ops.aten.t.default(clone)
    select = torch.ops.aten.select.int(t, 0, 0);  t = None
    mul = torch.ops.aten.mul.Tensor(select, 2);  select = None
    t_1 = torch.ops.aten.t.default(clone);  clone = None
    select_scatter = torch.ops.aten.select_scatter.default(t_1, mul, 0, 0);  t_1 = mul = None
    t_2 = torch.ops.aten.t.default(select_scatter);  select_scatter = None
    t_4 = torch.ops.aten.t.default(t_2)
    t_6 = torch.ops.aten.t.default(t_2);  t_2 = None
    view_1 = torch.ops.aten.view.default(t_6, [3, 3]);  t_6 = None
    return (t_4, view_1)""",
        )

    def test_view_and_inplace_view(self):
        def f(a, b):
            a.t_()
            return b.view(b.shape), a.view(a.shape)

        def create_inp(req_grad):
            return [
                torch.ones(3, 3, requires_grad=req_grad),
                torch.ones(3, 3, requires_grad=req_grad),
            ]

        self.verify_aot_autograd(f, create_inp(False), test_mutation=True)
        fw_graph = self.verify_aot_autograd(f, create_inp(True), test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    view = torch.ops.aten.view.default(arg1_1, [3, 3]);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(t, [3, 3])
    return (t, view, view_1)""",
        )

    def test_view_detach(self):
        def f(a):
            tmp = a.detach()
            a.mul_(2)
            return a, tmp

        inp = [torch.ones(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp, test_mutation=True)
        inp = [torch.ones(3, 3, requires_grad=False)]
        self.verify_aot_autograd(f, inp, test_mutation=True)

    def test_input_inplace_requires_grad_true(self):
        def f(a, b):
            a.requires_grad_(True)
            return a.mul(3), b.mul(4)

        inp = [
            # First inp doesn't require grad, but we switch it on
            torch.ones(3, 3, requires_grad=False),
            torch.ones(3, 3, requires_grad=True),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 4);  primals_2 = None
    return (mul, mul_1)""",
        )

    # This is a torture test:
    # a and b get turned into a synthetic base in the compiled graph
    # One gets a data mutation, the other gets a metadata mutation.
    # We need to make sure that the metadata mutation gets propagated
    # back to the original input.
    @skipIfDynamoInput("Dynamo removes runtime error")
    def test_input_data_and_metadata_mutation_aliases_other_input(self):
        # a and b are aliased
        def f(a, b):
            a.mul_(2)
            b.t_()
            return a.mul(b)

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[0]
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "Encountered aliased inputs that are mutated in the graph, but",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=True),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

    # https://github.com/pytorch/pytorch/issues/106456
    def test_input_mutation_noncontiguous(self):
        def f(a):
            a.mul_(2)
            return a + 1

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            # create a non-contiguous view to pass as an input to the compiler
            inp = x[:, 0]
            return [base], [inp]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

    def test_backward_mutation_data(self):
        class BwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                # bw mutation
                x.mul_(2)
                return grad_output.clone()

        def f(a, b):
            out = BwMutation.apply(b)
            return a * out

        inp_no_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        # Mutation on buffer that does not require grad during the backward is allowed
        self.verify_aot_autograd(f, inp_no_grad, test_mutation=True)

        inp_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp_grad, test_mutation=True)

    def test_fw_bw_mutation_no_functionalization1(self):
        class FwBwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                # input mutation
                torch._foreach_mul_([b], [2])
                x = b + 1
                # intermediate mutation
                torch._foreach_mul_([x], [3])
                ctx.save_for_backward(x)
                return x * a

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                # bw mutation
                torch._foreach_mul_([x], [4])
                return grad_output * x, grad_output * x

        def f(a, b):
            return FwBwMutation.apply(a, b).sin_().clone()

        inps = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]
        inps_ref = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        fw_graph = [None]
        bw_graph = [None]

        def fw_compiler(gm, example_inputs):
            fw_graph[0] = gm
            return gm

        def bw_compiler(gm, example_inputs):
            bw_graph[0] = gm
            return gm

        compiled_f = compiled_function(
            f,
            fw_compiler,
            bw_compiler,
            dynamic=False,
            partition_fn=default_partition,
            keep_inference_input_mutations=True,
            disable_functionalization=True,
        )

        out_ref = f(*inps_ref)
        out = compiled_f(*inps)
        self.assertEqual(out, out_ref)

        out_ref.sum().backward()
        out.sum().backward()
        self.assertEqual(inps_ref[0].grad, inps[0].grad)

        # important bit: there are 2 mutations in the fw
        self.assertExpectedInline(
            fw_graph[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    _foreach_mul_ = torch.ops.aten._foreach_mul_.ScalarList([primals_2], [2]);  _foreach_mul_ = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    _foreach_mul__1 = torch.ops.aten._foreach_mul_.ScalarList([add], [3]);  _foreach_mul__1 = None
    mul = torch.ops.aten.mul.Tensor(add, primals_1);  primals_1 = None
    clone = torch.ops.aten.clone.default(mul)
    sin_ = torch.ops.aten.sin_.default(mul);  mul = None
    clone_1 = torch.ops.aten.clone.default(sin_);  sin_ = None
    return (clone_1, add, clone)""",
        )

        # important bit: there is 1 mutation in the bw
        self.assertExpectedInline(
            bw_graph[0].code.strip(),
            """\
def forward(self, add, clone, tangents_1):
    cos = torch.ops.aten.cos.default(clone);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, cos);  tangents_1 = cos = None
    _foreach_mul__2 = torch.ops.aten._foreach_mul_.ScalarList([add], [4]);  _foreach_mul__2 = None
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, add);  mul_1 = add = None
    return (mul_2, None)""",
        )

    def test_fw_bw_mutation_no_functionalization2(self):
        class FwBwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # input mutation
                torch._foreach_mul_([x], [2])
                x = x + 1
                # intermediate mutation
                torch._foreach_mul_([x], [3])
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                # bw mutation
                torch._foreach_mul_([x], [4])
                return grad_output * x

        def f(a, b):
            out = FwBwMutation.apply(b)
            return out * a

        inps = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]
        inps_ref = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        fw_graph = [None]
        bw_graph = [None]

        def fw_compiler(gm, example_inputs):
            fw_graph[0] = gm
            return gm

        def bw_compiler(gm, example_inputs):
            bw_graph[0] = gm
            return gm

        compiled_f = compiled_function(
            f,
            fw_compiler,
            bw_compiler,
            dynamic=False,
            partition_fn=default_partition,
            keep_inference_input_mutations=True,
            disable_functionalization=True,
        )

        out_ref = f(*inps_ref)
        out = compiled_f(*inps)
        self.assertEqual(out, out_ref)

        out_ref.sum().backward()
        out.sum().backward()
        self.assertEqual(inps_ref[0].grad, inps[0].grad)

        # important bit: there are 2 mutations in the fw
        # (the mutation on an activation doesn't get moved to bw)
        self.assertExpectedInline(
            fw_graph[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    _foreach_mul_ = torch.ops.aten._foreach_mul_.ScalarList([primals_2], [2]);  _foreach_mul_ = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    _foreach_mul__1 = torch.ops.aten._foreach_mul_.ScalarList([add], [3]);  _foreach_mul__1 = None
    mul = torch.ops.aten.mul.Tensor(add, primals_1);  primals_1 = None
    return (mul, add)""",
        )

        self.assertExpectedInline(
            bw_graph[0].code.strip(),
            """\
def forward(self, add, tangents_1):
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, add);  tangents_1 = add = None
    return (mul_1, None)""",
        )

    def test_backward_mutation_metadata(self):
        class BwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(b)
                return a.clone(), b.clone()

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                (b,) = ctx.saved_tensors
                # bw metadata mutation
                b.transpose_(1, 0)
                return grad_a.clone(), grad_b.clone()

        def f(a, b):
            a_, b_ = BwMutation.apply(a, b)
            out = a_ * b_
            return out

        inp_no_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=False),
        ]

        with self.assertRaisesRegex(
            AssertionError, "input that had its metadata mutated in the backward"
        ):
            self.verify_aot_autograd(f, inp_no_grad, test_mutation=True)

    def test_backward_mutation_on_grad_out(self):
        class BwMutation(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                grad_output.mul_(2)
                return grad_output.clone()

        def f(a, b):
            tmp = a * b
            out = BwMutation.apply(tmp)
            return out

        inp_grad = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]
        inp_grad_ref = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]

        f_compiled = aot_function(f, nop)
        out = f_compiled(*inp_grad)
        out.mul(2).sum().backward()
        out_ref = f(*inp_grad_ref)
        out_ref.mul(2).sum().backward()
        self.assertEqual(inp_grad[0].grad, inp_grad_ref[0].grad)
        self.assertEqual(inp_grad[1].grad, inp_grad_ref[1].grad)

    def test_backward_mutation_forward_inputs(self):
        @torch.library.custom_op("_test::_clone", mutates_args={})
        def f(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            return x.clone()

        def f_fake(x, x1):
            return torch.empty_like(x)

        def backward(ctx, grad):
            with torch.no_grad():
                ctx.x1.zero_()
            return grad * 2, None

        def setup_context(ctx, inputs, output):
            (x, x1) = inputs
            ctx.x = x
            ctx.x1 = x1

        f.register_fake(f_fake)
        f.register_autograd(backward, setup_context=setup_context)

        def fn(x: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            x2.mul_(5)
            return torch.ops._test._clone(x, x1) + x2

        inp_x, inp_x1, inp_x2 = (
            torch.randn(3, requires_grad=True),
            torch.randn(3, requires_grad=False),
            torch.randn(3, requires_grad=False),
        )

        ref_x, ref_x1, ref_x2 = inp_x.clone(), inp_x1.clone(), inp_x2.clone()
        ref_y = fn(ref_x, ref_x1, ref_x2)

        compiled_f = aot_function(fn, nop, keep_inference_input_mutations=True)

        x, x1, x2 = inp_x.clone(), inp_x1.clone(), inp_x2.clone()
        y = compiled_f(x, x1, x2)

        # Verify mutation in forward applied and mutation in backward is not in forward
        self.assertEqual(ref_x, x)
        self.assertEqual(ref_x1, x1)
        self.assertEqual(ref_x2, x2)
        self.assertEqual(ref_y, y)

        ref_y.sum().backward()
        y.sum().backward()

        # Verify mutations in backward applied
        self.assertEqual(ref_x, x)
        self.assertEqual(ref_x1, x1)
        self.assertEqual(ref_x2, x2)
        self.assertEqual(ref_y, y)

        self.assertEqual(ref_x.grad, x.grad)
        self.assertEqual(ref_x1.grad, x1.grad)
        self.assertEqual(ref_x2.grad, x2.grad)

    def test_backward_mutation_forward_inputs_create_graph(self):
        @torch.library.custom_op("_test::_clone_create_graph", mutates_args={})
        def f(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            return x.clone()

        def f_fake(x, x1):
            return torch.empty_like(x)

        def backward(ctx, grad):
            with torch.no_grad():
                ctx.x1.zero_()
            return grad * 2, None

        def setup_context(ctx, inputs, output):
            (x, x1) = inputs
            ctx.x = x
            ctx.x1 = x1

        f.register_fake(f_fake)
        f.register_autograd(backward, setup_context=setup_context)

        def fn(x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
            return torch.ops._test._clone_create_graph(x, x1)

        inp_x, inp_x1 = (
            torch.randn(3, requires_grad=True),
            torch.randn(3, requires_grad=True),
        )

        ref_x, ref_x1 = inp_x.clone(), inp_x1.clone()
        ref_y = f(ref_x, ref_x1)
        ref_y.sum().backward()
        x, x1 = inp_x.clone(), inp_x1.clone()
        compiled_f = aot_function(fn, nop)
        y = compiled_f(x, x1)
        loss = y.sum()
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_autograd does not support input mutations with requires_grad in backward for create_graph=True",
        ):
            torch.autograd.grad(loss, inp_x, create_graph=True)
        # Not checking equality of ref and x as Exception is expected

    # Partially addresses https://github.com/pytorch/pytorch/issues/106457
    def test_input_mutation_false_aliasing(self):
        def f(a, b):
            a.mul_(3)
            b.mul_(2)
            return a.clone().view(-1) + b.clone().view(-1)

        # No overlap, contiguous
        def inp_callable1(req_grad):
            base = torch.ones(4, 4, requires_grad=req_grad)
            x = base.add(1)
            # create two views that share storage, but are actually non-overlapping
            a = x[0:2]
            b = x[2:4]
            return [base], [a, b]

        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable1, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable1, req_grad=True), test_mutation=True
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable1, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable1, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

        # Important characteristic: the graph takes in 2 inputs!
        # That shows that we didn't try to run our complicated synthetic base logic,
        # because we successfully detected false aliasing across the two inputs.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mul = torch.ops.aten.mul.Tensor(arg0_1, 3);  arg0_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    clone = torch.ops.aten.clone.default(mul)
    view = torch.ops.aten.view.default(clone, [-1]);  clone = None
    clone_1 = torch.ops.aten.clone.default(mul_1)
    view_1 = torch.ops.aten.view.default(clone_1, [-1]);  clone_1 = None
    add = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
    return (mul, mul_1, add)""",
        )

        # No overlap, non-contiguous: first tensor ends before second tensor start
        def inp_callable2(req_grad):
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (8, 1), storage_offset=0)
            b = x.as_strided((4, 4), (8, 1), storage_offset=28)
            return [base], [a, b]

        # No overlap, non-contiguous: tensors are perfectly interleaved
        def inp_callable3(req_grad):
            base = torch.ones(4, 4, requires_grad=req_grad)
            x = base.add(1)
            a = x[:, 0:2]
            b = x[:, 2:4]
            return [base], [a, b]

        # No overlap, non-contiguous
        def inp_callable4(req_grad):
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=22)
            return [base], [a, b]

        # No overlap, non-contiguous
        def inp_callable5(req_grad):
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=23)
            return [base], [a, b]

        # No overlap, non-contiguous
        def inp_callable6(req_grad):
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            # a's last element is at offset 195 (24 total elements)
            a = x.as_strided((2, 4, 3), (110, 24, 4), storage_offset=5)
            # b's first element is at offset 196: no overlap
            b = x[196 : 196 + a.numel()]
            return [base], [a, b]

        # overlap! non-contiguous
        def inp_callable_overlap1(req_grad):
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=24)
            return [base], [a, b]

        # overlap! non-contiguous
        def inp_callable_overlap2(req_grad):
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            a = x.as_strided((4, 4), (9, 1), storage_offset=0)
            b = x.as_strided((4, 4), (9, 1), storage_offset=25)
            return [base], [a, b]

        # overlap! non-contiguous
        def inp_callable_overlap3(req_grad):
            base = torch.ones(256, requires_grad=req_grad)
            x = base.add(1)
            # a's last element is at offset 195 (24 total elements)
            a = x.as_strided((2, 4, 3), (110, 24, 4), storage_offset=5)
            # b's first element is at offset 195: overlap!
            b = x[195 : 195 + a.numel()]
            return [base], [a, b]

        fw_graph2 = self.verify_aot_autograd(
            f, partial(inp_callable2, req_grad=False), test_mutation=True
        )
        fw_graph3 = self.verify_aot_autograd(
            f, partial(inp_callable3, req_grad=False), test_mutation=True
        )
        fw_graph4 = self.verify_aot_autograd(
            f, partial(inp_callable4, req_grad=False), test_mutation=True
        )
        fw_graph5 = self.verify_aot_autograd(
            f, partial(inp_callable5, req_grad=False), test_mutation=True
        )
        fw_graph6 = self.verify_aot_autograd(
            f, partial(inp_callable6, req_grad=False), test_mutation=True
        )

        fw_graph_overlap1 = self.verify_aot_autograd(
            f, partial(inp_callable_overlap2, req_grad=False), test_mutation=True
        )
        fw_graph_overlap2 = self.verify_aot_autograd(
            f, partial(inp_callable_overlap1, req_grad=False), test_mutation=True
        )

        # All non-overlap graphs should be the same since we detected false aliasing
        self.assertEqual(str(fw_graph.code), str(fw_graph2.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph3.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph4.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph5.code))
        self.assertEqual(str(fw_graph.code), str(fw_graph6.code))

        # All overlap graphs should be the same since we detected real aliasing
        self.assertNotEqual(str(fw_graph.code), str(fw_graph_overlap1.code))
        self.assertNotEqual(str(fw_graph.code), str(fw_graph_overlap2.code))
        self.assertTrue("as_strided_scatter" in str(fw_graph_overlap1.code))
        self.assertTrue("as_strided_scatter" in str(fw_graph_overlap2.code))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_mem_leak_from_save_for_bw(self):
        # See a full diagnosis at this issue: https://github.com/pytorch/pytorch/issues/94990
        # Note [Detaching saved tensors in AOTAutograd]
        # This program creates a ref-cycle. Long term, we should fix this ref cycle
        # (since it can arise, naturally albeit rarely, from uses of autograd.Function).
        # But AOTAutograd makes it more likely to show up from tracing user programs,
        # so we deal with it by manually detaching the tensors that we save for backward.
        # This is completely wrong and would give wrong results if we were to do double backward.
        # Fortunately today, double backward is explicitly banned in AOTAutograd.
        def f(a, b):
            add = a + a
            split = torch.functional.split(add, [4, 4], dim=1)
            getitem_2 = split[1]
            unsqueeze = getitem_2.unsqueeze(-1)
            mul = unsqueeze * b
            return (getitem_2, mul)

        f_compiled = aot_function(f, nop)
        inps = [
            torch.ones(8, 8, device="cuda", requires_grad=True),
            torch.ones(1, 4, 1, device="cuda", requires_grad=True),
        ]
        mem_before = torch.cuda.memory_allocated()
        f_compiled(*inps)
        mem_after = torch.cuda.memory_allocated()
        self.assertTrue(mem_after == mem_before)

    def test_output_aliases_multiple_inputs_get_correct_one(self):
        # a and b are aliased, but have different shapes
        # The first output should view off the first input, the 2nd output should view off the 2nd input
        def f(a, b):
            return a.view(a.shape), b.view(b.shape)

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.mul(2)
            inp1 = x.view(-1)
            inp2 = x[0]
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=False),
            test_mutation=True,
            make_inputs_subclasses=True,
        )
        self.verify_aot_autograd(
            f,
            partial(inp_callable, req_grad=True),
            test_mutation=True,
            make_inputs_subclasses=True,
        )

    def test_input_mutation_aliases_other_input(self):
        def f(a, b):
            a.add_(1)
            return a + b

        def inp_callable(req_grad):
            base = torch.ones(4, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[0]
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # Important parts of the graph:
        # - the compiled graph takes in a base, and we generate a and b (the views) off of the base
        # - clone() is still in the graph, because we need to call grad() on the original (non-mutated) inputs
        # - We re-generate the views *after* the clone, to preserve view relationships.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = add = None
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)
    add_1 = torch.ops.aten.add.Tensor(as_strided_2, as_strided_5);  as_strided_2 = as_strided_5 = None
    return (as_strided_scatter, add_1)""",
        )  # noqa: B950

    def test_input_mutation_aliases_other_input2(self):
        def f(a, b):
            a.add_(1)
            return a + b

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            inp1 = x[0]
            # Here, one of the aliased inputs is the base itself
            inp2 = x
            return [base], [inp1, inp2]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = add = None
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 0)
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [2, 2], [2, 1], 0)
    add_1 = torch.ops.aten.add.Tensor(as_strided_2, as_strided_5);  as_strided_2 = as_strided_5 = None
    return (as_strided_scatter, add_1)""",
        )  # noqa: B950

    def test_input_mutation_aliases_and_output_alias(self):
        def f(a, b):
            # Here, we need to take care:that because and b are aliased
            # since a and b are aliased, we generate a view off of "updated b"
            a.add_(1)
            return b.view(b.shape)

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return [base], [x.view(-1), x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None
    as_strided_9 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    view_1 = torch.ops.aten.view.default(as_strided_9, [4]);  as_strided_9 = None
    return (as_strided_scatter, view_1)""",
        )  # noqa: B950

    def test_input_aliased_with_mutation_output_alias(self):
        def f(a, b, c):
            # a and c alias
            c.mul_(2)
            # The main thing we're testing here is that
            # (1) We need to reconstruct c.view(-1) from the 3rd input to the forward
            # (2) But we need to be careful to do this *before* converting aliased inputs into synthetic bases.
            #     The original fw takes in 3 args, but the compiled fw takes in only 2 args.
            return b.add(1), c.view(-1)

        def inp_callable(req_grad):
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            x = base1.add(1)
            y = base2.add(1)
            return [base1, base2], [x.view(-1), y, x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    as_strided_7 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    view_1 = torch.ops.aten.view.default(as_strided_7, [-1]);  as_strided_7 = None
    return (as_strided_scatter, add, view_1)""",
        )  # noqa: B950

    def test_input_metadata_mutation_aliases(self):
        def f(a, b):
            # a and b alias, and we do a metadata mutation on a
            # Since we're not mutating data, then b isn't affected at all.
            # We expect aot autograd to not bother with constructing a synthetic base.
            a.t_()
            return a + b

        def inp_callable(req_grad):
            base = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return [base], [x.view(-1), x.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # Expectation: fwd() takes in 2 args, and we don't construct a synthetic base.
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    t = torch.ops.aten.t.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(t, primals_2);  t = primals_2 = None
    return (add,)""",
        )

    def test_input_mutation_aliases_and_none_require_gradients(self):
        def f(a, b, c):
            # a and b alias, but neither require gradients (so they don't have a _base)
            # aot autograd should construct the synthetic base from `torch.Tensor(a.storage())`
            a.mul_(2)
            return b + 1, c + 1

        def inp_callable(req_grad):
            base = torch.ones(2, 2)
            c_arg = torch.ones(2, 2, requires_grad=req_grad)
            x = base.add(1)
            return [base, c_arg], [x.view(-1), x.view(-1), c_arg]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )

        with self.assertRaisesRegex(
            RuntimeError, "is a tensor subclass. This is not supported today"
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    as_strided = torch.ops.aten.as_strided.default(primals_1, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided, 2);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(primals_1, mul, [4], [1], 0);  primals_1 = mul = None
    as_strided_3 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided_3, 1);  as_strided_3 = None
    add_1 = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    return (as_strided_scatter, add, add_1)""",
        )  # noqa: B950

    @skipIfDynamoInput("Fails with dynamo")
    def test_input_mutation_aliases_bases_out_of_order(self):
        # This tests our calling convention: if b and d are aliased, then the outer calling convention
        # that we send to the compiled forward becomes:
        # (b_d_base, a, c)
        # Importantly, even though a and c alias in our test, neither inputs are mutated,
        # So we don't need to do the base construction / deconstruction
        def f(a, b, c, d):
            b.add_(1)
            d.unsqueeze_(0)
            return a + c + d, b.view(-1)

        def inp_callable(req_grad):
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            x1 = base1.add(1)
            x2 = base2.add(1)
            # a and c alias, b and d alias
            return [base1, base2], [x1.view(-1), x2.view(-1), x1.view(-1), x2.view(-1)]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Metadata mutations are currently not allowed on tensor subclasses",
        ):
            self.verify_aot_autograd(
                f,
                partial(inp_callable, req_grad=False),
                test_mutation=True,
                make_inputs_subclasses=True,
            )

        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # 3 graph inputs: (b_d_base, a, c)
        # 2 returns: (b_updated, a+c+d)
        # (there are 2 original fw outs, but one is a view of b so it's not part of the graph)
        # (there are also 2 input mutations, but one is a metadata-only mutation so the compiled forward doesn't return it)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = add = None
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    unsqueeze = torch.ops.aten.unsqueeze.default(as_strided_5, 0);  as_strided_5 = None
    add_1 = torch.ops.aten.add.Tensor(primals_2, primals_3);  primals_2 = primals_3 = None
    add_2 = torch.ops.aten.add.Tensor(add_1, unsqueeze);  add_1 = None
    as_strided_14 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    view_2 = torch.ops.aten.view.default(as_strided_14, [-1]);  as_strided_14 = None
    return (as_strided_scatter, add_2, view_2, unsqueeze)""",
        )  # noqa: B950

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_synthetic_base_base_attribute_is_none(self):
        def f(a, b):
            a.add_(1)
            return a + b

        def inp_callable():
            base = torch.ones(4, 4, device="cuda")
            # detach() so that none of the inputs have a ._base attribute.
            a = base[0].detach()
            b = base[1].detach()
            base2 = torch.ones(2, 2, requires_grad=True)  # noqa: F841
            return [base], [a, b]

        self.verify_aot_autograd(f, inp_callable, test_mutation=True)

    def test_input_mutation_alias_everything(self):
        # Mondo test that tests a combination of:
        # input is mutated, that aliases another input (so we make a synthetic base)
        # an output is an alias of another output
        # an output is an alias of an intermediate
        # a and c are aliased
        def f(a, b, c):
            c.mul_(2)  # mutates c
            b.t_()  # metadata mutate b
            tmp = a + c
            out1 = tmp.view(-1)
            out2 = b.t()
            out3 = out1.unsqueeze(0)
            # out1 and out3 are aliases of an intermediate, and alias each other!
            # out2 aliases an input, so we don't return it
            return out1, out2, out3

        def inp_callable(req_grad):
            base1 = torch.ones(2, 2, requires_grad=req_grad)
            base2 = torch.ones(2, 2, requires_grad=req_grad)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            base1_ = base1.add(1)
            base2_ = base2.add(1)
            a = base1_.view(-1)
            b = base2_
            c = base1_.view(-1)
            return [base1, base2], [a, b, c]

        self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=False), test_mutation=True
        )
        fw_graph = self.verify_aot_autograd(
            f, partial(inp_callable, req_grad=True), test_mutation=True
        )
        # Expected:
        # - 2 inputs in the forward: synthetic_base_a_c, b
        # - 1 output in the forward: "tmp"
        #   out2 is an alias of an input, and will be generated off of b outside of the compiled fn
        #   out1 and out3 are aliases of tmp, that we generate outside of the compiled function
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    view = torch.ops.aten.view.default(primals_2, [2, 2]);  primals_2 = None
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = mul = None
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    t = torch.ops.aten.t.default(view);  view = None
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided_5, as_strided_2);  as_strided_5 = as_strided_2 = None
    view_1 = torch.ops.aten.view.default(add, [-1])
    t_1 = torch.ops.aten.t.default(t)
    unsqueeze = torch.ops.aten.unsqueeze.default(view_1, 0)
    return (as_strided_scatter, t, view_1, t_1, unsqueeze, add)""",
        )  # noqa: B950

    def test_dynamic_shape_output_not_in_bw_graph(self):
        def f(x):
            return [x + 1, x.shape[0]]

        inp = torch.ones(5, requires_grad=True)
        bw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            decompositions={},
            keep_inference_input_mutations=False,
            dynamic=True,
        )
        out = compiled_f(inp)
        out[0].sum().backward()
        # The important bit: the forward fn returns 2 outputs,
        # but one of them is a symint so we should only see
        # 1 grad_output as an input to the backward graph.
        # (Otherwise, autograd will plumb a None as the value of the grad_output,
        # which causes inductor to complain).
        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, tangents_1):
    return (tangents_1,)""",
        )

    def test_no_grad_input_output(self):
        def f(a, b):
            return a.cos(), b.cos(), a * b

        inp_thunks = [
            lambda: torch.randn(5, requires_grad=True),
            lambda: torch.randn(5, requires_grad=False),
        ]
        for inps in itertools.product(inp_thunks, repeat=2):
            inps = [i() for i in inps]
            self.verify_aot_autograd(f, inps)

    def test_some_output_requires_grad_input_doesnt(self):
        def f(a, b):
            a_view = a.view(-1)
            a_view.requires_grad_(True)
            return a_view

        inp = [torch.randn(3, 3), torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

    def test_some_outputs_dont_require_grad_view(self):
        def f(a, b):
            return a.detach(), b

        inp = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp)

    def test_some_outputs_dont_require_grad_non_view(self):
        def f(a, b):
            return a.add(1).detach(), b

        inp = [
            torch.randn(3, 3, requires_grad=True),
            torch.randn(3, 3, requires_grad=True),
        ]
        self.verify_aot_autograd(f, inp)

    def test_inner_grad(self):
        def foo(x):
            y = torch.exp(x)
            z = torch.autograd.grad(y, x)
            return z

        inps = [torch.randn((), requires_grad=True)]
        self.verify_aot_autograd(foo, inps)

    def test_grad_context(self):
        def foo(x):
            return x * 2

        inps = [torch.randn((), requires_grad=True)]
        graph_size = None

        def get_graph_size(fx_g, _):
            nonlocal graph_size
            graph_size = len(fx_g.graph.nodes)
            return fx_g

        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(False):
            f(*inps)
        self.assertIsNone(graph_size)

        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(True):
            out = f(*inps)
            self.assertIsNone(graph_size)
            out.sum().backward()
            self.assertTrue(graph_size > 2)

    def test_output_dict(self):
        def f(x):
            return {"a": x, "b": x}

        inp = [torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

        def f(x, y):
            return {"a": x, "b": y + x}

        inp = [torch.randn(3, requires_grad=True), torch.randn(3)]
        self.verify_aot_autograd(f, inp)

        def f(x):
            new_d = {}
            for k in x:
                new_d[k] = x[k] * 2
            return new_d

        a = torch.randn(3, requires_grad=True)
        b = torch.randn(3, requires_grad=True)

        def inp_callable():
            inps = [{"a": a, "b": b}]
            return inps, inps

        self.verify_aot_autograd(f, inp_callable)

    def test_module(self):
        mod = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        compiled_mod = compiled_module(mod, nop, nop)
        inp = torch.randn(32, 32)
        ref_out = mod(inp)
        ref_out.sum().backward()
        ref_grads = sorted([(name, p.grad) for name, p in mod.named_parameters()])
        out = compiled_mod(inp)
        out.sum().backward()
        grads = sorted([(name, p.grad) for name, p in mod.named_parameters()])
        self.assertEqual((out, grads), (ref_out, ref_grads))

    def test_batchnorm(self):
        mod = compiled_module(nn.BatchNorm2d(4), nop, nop)
        x = torch.ones(1, 4, 2, 2)
        mod(x).sum().backward()

    def test_list_codegen(self):
        def list_nop(f, _):
            def g(inps):
                return f(*inps)

            g._boxed_call = True
            return g

        def f(a, b, c):
            return a.sin() * b.cos() * c.sin()

        f = aot_function(f, list_nop)
        inp = [torch.randn(5, requires_grad=True) for _ in range(3)]
        f(*inp).sum().backward()

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    def test_compilation_context(self, counter):
        def f(x):
            return x.sin().sin()

        count = []

        def compiler(fx_g, _):
            context = get_aot_compilation_context()
            count.append((context[0], len(fx_g.graph.nodes)))
            return fx_g

        f = aot_function(f, compiler)
        out = f(torch.randn(5, requires_grad=True))
        f = aot_function(f, compiler)
        f(torch.randn(5))
        out.sum().backward()
        self.assertExpectedInline(
            str(count),
            """[(['0_forward'], 4), (['1_inference'], 4), (['0_backward'], 8)]""",
        )

    def test_dupe_arg(self):
        def f(x, y):
            return x + y

        x = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(f, [x, x])

    def test_dupe_arg_torture(self):
        def f(x, y):
            x.t_()
            y.unsqueeze_(0)
            return x + y

        x = torch.randn(3, 3, requires_grad=True).clone()
        self.verify_aot_autograd(f, [x, x])

    # See https://github.com/pytorch/pytorch/issues/100224
    def test_dupe_arg_returned_as_output(self):
        def f(a, b, a_):
            a[0].add_(1)
            return a_

        f_compiled = aot_function(f, nop)
        a = torch.ones(2)
        b = torch.ones(2)
        out_ref = f(a, b, a)

        a2 = torch.ones(2)
        b2 = torch.ones(2)
        out_test = f_compiled(a2, b2, a2)

        self.assertEqual(out_ref, out_test)
        self.assertEqual(a, a2)

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe_left_bias(self, counter):
        # This test checks that, just because only the first
        # argument did a metadata mutation, we still correctly
        # switch to strategy 2 (deduplicate)
        # See: https://github.com/pytorch/pytorch/pull/89896#discussion_r1036224447
        class F(torch.nn.Module):
            def forward(self, x, y):
                x.t_()
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(F(), [x, x])

        fxx = aot_module_simplified(F(), (x, x), nop)
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxx(x, y),
            """At compilation time, graph 2 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",  # noqa: B950
        )

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe(self, counter):
        self._test_invalid_dupe(counter, fake=False)

    # See Note: Dynamo recompilation guarding invalid grad for why this test exists
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe_fake(self, counter):
        self._test_invalid_dupe(counter, fake=True)

    def _test_invalid_dupe(self, counter, fake):
        class F(torch.nn.Module):
            def forward(self, x, y):
                x.unsqueeze_(0)
                y.unsqueeze_(0)
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()

        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)

            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)

        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)

        fxy(x, y)
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()
        fxy(x, x)  # is ok!

        if fake:
            fxx = aot_module_simplified(F(), (fake_x, fake_x), nop)
        else:
            fxx = aot_module_simplified(F(), (x, x), nop)

        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()
        fxx(x, x)
        # Note This should not raise! Once we have guards in place here,
        # we will have this working correctly, as it should recompile.
        x = torch.randn(3, 3, requires_grad=True).clone()
        y = torch.randn(3, 3, requires_grad=True).clone()
        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxx(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",  # noqa: B950
        )

    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_requires_grad(self, counter):
        self._test_invalid_requires_grad(counter, fake=False)

    # See Note: Dynamo recompilation guarding invalid grad for why this test exists
    @patch("torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_requires_grad_fake(self, counter):
        self._test_invalid_requires_grad(counter, fake=True)

    def _test_invalid_requires_grad(self, counter, fake):
        class F(torch.nn.Module):
            def forward(self, x, y):
                return (x + y,)

        x = torch.randn(3, 3, requires_grad=True)
        y = torch.randn(3, 3, requires_grad=True)
        z = torch.randn(3, 3, requires_grad=False)

        if fake:
            shape_env = ShapeEnv()
            fake_mode = FakeTensorMode(shape_env=shape_env)

            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)
            fake_z = fake_mode.from_tensor(z)

        if fake:
            fxy = aot_module_simplified(F(), (fake_x, fake_y), nop)
        else:
            fxy = aot_module_simplified(F(), (x, y), nop)

        compare_equal_outs_and_grads(self, F(), fxy, (x, y))
        compare_equal_outs_and_grads(self, F(), fxy, (x, z))

        if fake:
            fxz = aot_module_simplified(F(), (fake_x, fake_z), nop)
        else:
            fxz = aot_module_simplified(F(), (x, z), nop)

        compare_equal_outs_and_grads(self, F(), fxz, (x, z))

        self.assertExpectedRaisesInline(
            AssertionError,
            lambda: fxz(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would not require grad, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch.""",  # noqa: B950
        )

    def test_custom_autograd(self):
        class CustomFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output + 1

        def f(x):
            return CustomFn.apply(x)

        self.verify_aot_autograd(f, [torch.randn(3)])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_autocast_disable_guard(self):
        with torch._C._DisableAutocast():
            x = torch.rand([4, 4]).cuda()
            y = x @ x
            self.assertEqual(y.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_nonidempotent_amp(self):
        def f(self_s_emb, add_3):
            einsum_2 = torch.functional.einsum("ah,th->t", self_s_emb, add_3)
            log_softmax_2 = einsum_2.log_softmax(-1)
            return (log_softmax_2,)

        args = [
            torch.rand((1, 256), dtype=torch.float32, device="cuda"),
            torch.rand((30, 256), dtype=torch.float16, device="cuda"),
        ]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

        args = [e.requires_grad_(True) for e in args]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not torch.backends.cudnn.is_available(), "CUDNN is unavailable")
    @skipIfRocm  # https://github.com/pytorch/pytorch/issues/96560
    def test_batch_norm_amp(self):
        device = "cuda"
        input_dtype = torch.float16
        param_dtype = torch.float32
        weight, bias = (
            torch.ones(64, device=device, dtype=param_dtype, requires_grad=True)
            for _ in range(2)
        )
        running_mean, running_var = (
            torch.ones(64, device=device, dtype=param_dtype) for _ in range(2)
        )

        def bn(x):
            return torch.ops.aten.cudnn_batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                False,
                0.1,
                1e-05,
            )

        inp = torch.ones(
            torch.Size([16, 64, 112, 112]), dtype=input_dtype, device=device
        )

        ref = bn(inp)
        cudnn_batch_norm_decomp = torch._decomp.get_decompositions(
            {torch.ops.aten.cudnn_batch_norm}
        )
        aot_fn = make_fx(bn, decomposition_table=cudnn_batch_norm_decomp)(inp)
        res = aot_fn(inp)
        for a, b in zip(ref, res):
            assert torch.allclose(a, b)

    def test_output_op_depending_on_symint(self):
        """
        It won't be obvious from reading this test what it's testing for.  We should probably make it into a more
        focused unit test.

        An issue with the following program was the expand op would end up depending on a symint whose proxy was
        incorrectly associated with one of the grad tensors rather than input tensors.  It broke partitioner logic
        and the net result was aot_function failed to produce a function and threw an exception instead.
        """
        inp = torch.randn(5, requires_grad=True)

        def f(x):
            return x.expand(x.shape)

        # TODO(whc) make this work (test setup is wrong somehow)
        # joint_forward_backward = create_joint_forward_backward(f)
        # out = f(inp)
        # joint_inputs =  ([inp], [out.detach().contiguous()])
        # fx_g = make_fx(joint_forward_backward)(*joint_inputs)
        # TODO: assert outputs of fwd graph trace to correct symint

        # e2e test that fails without symint clone fix
        af = aot_function(
            f,
            nop,
            partition_fn=partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
            dynamic=True,
        )
        out = af(inp)
        self.assertEqual(out, f(inp))

    def test_inference_mode(self):
        m = torch.nn.Linear(4, 4)
        inp = torch.randn(4, 4)

        aot_mod = aot_module(m, fw_compiler=nop)

        with torch.inference_mode():
            out_ref = m(inp)
            out_test = aot_mod(inp)
        self.assertEqual(out_ref, out_test)

    def test_default_partitioner_saves_symints_not_tensors_for_bw(self):
        """
        In this test, the important thing is that primals_1 is **only** needed in the backward
        in order to grab its sizes.
        We need to assert that what we save for the backward are the tensor's sizes, and not the tensor itself.

        The way this test is set up, it will actually fail if we try to save the input tensor for backward.
        Why?
        b.masked_fill_(c, 0) has a backward that requires knowing a's sizes
        b.masked_fill_(c, 0) **also** mutates a (because b and a are aliased)
        The autograd engine yells at us if we save "a" for backward, and then try to mutate it.
        """

        def f(a):
            b = a[0]
            c = torch.ones_like(b, dtype=torch.bool)
            d = b.masked_fill_(c, 0)
            return d

        compiled_f = aot_function(f, nop, dynamic=True)
        inp_ref = torch.ones(2, 2, requires_grad=True)
        inp_test = torch.ones(2, 2, requires_grad=True)

        out_ref = f(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_copied_in_graph(self):
        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.w1 = torch.nn.Parameter(torch.zeros(1))
                self.w2 = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                self.buf.add_(1)
                return (self.w1 * x * self.w2).sum() + self.buf.sum()

        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)

        fw_graph_cell = [None]
        compiled_f = aot_module(
            model_for_compile,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp_ref = torch.ones(1, requires_grad=True)
        inp_test = torch.ones(1, requires_grad=True)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    add = torch.ops.aten.add.Tensor(primals_3, 1)
    mul = torch.ops.aten.mul.Tensor(primals_1, primals_4)
    mul_1 = torch.ops.aten.mul.Tensor(mul, primals_2)
    sum_1 = torch.ops.aten.sum.default(mul_1);  mul_1 = None
    sum_2 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    copy_ = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
    return (add_1, primals_1, primals_2, primals_4, mul)""",
        )

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        eager_grads = [p.grad for _, p in model_for_eager.named_parameters()]
        compile_grads = [p.grad for _, p in model_for_compile.named_parameters()]

        self.assertEqual(eager_grads, compile_grads)
        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_copied_in_graph_with_different_shapes(self):
        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(4, 4))
                self.w = torch.nn.Parameter(
                    torch.Tensor([[4, 5], [1, 2], [6, 7], [8, 9]])
                )

            def forward(self, x):
                self.buf.add_(1)
                return (self.w @ x).sum() + self.buf.sum()

        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)

        fw_graph_cell = [None]
        compiled_f = aot_module(
            model_for_compile,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=nop,
            keep_inference_input_mutations=True,
        )
        inp_ref = torch.ones(2, 4, requires_grad=True)
        inp_test = torch.ones(2, 4, requires_grad=True)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    add = torch.ops.aten.add.Tensor(primals_2, 1)
    mm = torch.ops.aten.mm.default(primals_1, primals_3)
    sum_1 = torch.ops.aten.sum.default(mm);  mm = None
    sum_2 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    copy_ = torch.ops.aten.copy_.default(primals_2, add);  primals_2 = add = copy_ = None
    return (add_1, primals_1, primals_3)""",
        )
        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        eager_grads = [p.grad for _, p in model_for_eager.named_parameters()]
        compile_grads = [p.grad for _, p in model_for_compile.named_parameters()]

        self.assertEqual(eager_grads, compile_grads)

        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_buffer_batch_norm(self):
        class MyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = torch.nn.BatchNorm1d(100)

            def forward(self, x):
                return self.m(x)

        model_for_eager = MyModel()
        model_for_compile = copy.deepcopy(model_for_eager)

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_f = aot_module(
            model_for_compile,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=bw_graph_cell)
            ),
            keep_inference_input_mutations=True,
        )
        inp_ref = torch.ones(20, 100, requires_grad=True)
        inp_test = torch.ones(20, 100, requires_grad=True)

        out_ref = model_for_eager(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6):
    add = torch.ops.aten.add.Tensor(primals_5, 1)
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(primals_6, primals_1, primals_2, primals_3, primals_4, True, 0.1, 1e-05);  primals_2 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_1 = _native_batch_norm_legit_functional[1]
    getitem_2 = _native_batch_norm_legit_functional[2]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    copy_ = torch.ops.aten.copy_.default(primals_3, getitem_3);  primals_3 = copy_ = None
    copy__1 = torch.ops.aten.copy_.default(primals_4, getitem_4);  primals_4 = copy__1 = None
    copy__2 = torch.ops.aten.copy_.default(primals_5, add);  primals_5 = add = copy__2 = None
    return (getitem, primals_1, primals_6, getitem_1, getitem_2, getitem_3, getitem_4)""",  # noqa: B950
        )

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        eager_grads = [p.grad for _, p in model_for_eager.named_parameters()]
        compile_grads = [p.grad for _, p in model_for_compile.named_parameters()]
        self.assertEqual(eager_grads, compile_grads)

        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_6, getitem_1, getitem_2, getitem_3, getitem_4, tangents_1):
    native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(tangents_1, primals_6, primals_1, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  tangents_1 = primals_6 = primals_1 = getitem_3 = getitem_4 = getitem_1 = getitem_2 = None
    getitem_5 = native_batch_norm_backward[0]
    getitem_6 = native_batch_norm_backward[1]
    getitem_7 = native_batch_norm_backward[2];  native_batch_norm_backward = None
    return (getitem_6, getitem_7, None, None, None, getitem_5)""",  # noqa: B950
        )

        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_new_inp_requires_grad_now(self):
        def f(x, y):
            return x.add_(y)

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_f = aot_function(
            f,
            fw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=fw_graph_cell)
            ),
            bw_compiler=make_boxed_compiler(
                partial(extract_graph, graph_cell=bw_graph_cell)
            ),
            keep_inference_input_mutations=True,
        )

        inp_ref = (
            torch.ones(20, 100, requires_grad=False),
            torch.ones(20, 100, requires_grad=True),
        )
        inp_test = (
            torch.ones(20, 100, requires_grad=False),
            torch.ones(20, 100, requires_grad=True),
        )

        out_ref = f(*inp_ref)
        out_test = compiled_f(*inp_test)

        # There is no copy_ method
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(clone, primals_2);  clone = primals_2 = None
    return (add, add)""",
        )  # noqa: B950

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, tangents_1):
    return (None, tangents_1)""",
        )  # noqa: B950

    def test_real_weights_in_symbolic_mode(self):
        from functorch.experimental import functionalize

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = M().eval()

        inp = torch.randn(2, 5)

        gm = make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)
        self.assertEqual(gm(torch.ones(2, 5)), m(torch.ones(2, 5)))

        gm_functionalized = make_fx(
            functionalize(
                gm,
            ),
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )(inp)
        self.assertEqual(gm_functionalized(torch.ones(2, 5)), m(torch.ones(2, 5)))

        inp_count = 0
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                inp_count += 1

        # No more param lifting
        self.assertEqual(inp_count, 1)

        inp_count = 0
        for node in gm_functionalized.graph.nodes:
            if node.op == "placeholder":
                inp_count += 1

        # No more param lifting
        self.assertEqual(inp_count, 1)

        with self.assertRaisesRegex(
            Exception, "Please convert all Tensors to FakeTensors"
        ):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=False)(
                torch.randn(2, 5)
            )

    def test_real_weights_in_symbolic_mode_with_inplace_ops(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 5))

            def forward(self, x):
                y = self.buffer.add_(3)
                y.resize_([20])
                assert y.shape == self.buffer.shape
                return x.sum() + self.buffer.sum()

        m = M().eval()
        inp = torch.randn(2, 5)
        # inplace mutation on attr is not allowed
        with self.assertRaisesRegex(Exception, "Can't call metadata"):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)

    def _compile_and_erase_bases(self, *output_view_indices):
        # Overrides _base and _view_func tensor attributes, so as to avoid the view-replay
        # execution path when reconstructing views.
        class NoViewReplayTensor(torch.Tensor):
            @property
            def _base(self):
                return None

            @property
            def _view_func(self):
                return None

        # Wraps the outputs that are views of the FX graph 'g' with NoViewReplayTensor,
        # since they are the only ones that will get reconstructed.
        def wrapper(g, *args, **kwargs):
            outs = list(g(*args, **kwargs))
            for i in output_view_indices:
                outs[i] = NoViewReplayTensor(outs[i])
            return tuple(outs)

        return lambda f: aot_function(f, fw_compiler=lambda g, _: partial(wrapper, g))

    def test_output_aliases_input_view_meta_replay(self):
        @self._compile_and_erase_bases(0)
        def f(a):
            return a.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out = f(inp)

        self.assertIsNotNone(out.grad_fn)
        self.assertExpectedInline(
            str(out.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_output_aliases_intermediate_view_meta_replay(self):
        @self._compile_and_erase_bases(0, 1)
        def f(a):
            b = a.clone()
            return b.view(-1), b.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp)

        self.assertIsNotNone(out1.grad_fn)
        self.assertExpectedInline(
            str(out1.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_output_aliases_output_view_meta_replay(self):
        @self._compile_and_erase_bases(1)
        def f(a):
            b = a.add(10)
            return b, b.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp)

        self.assertEqual(out1.untyped_storage(), out2.untyped_storage())
        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    @patch("torch._dynamo.config.assume_static_by_default", False)
    def test_dynamic_output_aliases_input_view_meta_replay(self):
        # - torch.compile: using it so we can have a SymInt in the FX graph.
        # - Compiling with inductor, so that tensor._base isn't tracked.
        #
        # This should force the use of as_strided in the view reconstruction path.
        # The first 2 view-replay paths won't be taken because:
        #   - target_functional_tensor will be symbolic (_functionalize_is_symbolic call)
        #   - tensor._base will be None
        @torch.compile(backend="inductor")
        def f(a, sz):
            return a.view(sz), a.view(-1)

        inp = torch.ones(2, 2, requires_grad=True)
        out1, out2 = f(inp, (4,))

        self.assertIsNotNone(out1.grad_fn)
        self.assertExpectedInline(
            str(out1.grad_fn.__class__), """<class 'AsStridedBackward0'>"""
        )

        self.assertIsNotNone(out2.grad_fn)
        self.assertExpectedInline(
            str(out2.grad_fn.__class__), """<class 'ViewBackward0'>"""
        )

    def test_duplicated_arguments_on_tensor_overlap(self):
        # Test whether we correctly handle duplicated arguments when changing the
        # parameters, so that we take the base tensor as argument.
        #
        # - t0 and t1 must have storage overlap: triggers the target execution flow.
        # - s0 and s1 must be equal: triggers the error in the target execution flow.

        @torch.compile(dynamic=True)
        def foo(t0, t1, s0, s1):
            return t0.add_(s0), t1.add_(s1)

        tensor = torch.rand(10)
        foo(tensor, tensor[1:-1], 2, 2)

    @parametrize("use_autograd", [False, True])
    def test_mark_outputs_dynamic(self, use_autograd: bool):
        counters.clear()
        torch._dynamo.reset()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y):
            return torch.matmul(x, y)

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn2(z):
            return z * 2

        # 1. static
        x = torch.randn(10, 10, requires_grad=use_autograd)
        y = torch.randn(10, 10, requires_grad=use_autograd)
        out = fn(x, y)
        self.assertFalse(hasattr(out, "_dynamo_weak_dynamic_indices"))
        out2 = fn2(out)
        self.assertFalse(hasattr(out2, "_dynamo_weak_dynamic_indices"))
        self.assertEqual(counters["aot_autograd"]["total"], 2)
        counters.clear()

        # 2. dynamic
        x = torch.randn(20, 20)
        y = torch.randn(20, 20)
        out = fn(x, y)
        self.assertTrue(hasattr(out, "_dynamo_weak_dynamic_indices"))
        out2 = fn2(out)
        self.assertTrue(hasattr(out2, "_dynamo_weak_dynamic_indices"))
        self.assertEqual(counters["aot_autograd"]["total"], 2)
        counters.clear()
        torch._dynamo.reset()

    def test_mark_activations_dynamic(self):
        counters.clear()
        torch._dynamo.reset()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y):
            out = torch.matmul(x, y)
            out2 = torch.matmul(out, y)
            out3 = torch.matmul(out2, y)
            return torch.matmul(out3, y)

        def make_assert_pack(dynamic):
            def pack(activation):
                assert hasattr(activation, "_dynamo_weak_dynamic_indices") == dynamic
                return activation

            return pack

        def make_assert_unpack(dynamic):
            def unpack(activation):
                assert hasattr(activation, "_dynamo_weak_dynamic_indices") == dynamic
                return activation

            return unpack

        # 1. static
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(False), make_assert_unpack(False)
        ):
            fn(x, y)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()

        # 2. dynamic
        x = torch.randn(20, 20, requires_grad=True)
        y = torch.randn(20, 20, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(True), make_assert_unpack(True)
        ):
            fn(x, y)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()
        torch._dynamo.reset()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @torch._functorch.config.patch(saved_tensors_hooks_filtering_mode="no_static")
    @torch._functorch.config.patch(recompute_views=True)
    def test_saved_tensors_hooks_mutations_raise(self):
        ctx = torch.autograd.graph.saved_tensors_hooks
        device = "cuda"

        class SAF(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gx):
                (saved_x,) = ctx.saved_tensors
                return gx + saved_x

        def mutate(x):
            return x.mul_(2)

        def fn(x):
            x = 2 * x
            x = SAF.apply(x)
            return x

        def inp_fn():
            x = torch.ones(2, 3, device=device, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return x

        with self.assertRaisesRegex(
            AssertionError, "Saved tensors hooks with inputs mutations are not allowed"
        ):
            try:
                with ctx(*saved_tensors_hooks_to_gm(mutate, mutate, None, None)):
                    x = inp_fn()
                    y = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
                    y.sum().backward()
            except torch._dynamo.exc.BackendCompilerFailed as e:
                raise e.inner_exception from e

    def test_mark_activations_dynamic_with_nested(self):
        # The flattened tensors of the nested tensor aren't
        # marked as activations, but they add some offset
        # to the fw_outs. This test ensures that we handle
        # that offset properly.
        counters.clear()
        torch._dynamo.reset()

        def make_assert_pack(dynamic):
            def pack(activation):
                assert hasattr(activation, "_dynamo_weak_dynamic_indices") == dynamic
                return activation

            return pack

        def make_assert_unpack(dynamic):
            def unpack(activation):
                assert hasattr(activation, "_dynamo_weak_dynamic_indices") == dynamic
                return activation

            return unpack

        # 1. static
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x, y, nt):
            out = torch.matmul(x, y)
            return out.sum() + nt.clone()

        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(False), make_assert_unpack(False)
        ):
            fn(x, y, nt)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()

        # 2. dynamic
        x = torch.randn(20, 20, requires_grad=True)
        y = torch.randn(20, 20, requires_grad=True)
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        with torch.autograd.graph.saved_tensors_hooks(
            make_assert_pack(True), make_assert_unpack(True)
        ):
            fn(x, y, nt)
        self.assertEqual(counters["aot_autograd"]["total"], 1)
        counters.clear()
        torch._dynamo.reset()


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


def get_ins_outs(fx_g):
    ins = []
    outs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            ins.append(n)
        elif n.op == "output":
            outs = tuple(n.args[0])
    return ins, outs


def get_num_ins_outs(fx_g):
    return tuple(len(i) for i in get_ins_outs(fx_g))


def get_fw_bw_graph(
    f, inps, partitioner=min_cut_rematerialization_partition, dynamic=False
):
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    aot_function(
        f,
        fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
        bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
        partition_fn=partitioner,
        decompositions=default_decompositions,
        dynamic=dynamic,
    )(*inps).sum().backward()
    return (fw_graph_cell[0], bw_graph_cell[0])


class TestMod(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2, requires_grad=True))
        self.fn = fn

    def forward(self, *args):
        return self.fn(self.p, *args)


class TestAOTExport(AOTTestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def test_aot_export_ban_dropout_mut_pre_dispatch(self):
        def fn(p, x):
            y = torch.ops.aten.dropout.default(x, 0.1, train=False)
            y.add_(1)
            return (y,)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    clone = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    return (add,)""",
        )

        fw_graph_cell = [None]
        bw_graph_cell = [None]

        aot_function(
            fn,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=default_partition,
            decompositions=default_decompositions,
            dynamic=True,
        )(*inp)
        fw_graph = fw_graph_cell[0]

        self.assertExpectedInline(
            str(fw_graph.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    clone = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    return (add,)""",
        )

    def test_aot_export_predispatch_func_simple(self):
        def fn(p, x):
            y = x + 2
            with torch.no_grad():
                y.add_(2)
            return (x * 2 + y,)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        with torch.no_grad():
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg1_1, 2)
    _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None
    add_1 = torch.ops.aten.add.Tensor(add, 2);  add = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    mul = torch.ops.aten.mul.Tensor(arg1_1, 2);  arg1_1 = None
    add_2 = torch.ops.aten.add.Tensor(mul, add_1);  mul = add_1 = None
    return (add_2,)""",
        )

    def test_aot_export_predispatch_func_composite_implicit(self):
        def fn(p, x):
            with torch.enable_grad():
                y = x @ x
            y.add_(2)
            return (x.sum() + y.sum(),)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        with torch.no_grad():
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _set_grad_enabled = torch._C._set_grad_enabled(True);  _set_grad_enabled = None
    matmul = torch.ops.aten.matmul.default(arg1_1, arg1_1)
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    add = torch.ops.aten.add.Tensor(matmul, 2);  matmul = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    sum_2 = torch.ops.aten.sum.default(add);  add = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_1,)""",
        )

    def test_aot_export_predispatch_composite_implicit_inplace(self):
        def fn(x, p):
            return (torch.ops.aten.absolute_.default(x.clone()),)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    abs_1 = torch.ops.aten.abs.default(clone);  clone = None
    return (abs_1,)""",
        )

    def test_aot_export_predispatch_composite_implicit_linear(self):
        class MM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return (self.linear(x),)

        mod = MM()
        inp = torch.randn(2, 2)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    linear = torch.ops.aten.linear.default(arg2_1, arg0_1, arg1_1);  arg2_1 = arg0_1 = arg1_1 = None
    return (linear,)""",
        )

    @unittest.expectedFailure
    def test_aot_export_predispatch_outdtype(self):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                y = x + 2
                y.add_(5)
                return (
                    out_dtype(torch.ops.aten.mm.default, torch.int32, y, self.weight),
                )

        weight = torch.randint(-128, 127, (5, 5), dtype=torch.int8)
        mod = M(weight)
        inp = torch.randint(-128, 127, (5, 5), dtype=torch.int8)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _set_grad_enabled = torch._C._set_grad_enabled(True);  _set_grad_enabled = None
    mm = torch.ops.aten.mm.default(arg1_1, arg1_1)
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    add = torch.ops.aten.add.Tensor(mm, 2);  mm = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    sum_2 = torch.ops.aten.sum.default(add);  add = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_1,)""",
        )

    def test_aot_export_predispatch_func_view(self):
        def fn(p, x):
            y = x @ x
            y.add_(2)
            return (x.sum() + y.view(1, 4).sum(),)

        mod = TestMod(fn)
        inp = torch.randn(2, 2)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    matmul = torch.ops.aten.matmul.default(arg1_1, arg1_1)
    add = torch.ops.aten.add.Tensor(matmul, 2);  matmul = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(add, [1, 4]);  add = None
    sum_2 = torch.ops.aten.sum.default(view_1);  view_1 = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add_1,)""",
        )

    def test_aot_export_predispatch_buffer_mutation_metadata(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Buffer(torch.zeros(2, 2))

            def forward(self, x):
                self.foo.add_(4)
                return (x.sum() + self.foo.sum(),)

        inp = torch.randn(2, 2)

        gm, graph_sig = aot_export_module(
            Foo(), [inp], trace_joint=False, pre_dispatch=True
        )
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 4);  arg0_1 = None
    sum_1 = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
    sum_2 = torch.ops.aten.sum.default(add)
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add, add_1)""",
        )
        eager_mod = Foo()
        output_1, output_2 = gm(torch.zeros(2, 2), inp)
        eager_output = eager_mod(inp)
        self.assertTrue(torch.allclose(output_2, eager_output[0]))

        _, output_2 = gm(output_1, inp)
        eager_output = eager_mod(inp)
        self.assertTrue(torch.allclose(output_2, eager_output[0]))
        self.assertTrue("foo" in graph_sig.buffers)
        self.assertTrue(graph_sig.inputs_to_buffers["arg0_1"] == "foo")

    def test_aot_export_predispatch_with_autograd_op(self):
        def foo(p, x):
            with torch.enable_grad():
                y = x + 5
                y.add_(5)
                y.add_(7)
                return (x.cos() + y.sin(),)

        inp = torch.randn(2, 2)
        mod = TestMod(foo)

        with torch.no_grad():
            gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    _set_grad_enabled = torch._C._set_grad_enabled(True);  _set_grad_enabled = None
    add = torch.ops.aten.add.Tensor(arg1_1, 5)
    add_1 = torch.ops.aten.add.Tensor(add, 5);  add = None
    add_2 = torch.ops.aten.add.Tensor(add_1, 7);  add_1 = None
    cos = torch.ops.aten.cos.default(arg1_1);  arg1_1 = None
    sin = torch.ops.aten.sin.default(add_2);  add_2 = None
    add_3 = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    return (add_3,)""",
        )

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torchdynamo.is_dynamo_supported(), "TorchDynamo is not supported"
    )
    def test_aot_export_predispatch_with_cond_nested(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                def true_fn(x):
                    y = x.sin()
                    y.add_(5)

                    def true_true_fn(x):
                        y = x.sin()
                        y.add_(7)
                        return y.sin()

                    def true_false_fn(x):
                        return x.cos()

                    return torch.cond(
                        y.cos().sum() > 5, true_true_fn, true_false_fn, [y.cos()]
                    )

                def false_fn(x):
                    z = x.cos()
                    z.add_(6)
                    return z.sin()

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x])
                return (a + 3, a + 4)

        inp = torch.randn(2, 2)
        gm, _ = aot_export_module(M(), [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1,));  gt = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    add = torch.ops.aten.add.Tensor(getitem, 3)
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    return (add, add_1)""",  # noqa: B950
        )

        self.assertExpectedInline(
            str(gm.true_graph_0.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(sin, 5);  sin = None
    cos = torch.ops.aten.cos.default(add)
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    gt = torch.ops.aten.gt.Scalar(sum_1, 5);  sum_1 = None
    cos_1 = torch.ops.aten.cos.default(add);  add = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (cos_1,));  gt = true_graph_0 = false_graph_0 = cos_1 = None
    getitem = cond[0];  cond = None
    return (getitem,)""",  # noqa: B950
        )

        self.assertExpectedInline(
            str(gm.true_graph_0.true_graph_0.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(sin, 7);  sin = None
    sin_1 = torch.ops.aten.sin.default(add);  add = None
    return (sin_1,)""",
        )

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torchdynamo.is_dynamo_supported(), "TorchDynamo is not supported"
    )
    def test_aot_export_predispatch_map_1(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                def true_fn(x, r):
                    y = x.sin()
                    y.add_(5)
                    return y.cos() + r.sum()

                def false_fn(x, r):
                    z = x.cos()

                    def f(x, y):
                        a = x.cos()
                        a.add_(5)
                        return a + y

                    return (
                        z
                        + control_flow.map(f, z, r).sum()
                        + control_flow.map(f, z, r).sum()
                    )

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x, y])
                return (a + 3, a + 4)

        inps = [torch.randn(2, 2), torch.ones(2)]
        gm, _ = aot_export_module(M(), inps, trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(False, expanded_def=True)),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[2, 2]",  # PlainAOTInput(idx=0)
        arg1_1: "f32[2]",  # PlainAOTInput(idx=1)
    ):
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1, arg1_1));  gt = true_graph_0 = false_graph_0 = arg0_1 = arg1_1 = None
        getitem: "f32[2, 2]" = cond[0];  cond = None

        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem, 3)
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
        return (
            add,  # PlainAOTOutput(idx=0)
            add_1,  # PlainAOTOutput(idx=1)
        )

    class true_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2]"):
            sin: "f32[2, 2]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None

            add: "f32[2, 2]" = torch.ops.aten.add.Tensor(sin, 5);  sin = None

            cos: "f32[2, 2]" = torch.ops.aten.cos.default(add);  add = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None

            add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
            return (add_1,)

    class false_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2]"):
            cos: "f32[2, 2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

            body_graph_0 = self.body_graph_0
            map_impl = torch.ops.higher_order.map_impl(body_graph_0, [cos], [arg1_1]);  body_graph_0 = None
            getitem_2: "f32[2, 2]" = map_impl[0];  map_impl = None

            sum_1: "f32[]" = torch.ops.aten.sum.default(getitem_2);  getitem_2 = None

            add: "f32[2, 2]" = torch.ops.aten.add.Tensor(cos, sum_1);  sum_1 = None

            body_graph_1 = self.body_graph_1
            map_impl_1 = torch.ops.higher_order.map_impl(body_graph_1, [cos], [arg1_1]);  body_graph_1 = cos = arg1_1 = None
            getitem_5: "f32[2, 2]" = map_impl_1[0];  map_impl_1 = None

            sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_5);  getitem_5 = None

            add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, sum_2);  add = sum_2 = None
            return (add_1,)

        class body_graph_0(torch.nn.Module):
            def forward(self, arg0_1: "f32[2]", arg1_1: "f32[2]"):
                cos: "f32[2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

                add: "f32[2]" = torch.ops.aten.add.Tensor(cos, 5);  cos = None

                add_1: "f32[2]" = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
                return (add_1,)

        class body_graph_1(torch.nn.Module):
            def forward(self, arg0_1: "f32[2]", arg1_1: "f32[2]"):
                cos: "f32[2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

                add: "f32[2]" = torch.ops.aten.add.Tensor(cos, 5);  cos = None

                add_1: "f32[2]" = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
                return (add_1,)
""",  # noqa: B950
        )

    def test_aot_export_predispatch_map_2(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, y):
                z = x.cos()

                def f(x, y):
                    a = x.cos()
                    a.add_(5)
                    return a + y

                return (z + control_flow.map(f, z, y).sum(),)

        inps = [torch.randn(2, 2), torch.ones(2)]
        gm, _ = aot_export_module(M(), inps, trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            normalize_gm(gm.print_readable(False, expanded_def=True)),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[2, 2]",  # PlainAOTInput(idx=0)
        arg1_1: "f32[2]",  # PlainAOTInput(idx=1)
    ):
        cos: "f32[2, 2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

        body_graph_0 = self.body_graph_0
        map_impl = torch.ops.higher_order.map_impl(body_graph_0, [cos], [arg1_1]);  body_graph_0 = arg1_1 = None
        getitem_2: "f32[2, 2]" = map_impl[0];  map_impl = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem_2);  getitem_2 = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
        return (
            add,  # PlainAOTOutput(idx=0)
        )

    class body_graph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[2]", arg1_1: "f32[2]"):
            cos: "f32[2]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None

            add: "f32[2]" = torch.ops.aten.add.Tensor(cos, 5);  cos = None

            add_1: "f32[2]" = torch.ops.aten.add.Tensor(add, arg1_1);  add = arg1_1 = None
            return (add_1,)
""",
        )

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torchdynamo.is_dynamo_supported(), "TorchDynamo is not supported"
    )
    def test_aot_export_predispatch_with_cond(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                def true_fn(x):
                    y = x.sin()
                    z = torch.ops.aten.linear.default(y, torch.randn(2, 2))
                    z.add_(5)
                    return z.cos()

                def false_fn(x):
                    z = x.cos()
                    z.add_(6)
                    return z.sin()

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x])
                return (a + 3, a + 4)

        inp = torch.randn(2, 2)
        gm, _ = aot_export_module(M(), [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1,));  gt = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    add = torch.ops.aten.add.Tensor(getitem, 3)
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    return (add, add_1)""",  # noqa: B950
        )
        self.assertExpectedInline(
            str(gm.true_graph_0.code).strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    randn = torch.ops.aten.randn.default([2, 2], device = device(type='cpu'), pin_memory = False)
    linear = torch.ops.aten.linear.default(sin, randn);  sin = randn = None
    add = torch.ops.aten.add.Tensor(linear, 5);  linear = None
    cos = torch.ops.aten.cos.default(add);  add = None
    return (cos,)""",
        )

    def test_aot_export_predispatch_conv_and_bn(self):
        class ConvBatchnorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return (x,)

        mod = ConvBatchnorm()
        mod.train()
        inp = torch.randn(1, 1, 3, 3)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1):
    conv2d = torch.ops.aten.conv2d.default(arg7_1, arg0_1, arg1_1);  arg7_1 = arg0_1 = arg1_1 = None
    add = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(conv2d, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  conv2d = arg2_1 = arg3_1 = arg4_1 = arg5_1 = None
    getitem = _native_batch_norm_legit_functional[0]
    getitem_3 = _native_batch_norm_legit_functional[3]
    getitem_4 = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    return (getitem_3, getitem_4, add, getitem)""",  # noqa: B950
        )

    def test_aot_export_predispatch_reshape(self):
        class Reshape(torch.nn.Module):
            def forward(self, x):
                y = x.reshape(4, 4)
                return (y.sum(),)

        mod = Reshape()
        inp = torch.randn(2, 8)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    view = torch.ops.aten.view.default(arg0_1, [4, 4]);  arg0_1 = None
    sum_1 = torch.ops.aten.sum.default(view);  view = None
    return (sum_1,)""",
        )  # noqa: B950

    def test_aot_export_predispatch_contiguous(self):
        class Cont(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.aten.contiguous.default(x)
                return (y.sum(),)

        mod = Cont()
        inp = torch.randn(2, 8)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False, pre_dispatch=True)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    return (sum_1,)""",
        )  # noqa: B950

    def test_aot_export_module_joint(self):
        class ConvBatchnormRelu(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                user_out = torch.nn.functional.relu(x)
                loss = user_out.sum()
                return loss, user_out.detach()

        mod = ConvBatchnormRelu()
        mod.train()
        inp = torch.randn(1, 1, 3, 3)
        mod(inp)
        fx_g, signature = aot_export_module(
            mod, [inp], trace_joint=True, output_loss_index=0
        )
        # Some important characteristics of the exported graph below:
        # 8 arguments: 2 params from conv, 2 params from batchnorm, 2 buffers from 1 batchnorm, 1 user input
        # 9 outputs: 3 mutated buffers (from batchnorm), 2 user outputs and 4 gradients (since there were 4 parameters)
        for node in fx_g.graph.nodes:
            node.meta.pop("stack_trace", None)
        self.assertExpectedInline(
            fx_g.print_readable(print_output=False, expanded_def=True),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[3, 1, 1, 1]",
        arg1_1: "f32[3]",
        arg2_1: "f32[3]",
        arg3_1: "f32[3]",
        arg4_1: "f32[3]",
        arg5_1: "f32[3]",
        arg6_1: "i64[]",
        arg7_1: "f32[1, 1, 3, 3]",
    ):
        # No stacktrace found for following nodes
        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(arg7_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg1_1 = None
        add: "i64[]" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  arg3_1 = arg4_1 = arg5_1 = None
        getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
        getitem_1: "f32[3]" = _native_batch_norm_legit_functional[1]
        getitem_2: "f32[3]" = _native_batch_norm_legit_functional[2]
        getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
        getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
        relu: "f32[1, 3, 3, 3]" = torch.ops.aten.relu.default(getitem);  getitem = None
        detach: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu);  detach = None
        detach_1: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu)
        sum_1: "f32[]" = torch.ops.aten.sum.default(relu)
        detach_2: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu);  relu = None
        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)
        expand: "f32[1, 3, 3, 3]" = torch.ops.aten.expand.default(ones_like, [1, 3, 3, 3]);  ones_like = None
        detach_3: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        threshold_backward: "f32[1, 3, 3, 3]" = torch.ops.aten.threshold_backward.default(expand, detach_3, 0);  expand = detach_3 = None
        native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(threshold_backward, convolution, arg2_1, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  threshold_backward = convolution = arg2_1 = getitem_1 = getitem_2 = None
        getitem_5: "f32[1, 3, 3, 3]" = native_batch_norm_backward[0]
        getitem_6: "f32[3]" = native_batch_norm_backward[1]
        getitem_7: "f32[3]" = native_batch_norm_backward[2];  native_batch_norm_backward = None
        convolution_backward = torch.ops.aten.convolution_backward.default(getitem_5, arg7_1, arg0_1, [3], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  getitem_5 = arg7_1 = arg0_1 = None
        getitem_8 = convolution_backward[0];  getitem_8 = None
        getitem_9: "f32[3, 1, 1, 1]" = convolution_backward[1]
        getitem_10: "f32[3]" = convolution_backward[2];  convolution_backward = None
        return (getitem_3, getitem_4, add, sum_1, detach_2, getitem_9, getitem_10, getitem_6, getitem_7)
        """,  # noqa: B950
        )

        self.assertExpectedInline(
            str(signature.parameters),
            """['conv.weight', 'conv.bias', 'bn.weight', 'bn.bias']""",
        )
        self.assertExpectedInline(
            str(signature.buffers),
            """['bn.running_mean', 'bn.running_var', 'bn.num_batches_tracked']""",
        )
        self.assertExpectedInline(str(signature.user_inputs), """['arg7_1']""")
        self.assertExpectedInline(
            str(signature.inputs_to_parameters),
            """{'arg0_1': 'conv.weight', 'arg1_1': 'conv.bias', 'arg2_1': 'bn.weight', 'arg3_1': 'bn.bias'}""",
        )  # noqa: B950
        self.assertExpectedInline(
            str(signature.inputs_to_buffers),
            """{'arg4_1': 'bn.running_mean', 'arg5_1': 'bn.running_var', 'arg6_1': 'bn.num_batches_tracked'}""",
        )  # noqa: B950
        self.assertExpectedInline(
            str(signature.buffers_to_mutate),
            """{'getitem_3': 'bn.running_mean', 'getitem_4': 'bn.running_var', 'add': 'bn.num_batches_tracked'}""",
        )  # noqa: B950
        self.assertExpectedInline(
            str(signature.backward_signature.gradients_to_parameters),
            """{'getitem_9': 'conv.weight', 'getitem_10': 'conv.bias', 'getitem_6': 'bn.weight', 'getitem_7': 'bn.bias'}""",
        )  # noqa: B950
        self.assertExpectedInline(
            str(signature.backward_signature.gradients_to_user_inputs), """{}"""
        )
        self.assertExpectedInline(
            str(signature.backward_signature.loss_output), """getitem_3"""
        )

        # Also check the inference graph
        # Main important thing here is that there are 5 total outputs: 3 total mutated buffers (from batchnorm), 2 user outputs.
        fx_g_inference, signature_inference = aot_export_module(
            mod, [inp], trace_joint=False
        )
        for node in fx_g_inference.graph.nodes:
            node.meta.pop("stack_trace", None)
        self.assertExpectedInline(
            fx_g_inference.print_readable(print_output=False, expanded_def=True),
            """\
class <lambda>(torch.nn.Module):
    def forward(
        self,
        arg0_1: "f32[3, 1, 1, 1]",  # PlainAOTInput(idx=0)
        arg1_1: "f32[3]",  # PlainAOTInput(idx=1)
        arg2_1: "f32[3]",  # PlainAOTInput(idx=2)
        arg3_1: "f32[3]",  # PlainAOTInput(idx=3)
        arg4_1: "f32[3]",  # PlainAOTInput(idx=4)
        arg5_1: "f32[3]",  # PlainAOTInput(idx=5)
        arg6_1: "i64[]",  # PlainAOTInput(idx=6)
        arg7_1: "f32[1, 1, 3, 3]",  # PlainAOTInput(idx=7)
    ):
        # No stacktrace found for following nodes
        convolution: "f32[1, 3, 3, 3]" = torch.ops.aten.convolution.default(arg7_1, arg0_1, arg1_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg7_1 = arg0_1 = arg1_1 = None
        add: "i64[]" = torch.ops.aten.add.Tensor(arg6_1, 1);  arg6_1 = None
        _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, arg2_1, arg3_1, arg4_1, arg5_1, True, 0.1, 1e-05);  convolution = arg2_1 = arg3_1 = arg4_1 = arg5_1 = None
        getitem: "f32[1, 3, 3, 3]" = _native_batch_norm_legit_functional[0]
        getitem_3: "f32[3]" = _native_batch_norm_legit_functional[3]
        getitem_4: "f32[3]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
        relu: "f32[1, 3, 3, 3]" = torch.ops.aten.relu.default(getitem);  getitem = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(relu)
        detach: "f32[1, 3, 3, 3]" = torch.ops.aten.detach.default(relu);  relu = None
        return (
            getitem_3,  # InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=4))
            getitem_4,  # InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=5))
            add,  # InputMutationAOTOutput(mutated_input=PlainAOTInput(idx=6))
            sum_1,  # PlainAOTOutput(idx=0)
            detach,  # PlainAOTOutput(idx=1)
        )
        """,  # noqa: B950
        )
        # Some important characteristics of the exported graph below:
        # 8 arguments: 2 params from conv, 2 params from batchnorm, 2 buffers from 1 batchnorm, 1 user input
        # 9 outputs: 2 mutated buffers (from batchnorm), 2 user outputs and 4 gradients (since there were 4 parameters)

    def test_aot_export_simplified_basic(self):
        def f(x, y):
            return x * y, y * y.detach()

        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)

        f_graph_fw = aot_export_joint_simple(f, [x, y], trace_joint=False)
        out_ref = f(x, y)
        # No calling convention changes necessary to invoke the traced graph
        out_test = f_graph_fw(x, y)
        self.assertEqual(out_ref, out_test)

        # Now test the backward
        x = torch.randn(2, requires_grad=True)
        y = torch.randn(2, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)
        x3 = x.detach().clone().requires_grad_(True)
        y3 = y.detach().clone().requires_grad_(True)
        f_graph_joint = aot_export_joint_simple(f, [x, y], trace_joint=True)
        num_fw_outputs = 2
        fw_g, bw_g = default_partition(
            f_graph_joint, [x, y], num_fwd_outputs=num_fw_outputs
        )
        out_ref2 = f(x2, y2)
        fw_outs = fw_g(x3, y3)
        out_test2, activations = fw_outs[:num_fw_outputs], fw_outs[num_fw_outputs:]
        self.assertEqual(out_ref2, out_test2)

        # Test running the traced backward graph with a mocked-up grad_output
        grad_outs = [torch.ones_like(x) for x in out_ref2]
        grads_ref = torch.autograd.grad(out_ref2, [x2, y2], grad_outputs=grad_outs)
        grads_test = bw_g(*activations, *grad_outs)
        for g_ref, g_test in zip(grads_ref, grads_test):
            self.assertEqual(g_ref, g_test)

    def test_aot_export_metadata_mutation_banned(self):
        def fn(p, x):
            x.t_()
            return (x * 2,)

        mod = TestMod(fn)
        inp = torch.randn(2, 4)
        with self.assertRaisesRegex(
            RuntimeError, "Found an input that received a metadata mutation"
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            aot_export_module(mod, [inp], trace_joint=False)

    def test_aot_export_forward_mutation_no_buffer_mut(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer1 = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x):
                x.add_(4)
                return (x.cos().sum() + self.buffer1.sum(),)

        mod = M()
        inp = torch.ones(6, 4)
        gm, sig = aot_export_module(mod, [inp], trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg1_1, 4);  arg1_1 = None
    cos = torch.ops.aten.cos.default(add)
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    sum_2 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    return (add, add_1)""",
        )  # noqa: B950
        self.assertEqual(sig.user_inputs_to_mutate, {"add": "arg1_1"})

    def test_aot_export_forward_mutation_multiple_mut(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer1 = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self, x, y):
                y.add_(4)
                self.buffer1.add_(5)
                return (
                    x.cos().sum() + y.sin().sum(),
                    self.buffer1.sum(),
                )

        mod = M()
        inp = [torch.ones(6, 4), torch.zeros(6, 4)]
        gm, sig = aot_export_module(mod, inp, trace_joint=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    add = torch.ops.aten.add.Tensor(arg2_1, 4);  arg2_1 = None
    add_1 = torch.ops.aten.add.Tensor(arg0_1, 5);  arg0_1 = None
    cos = torch.ops.aten.cos.default(arg1_1);  arg1_1 = None
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    sin = torch.ops.aten.sin.default(add)
    sum_2 = torch.ops.aten.sum.default(sin);  sin = None
    add_2 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    sum_3 = torch.ops.aten.sum.default(add_1)
    return (add_1, add, add_2, sum_3)""",
        )  # noqa: B950
        self.assertEqual(sig.user_inputs_to_mutate, {"add": "arg2_1"})
        self.assertEqual(sig.buffers_to_mutate, {"add_1": "buffer1"})

    def test_aot_export_input_mutation_on_input_requiring_grad_banned(self):
        class M(torch.nn.Module):
            def forward(self, x):
                x.add_(4)
                return (x,)

        mod = M()
        inp = torch.randn(2, requires_grad=True)
        gm, _ = aot_export_module(mod, [inp], trace_joint=False)
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 4), kwargs = {})
    return (add, add)""",
        )

    def test_aot_export_input_mutation_on_parameter_banned(self):
        def fn(p, x):
            p.mul_(2)
            return (p + x,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_export_joint_simple does not support input mutations. ViewAndMutationMeta",
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
        with self.assertRaisesRegex(
            RuntimeError,
            "Found a graph input that requires gradients, and received a mutation",
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)

        gm, _ = aot_export_module(mod, [inp], trace_joint=False)
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg1_1), kwargs = {})
    return (mul, add)""",
        )

    def test_aot_export_synthetic_bases_banned(self):
        def fn(p, x, y):
            x.mul_(2)
            return (x + y,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        inp2 = inp.view(-1)
        with self.assertRaisesRegex(
            RuntimeError, "Encountered aliased inputs that are mutated"
        ):
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp, inp2], trace_joint=True)
            aot_export_module(mod, [inp, inp2], trace_joint=False)

    def test_aot_export_input_dupes_banned(self):
        def fn(p, x, y):
            x.mul_(2)
            return (x + y,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(
            RuntimeError, "Encountered duplicated inputs that are mutated in the graph"
        ):
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp, inp], trace_joint=True)
            aot_export_module(mod, [inp, inp], trace_joint=False)

    def test_aot_export_multiple_outputs_require_grad_banned(self):
        def fn(p, x):
            out = p * x
            return out, out.sum()

        mod = TestMod(fn)
        inp = torch.randn(2)
        with self.assertRaisesRegex(
            RuntimeError,
            "Found an output of the forward that requires gradients, that was not",
        ):
            aot_export_module(mod, [inp], trace_joint=True, output_loss_index=1)

    @unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
    @unittest.skipIf(
        not torch._dynamo.is_dynamo_supported(), "Cond needs dynamo to run"
    )
    def test_aot_export_with_torch_cond(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                def true_fn(x):
                    y = x + 4
                    y.add_(5)
                    return x.cos()

                def false_fn(x):
                    y = x + 5
                    y.add_(6)
                    return x.sin()

                a = torch.cond(x.sum() > 4, true_fn, false_fn, [x])
                return (a + 3, a + 4)

        inp = torch.randn(3, 4)
        gm, _ = aot_export_module(M(), (inp,), trace_joint=False)
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1)
    gt = torch.ops.aten.gt.Scalar(sum_1, 4);  sum_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (arg0_1,));  gt = true_graph_0 = false_graph_0 = arg0_1 = None
    getitem = cond[0];  cond = None
    add = torch.ops.aten.add.Tensor(getitem, 3)
    add_1 = torch.ops.aten.add.Tensor(getitem, 4);  getitem = None
    return (add, add_1)""",  # noqa: B950
        )

        self.assertExpectedInline(
            gm.true_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    cos = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
    return (cos,)""",
        )

        self.assertExpectedInline(
            gm.false_graph_0.code.strip(),
            """\
def forward(self, arg0_1):
    sin = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
    return (sin,)""",
        )

    def test_aot_export_simplified_pytrees_banned(self):
        def fn(inps):
            return (inps[0] + inps[1],)

        inp1 = torch.randn(2)
        inp2 = torch.randn(2)
        inps = [inp1, inp2]
        with self.assertRaisesRegex(
            RuntimeError,
            "aot_export_joint_simple requires individual inputs not to be pytrees",
        ):
            aot_export_joint_simple(fn, [inps], trace_joint=False)
            aot_export_joint_simple(fn, [inps], trace_joint=True)

    def test_aot_export_functionalized_rng_banned(self):
        def fn(p, x):
            return (p + x,)

        mod = TestMod(fn)
        inp = torch.randn(2)
        with (
            patch("functorch.compile.config.functionalize_rng_ops", True),
            self.assertRaisesRegex(
                RuntimeError,
                "Functionalized RNG is not currently supported in the aot_export",
            ),
        ):
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=False)
            aot_export_joint_simple(fn, [mod.p, inp], trace_joint=True)
            aot_export_module(mod, [inp], trace_joint=False)

    def test_aot_export_unbacked_arg(self):
        class M(torch.nn.Module):
            def forward(self):
                full = torch.full((), 11)
                i0 = full.item()
                return (torch.full((i0,), 0),)

        gm, _ = aot_export_module(
            mod=M(), args=(), trace_joint=False, dynamic_shapes=True
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self):
    full = torch.ops.aten.full.default([], 11, device = device(type='cpu'), pin_memory = False)
    _local_scalar_dense = torch.ops.aten._local_scalar_dense.default(full);  full = None
    full_1 = torch.ops.aten.full.default([_local_scalar_dense], 0, device = device(type='cpu'), pin_memory = False);  _local_scalar_dense = None
    return (full_1,)""",  # noqa: B950
        )

    def test_aot_export_input_mutation(self):
        def f(x, buf):
            buf.add_(1)
            return buf * x

        x = torch.randn(2, requires_grad=True)
        buf = torch.zeros(2, requires_grad=False)

        gm, _, _, _ = _aot_export_function(
            f,
            (x, buf),
            decompositions={},
            num_params_buffers=1,
            no_tangents=False,
            pre_dispatch=False,
            dynamic_shapes=False,
            keep_input_mutations=True,
            kwargs={},
        )
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, primals, tangents):
    primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    add = torch.ops.aten.add.Tensor(primals_2, 1)
    mul = torch.ops.aten.mul.Tensor(add, primals_1);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, add);  tangents_1 = None
    copy_ = torch.ops.aten.copy_.default(primals_2, add);  primals_2 = add = copy_ = None
    return pytree.tree_unflatten([mul, mul_1, None], self._out_spec)""",
        )


class TestPartitioning(AOTTestCase):
    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_recompute_partitioning(self):
        def fn(a, b):
            return torch.sin(torch.sin(a)) + b

        # Reference calculation
        ref_a = torch.rand(10, 10, requires_grad=True)
        ref_b = torch.rand(10, 10, requires_grad=True)
        ref = fn(ref_a, ref_b)
        ref.sum().backward()

        # Compiled function calculation
        res_a = ref_a.detach().clone().requires_grad_(True)
        res_b = ref_b.detach().clone().requires_grad_(True)

        def compile_fn(x, _):
            return x

        compiled_fn = compiled_function(
            fn, compile_fn, compile_fn, min_cut_rematerialization_partition
        )
        res = compiled_fn(res_a, res_b)
        res.sum().backward()
        assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)
        assert torch.allclose(ref_a.grad, res_a.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(ref_b.grad, res_b.grad, atol=1e-3, rtol=1e-3)

    def test_meta_tensor_inplace_op(self):
        # Following module results in inplace ops while tracing. The test checks
        # that the meta tensor information is stored for inplace ops.
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(3072, 768, requires_grad=True)
                )
                self.bias = torch.nn.Parameter(torch.randn(3072, requires_grad=True))

            def forward(self, add_4):
                linear_4 = torch.nn.functional.linear(
                    add_4, self.weight, bias=self.bias
                )
                gelu = torch.nn.functional.gelu(linear_4)
                return gelu

        def check_meta_tensor(fx_g, _):
            for node in fx_g.graph.nodes:
                if node.op != "output":
                    assert "tensor_meta" in node.meta
            return fx_g

        inp0 = torch.randn(16, 128, 768, requires_grad=True)
        inputs = [
            inp0,
        ]
        mod = MockModule().to(device="cpu")
        aot_mod = aot_module(mod, fw_compiler=check_meta_tensor)
        aot_mod(*inputs)

    def test_default_partitioner_getitem(self):
        mod = nn.LayerNorm([10])

        def f(x, mod_weight, mod_bias):
            return torch.nn.functional.layer_norm(
                x, [10], mod_weight, mod_bias, eps=1e-6
            )

        fw_graph, bw_graph = get_fw_bw_graph(
            f,
            [torch.randn(3, 10, requires_grad=True), mod.weight, mod.bias],
            partitioner=default_partition,
        )
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 6))
        self.assertEqual(get_num_ins_outs(bw_graph), (6, 3))

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_raise_getitems(self):
        def f(x):
            y = torch.split(x, x.size(0) // 2, dim=0)
            a = y[0].sin()
            b = y[1].cos()
            return a + b

        _, bw_graph = get_fw_bw_graph(f, [torch.randn(4, 4, requires_grad=True)])

        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, primals_1, tangents_1):
    split = torch.ops.aten.split.Tensor(primals_1, 2);  primals_1 = None
    getitem_1 = split[1]
    getitem = split[0];  split = None
    sin_1 = torch.ops.aten.sin.default(getitem_1);  getitem_1 = None
    neg = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, neg);  neg = None
    cos_1 = torch.ops.aten.cos.default(getitem);  getitem = None
    mul_1 = torch.ops.aten.mul.Tensor(tangents_1, cos_1);  tangents_1 = cos_1 = None
    cat = torch.ops.aten.cat.default([mul_1, mul]);  mul_1 = mul = None
    return (cat,)""",
        )

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_custom_partitioner_fn(self):
        class MyCustomPartitionerFn(CustomPartitionerFn):
            def __init__(self):
                super().__init__()
                self.called = False

            def __call__(self, gm, joint_inputs, **kwargs):
                self.called = True
                return min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)

            def uuid(self):
                return None

        def f(x):
            return x.cos().cos()

        inp = [torch.randn((4, 4), requires_grad=True)]
        custom_partitioner_fn = MyCustomPartitionerFn()
        fw_graph, bw_graph = get_fw_bw_graph(f, inp, partitioner=custom_partitioner_fn)
        self.assertTrue(custom_partitioner_fn.called)
        self.assertExpectedInline(
            fw_graph.code.strip(),
            """\
def forward(self, primals_1):
    cos = torch.ops.aten.cos.default(primals_1)
    cos_1 = torch.ops.aten.cos.default(cos);  cos = None
    return (cos_1, primals_1)""",
        )
        self.assertExpectedInline(
            bw_graph.code.strip(),
            """\
def forward(self, primals_1, tangents_1):
    cos = torch.ops.aten.cos.default(primals_1)
    sin = torch.ops.aten.sin.default(cos);  cos = None
    neg = torch.ops.aten.neg.default(sin);  sin = None
    mul = torch.ops.aten.mul.Tensor(tangents_1, neg);  tangents_1 = neg = None
    sin_1 = torch.ops.aten.sin.default(primals_1);  primals_1 = None
    neg_1 = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, neg_1);  mul = neg_1 = None
    return (mul_1,)""",
        )

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_save_shape(self):
        def f(x):
            s = x.sum(dim=1)
            return s

        inp = [torch.ones([10, 10], requires_grad=True)]
        fw_graph, bw_graph = get_fw_bw_graph(f, inp, dynamic=True)
        _, fw_output = get_ins_outs(fw_graph)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))
        self.assertEqual(str(fw_output[0]), "sum_1")
        # make sure we don't do the suboptimal thing of saving the bigger primals input to sum,
        # rather than saving the sizes of the primals input for use in backward expand
        self.assertEqual(str(fw_output[1]), "sym_size_int")
        self.assertEqual(str(fw_output[2]), "sym_size_int_1")

        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
        ]

        def f(a, b, c):
            # tried to test what happens if we save a size tuple in the graph;
            # turns out we never will due to how we trace, but this is probably
            # still a good test case for various size manipulations
            sb = torch.ops.aten.sym_size(b)
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            return torch.cat([a.expand(a_sz), b, c])

        fw_graph, bw_graph = get_fw_bw_graph(f, inp, dynamic=True)
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 4))
        self.assertEqual(get_num_ins_outs(bw_graph), (4, 3))
        _, outs = get_ins_outs(fw_graph)
        self.assertTrue(all(is_sym_node(n) for n in outs[1:]))

    def test_default_partitioner_output_tensor_shape_tensor(self):
        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
            torch.randn((10, 1), requires_grad=True),
        ]

        def f(a, b, c, d):
            # Try to force symints intermixed with outputs in the function's returns
            sb = b.size()
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            cat = torch.cat([a.expand(a_sz), b, c])
            mm = torch.mm(cat, d)
            mm2 = torch.mm(
                mm, a.view(mm.size(1), a.size(0))
            )  # this saves 4 new ints for backward. why?
            # and what do i have to do to make it save a tensor for backward?
            return cat, sb, c, mm2

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=default_partition,
            decompositions=default_decompositions,
            dynamic=True,
        )(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]

        # in the fwd graph, 13 outs because:
        # - 5 original outputs (sb is a tuple, gets expanded to 2 symints)
        # - 8 saved outputs for backward: 5 tensors, 3 symints
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 13))
        # in the bwd graph, 10 inputs (grad outs) because:
        # - The fwd graph had 13 outputs
        # - 1 was a view of an input, which gets regenerated outside of the graph
        #   and doesn't participate in the backward
        # - 2 user outs were symints (b.size()), which don't get tangents in the backward
        self.assertEqual(get_num_ins_outs(bw_graph), (10, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            #
            # TODO(whc)- are the saved-tensors/saved-symints correct here?
            # i just made the test pass based on what default partition did
            # Of the 5 original forward outputs, the 4th (c) is an input,
            # which won't show up in the compiled forward graph
            [False, True, True, False, False] + [False] * 4 + [True] * 4,
            [is_sym_node(n) for n in fw_graph_out_nodes],
        )

        real_outs = f(*inp)
        self.assertEqual(compiled_outs, real_outs)
        self.assertTrue(isinstance(real_outs[1], torch.Size))

        # TODO(whc) we should learn to return torch.Sizes
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_output_tensor_shape_tensor(self):
        inp = [
            torch.randn(10, requires_grad=True),
            torch.randn((3, 10), requires_grad=True),
            torch.randn((2, 10), requires_grad=True),
            torch.randn((10, 1), requires_grad=True),
        ]

        def f(a, b, c, d):
            # Try to force symints intermixed with outputs in the function's returns
            sb = b.size()
            sc = c.size()
            x = sb[0] + sc[0]
            a_sz = (x, a.size(0))
            cat = torch.cat([a.expand(a_sz), b, c])
            mm = torch.mm(cat, d)
            mm2 = torch.mm(
                mm, a.view(mm.size(1), a.size(0))
            )  # this saves 4 new ints for backward. why?
            # and what do i have to do to make it save a tensor for backward?
            return cat, sb, c, mm2

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=min_cut_rematerialization_partition,
            decompositions=default_decompositions,
            dynamic=True,
        )(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]

        self.assertEqual(get_num_ins_outs(fw_graph), (4, 12))
        self.assertEqual(get_num_ins_outs(bw_graph), (9, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            # then 4 tensors (transposes of matrices used for mm) are saved
            # finally 3 symints are saved
            [False, True, True, False, False] + [False] * 4 + [True] * 3,
            [is_sym_node(n) for n in fw_graph_out_nodes],
        )

        real_outs = f(*inp)
        self.assertEqual(compiled_outs, real_outs)
        self.assertTrue(isinstance(real_outs[1], torch.Size))

        # TODO(whc) we should learn to return torch.Sizes
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner(self):
        def f(x):
            return x.cos().cos().cos()

        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)])
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 1))

        def f(a, b, c, d):
            x = a + b + c + d
            return x.cos().cos()

        fw_graph, bw_graph = get_fw_bw_graph(
            f, [torch.randn(3, requires_grad=True) for _ in range(4)]
        )
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 4))

    def test_contiguous(self):
        # The test simulates the condition where transpose followed by view
        # happens in the backward pass.
        # https://discuss.pytorch.org/t/error-on-transpose-and-view/434
        def f(x):
            return x.view(2, 3).t()

        inp = torch.randn(6, requires_grad=True)
        out = aot_function(f, nop)(inp)
        torch.autograd.grad(out, inp, torch.randn(3, 2))

    def test_preserve_random(self):
        def fn(x):
            return torch.nn.functional.dropout(x, 0.5) + x

        x = torch.randn(4)

        torch.manual_seed(0)
        ref = fn(x)

        torch.manual_seed(0)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        assert torch.allclose(ref, res)

    # https://github.com/pytorch/pytorch/issues/110666
    def test_generate_gives_inference_graph(self):
        # We expect this to give an inference graph
        def generate(x):
            with torch.no_grad():
                return torch.mul(x, x)

        inference_graph_cell = [None]
        inference_compiler = make_boxed_compiler(
            partial(extract_graph, graph_cell=inference_graph_cell)
        )
        aot_fn = aot_function(generate, nop, inference_compiler=inference_compiler)
        # Even though x requires grad, we should still get an inference graph
        x = torch.randn(4, requires_grad=True)
        aot_fn(x)
        self.assertTrue(inference_graph_cell[0] is not None)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_autocast(self):
        mod = torchvision.models.resnet18().cuda()
        mod.train()

        x = torch.randn(16, 3, 32, 32, device="cuda")
        aot_mod = memory_efficient_fusion(mod)

        # Ensure that AOT Autograd works with AMP
        with torch.cuda.amp.autocast(True):
            res = aot_mod(x)
        res.sum().backward()

    def test_quantize_activation_duplicate_nodes(self):
        """Test both quantize_activation_fw and quantize_activation_bw handle duplicate nodes correctly"""
        import torch.fx as fx
        from torch._functorch.partitioners import (
            quantize_activation_bw,
            quantize_activation_fw,
        )
        from torch._subclasses.fake_tensor import extract_tensor_metadata

        # Mock the inductor config
        with patch.dict(
            "torch._inductor.config.post_grad_fusion_options",
            {
                "activation_quantization_aten_pass": {
                    "allowed_dtypes": "torch.bfloat16",
                    "size_in_mb": 1,
                    "use_scaling": True,
                    "exclude_primals": False,
                    "skip_dynamo_guards": True,
                    "quantize_dynamic_shape": False,
                    "quant_type": "torch.float16",  # float8_e5m2 must be GPU
                }
            },
        ):
            # Test Forward Graph with duplicate nodes
            fwd_graph = fx.Graph()

            # Create input nodes
            x = fwd_graph.placeholder("x")
            x.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            x.meta["tensor_meta"] = extract_tensor_metadata(x.meta["val"])

            y = fwd_graph.placeholder("y")
            y.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            y.meta["tensor_meta"] = extract_tensor_metadata(y.meta["val"])

            # Create a computation node that will be duplicated in outputs
            mul_node = fwd_graph.call_function(torch.ops.aten.mul.Tensor, (x, y))
            mul_node.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            mul_node.meta["tensor_meta"] = extract_tensor_metadata(mul_node.meta["val"])
            mul_node.meta["saved_for_quantization"] = True

            # Create another node
            add_node = fwd_graph.call_function(torch.ops.aten.add.Tensor, (x, y))
            add_node.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            add_node.meta["tensor_meta"] = extract_tensor_metadata(add_node.meta["val"])

            # Create output with DUPLICATE nodes - mul_node appears at positions 0 and 2
            fwd_graph.output((mul_node, add_node, mul_node))

            # Test the forward quantization function
            quantize_activation_fw(fwd_graph)

            # Get the forward output node
            fwd_output_node = fwd_graph.find_nodes(op="output")[0]
            fwd_output_args = fwd_output_node.args[0]

            # Verify forward graph has the correct structure
            self.assertGreaterEqual(
                len(fwd_output_args), 3, "Should have at least the original 3 outputs"
            )

            # Check that positions 0 and 2 reuse the same quantized node
            pos_0_node = fwd_output_args[0]
            pos_2_node = fwd_output_args[2]

            # Both should be quantized nodes
            self.assertTrue(
                pos_0_node.name.startswith("fp8_quant_"),
                f"Position 0 should be quantized node, got: {pos_0_node.name}",
            )
            self.assertTrue(
                pos_2_node.name.startswith("fp8_quant_"),
                f"Position 2 should be quantized node, got: {pos_2_node.name}",
            )

            # The shared quantized node should have the first occurrence position in its name
            self.assertIn(
                "_pos_0",
                pos_0_node.name,
                f"Shared quantized node should have '_pos_0' in name: {pos_0_node.name}",
            )
            self.assertIn(
                "_pos_2",
                pos_2_node.name,
                f"Shared quantized node should have '_pos_2' in name: {pos_2_node.name}",
            )
            # Find scale nodes in the forward output
            fwd_scale_nodes = [
                node for node in fwd_output_args if "fp8_scale_" in node.name
            ]
            self.assertEqual(
                len(fwd_scale_nodes),
                2,
                "Should have exactly 2 scale node (shared for both quantized instances)",
            )

            # Test Backward Graph with duplicate nodes
            bwd_graph = fx.Graph()

            # Create backward placeholders corresponding to forward outputs
            quant_input1 = bwd_graph.placeholder("fp8_quant_pos_0_mul_tensor")
            quant_input1.meta["val"] = torch.randn(100, 100, dtype=torch.float16)
            quant_input1.meta["tensor_meta"] = extract_tensor_metadata(
                quant_input1.meta["val"]
            )
            quant_input1.meta["saved_for_quantization"] = True
            quant_input1.meta["dequant_type"] = torch.bfloat16

            add_input = bwd_graph.placeholder("add")
            add_input.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            add_input.meta["tensor_meta"] = extract_tensor_metadata(
                add_input.meta["val"]
            )

            quant_input2 = bwd_graph.placeholder("fp8_quant_pos_2_mul_tensor")
            quant_input2.meta["val"] = torch.randn(100, 100, dtype=torch.float16)
            quant_input2.meta["tensor_meta"] = extract_tensor_metadata(
                quant_input2.meta["val"]
            )
            quant_input2.meta["saved_for_quantization"] = True
            quant_input2.meta["dequant_type"] = torch.bfloat16

            # Add scale node (would come from forward)
            scale_input = bwd_graph.placeholder("fp8_scale_pos_0_mul_tensor")
            scale_input.meta["val"] = torch.randn(100, 100, dtype=torch.float32)
            scale_input.meta["tensor_meta"] = extract_tensor_metadata(
                scale_input.meta["val"]
            )

            scale_input2 = bwd_graph.placeholder("fp8_scale_pos_2_mul_tensor")
            scale_input2.meta["val"] = torch.randn(100, 100, dtype=torch.float32)
            scale_input2.meta["tensor_meta"] = extract_tensor_metadata(
                scale_input.meta["val"]
            )
            # Create some backward computation using both quantized inputs
            grad_output1 = bwd_graph.placeholder("tangents_1")
            grad_output1.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            grad_output1.meta["tensor_meta"] = extract_tensor_metadata(
                grad_output1.meta["val"]
            )

            grad_output2 = bwd_graph.placeholder("tangents_2")
            grad_output2.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            grad_output2.meta["tensor_meta"] = extract_tensor_metadata(
                grad_output2.meta["val"]
            )

            # Create backward operations using the quantized inputs
            mul_bwd1 = bwd_graph.call_function(
                torch.ops.aten.mul.Tensor, (quant_input1, grad_output1)
            )
            mul_bwd1.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            mul_bwd1.meta["tensor_meta"] = extract_tensor_metadata(mul_bwd1.meta["val"])

            mul_bwd2 = bwd_graph.call_function(
                torch.ops.aten.mul.Tensor, (quant_input2, grad_output2)
            )
            mul_bwd2.meta["val"] = torch.randn(100, 100, dtype=torch.bfloat16)
            mul_bwd2.meta["tensor_meta"] = extract_tensor_metadata(mul_bwd2.meta["val"])

            # Create output
            bwd_graph.output((mul_bwd1, mul_bwd2))

            # Test the backward quantization function
            quantize_activation_bw(bwd_graph)

            # Verify backward graph processing
            bwd_placeholders = list(bwd_graph.find_nodes(op="placeholder"))
            quantized_placeholders = [
                p for p in bwd_placeholders if "fp8_quant_" in p.name
            ]
            scale_placeholders = [p for p in bwd_placeholders if "fp8_scale_" in p.name]

            # Should have processed the quantized placeholders
            self.assertGreater(
                len(quantized_placeholders), 0, "Should have quantized placeholders"
            )
            self.assertGreater(
                len(scale_placeholders), 0, "Should have scale placeholders"
            )

            # Check that dequantization operations were added
            dequant_operations = [
                node
                for node in bwd_graph.nodes
                if node.op == "call_function"
                and "convert_element_type" in str(node.target)
            ]

            # Should have dequantization operations for each quantized input that was processed
            self.assertGreater(
                len(dequant_operations),
                0,
                "Should have dequantization operations in backward graph",
            )

            # Verify the backward graph users were properly updated
            for quant_placeholder in quantized_placeholders:
                # The quantized placeholder should not be directly used in final operations
                # (it should be replaced by dequantized versions)
                direct_users = [
                    user
                    for user in quant_placeholder.users
                    if user.op == "call_function" and "mul" in str(user.target)
                ]
                # Direct usage should be minimal (only for dequantization chain)
                self.assertLessEqual(
                    len(direct_users),
                    1,
                    f"Quantized placeholder {quant_placeholder.name} should have minimal direct users",
                )


class TestAOTDispatch(AOTTestCase):
    # Tests to add cases for (non-exhaustive list, mostly for my notes):
    # - subclass / mode introduced in the middle of the compiled fn
    # - various input mutation / intermediate base tests
    # - input mutation that changes a tensor into a subclass
    # - metadata mutation? (TBD)
    # - guard tests (fw guards *and* bw guards)
    # - subclass test involving _indices_of_inps_to_detach
    def test_aot_dispatch_simple(self):
        # a is a subclass, b is not
        def f(a, b):
            aa = torch.mul(a, 6)
            bb = torch.div(b, 2)
            return aa + bb

        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3, requires_grad=True)

        a1_test = a1_ref.detach().clone().requires_grad_(True)
        a2_test = a2_ref.detach().clone().requires_grad_(True)
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.detach().clone().requires_grad_(True)

        fw_graph_cell = [None]
        bw_graph_cell = [None]

        compiled_f = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)

        # Output is a TwoTensor (check both inner tensors)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        out_ref.sum().backward()
        out_test.sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref.grad.a, a_test.grad.a)
        self.assertEqual(a_ref.grad.b, a_test.grad.b)
        self.assertEqual(b_ref.grad.a, b_test.grad.a)
        self.assertEqual(b_ref.grad.b, b_test.grad.b)

        # Important pieces of the graph:
        # - mul() and div() show up twice, because we called them on a TwoTensor
        # - add() shows up once, because we called it on a plain Tensor
        # - The user forward() fn returns 1 output (the result of add),
        #   while the graph itself returns two outputs (add, add_1)
        # - add, add_1 correspond to the two inner dense tensors that will be wrapped
        # - into a single TwoTensor output.
        self.assertExpectedInline(
            fw_graph_cell[0].code.strip(),
            """\
def forward(self, primals_1, primals_2, primals_3):
    mul = torch.ops.aten.mul.Tensor(primals_1, 6);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 6);  primals_2 = None
    div = torch.ops.aten.div.Tensor(primals_3, 2);  primals_3 = None
    add = torch.ops.aten.add.Tensor(mul, div);  mul = None
    add_1 = torch.ops.aten.add.Tensor(mul_1, div);  mul_1 = div = None
    return (add, add_1)""",
        )

        # Important pieces of the graph:
        # - 4 total dense outputs.
        #   This corresponds to the fact that each user fwd input (a, b)
        #   will get a gradient that is a TwoTensor subclass,
        #   so (mul_2, mul_3) will be wrapped into a.grad
        #   and (div_1, div_2) will be wrapped into b.grad
        # - 4 total dense outputs,
        self.assertExpectedInline(
            bw_graph_cell[0].code.strip(),
            """\
def forward(self, tangents_1, tangents_2):
    div_1 = torch.ops.aten.div.Tensor(tangents_1, 2)
    div_2 = torch.ops.aten.div.Tensor(tangents_2, 2)
    mul_2 = torch.ops.aten.mul.Tensor(tangents_1, 6);  tangents_1 = None
    mul_3 = torch.ops.aten.mul.Tensor(tangents_2, 6);  tangents_2 = None
    return (mul_2, mul_3, div_1, div_2)""",
        )

    def test_aot_dispatch_inference(self):
        # a is a subclass, b is not
        def f(a, b):
            aa = torch.mul(a, 6)
            bb = torch.div(b, 2)
            return aa + bb

        a1_ref = torch.ones(3, 3)
        a2_ref = torch.ones(3, 3)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3)

        a1_test = a1_ref.clone()
        a2_test = a2_ref.clone()
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.clone()

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)

        # Output is a TwoTensor (check both inner tensors)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

    def test_aot_dispatch_incorrect_backward(self):
        # a is a subclass, b is not
        def f(a, b):
            aa = torch.mul(a, 2)
            bb = torch.add(b, 3)
            out_subclass = torch.div(aa, bb)
            out_reg = torch.add(b, b)
            # When creating the joint, we assume that the second grad_out
            # is not a subclass.
            # In the below test case though, we end up being wrong.
            # This would require re-tracing and recompiling the backward.
            return out_subclass, out_reg

        a1_ref = torch.ones(3, 3, requires_grad=True)
        a2_ref = torch.ones(3, 3, requires_grad=True)
        a_ref = TwoTensor(a1_ref, a2_ref)
        b_ref = torch.ones(3, 3, requires_grad=True)

        a1_test = a1_ref.detach().clone().requires_grad_(True)
        a2_test = a2_ref.detach().clone().requires_grad_(True)
        a_test = TwoTensor(a1_test, a2_test)
        b_test = b_ref.detach().clone().requires_grad_(True)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        # First out is a TwoTensor, second is an ordinary tensor
        self.assertEqual(out_ref[0].a, out_test[0].a)
        self.assertEqual(out_ref[0].b, out_test[0].b)
        self.assertEqual(out_ref[1], out_test[1])

        # We compiled our graph assuming type(grad_out[1]) is torch.Tensor,
        # but we were wrong: in the below tests, it is a subclass.
        # This will eventually require a repartition + recompile
        with self.assertRaisesRegex(
            RuntimeError,
            """
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
""",  # noqa: F541
        ):
            (out_test[0] + out_test[1]).sum().backward()

    def test_aot_dispatch_output_alias(self):
        # a is a tensor, b is a TwoTensor
        def f(a, b):
            return b.view(b.shape), a * b

        b1_ref = torch.ones(3, 3, requires_grad=True)
        b2_ref = torch.ones(3, 3, requires_grad=True)
        b_ref = TwoTensor(b1_ref, b2_ref)
        a_ref = torch.ones(3, 3, requires_grad=True)

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test = TwoTensor(b1_test, b2_test)
        a_test = a_ref.detach().clone().requires_grad_(True)

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref1, out_ref2 = f(a_ref, b_ref)
        out_test1, out_test2 = compiled_f(a_test, b_test)
        self.assertEqual(out_ref1, out_test1)
        self.assertEqual(out_ref2.a, out_test2.a)
        self.assertEqual(out_ref2.b, out_test2.b)

        (out_ref1 + out_ref2).sum().backward()
        (out_test1 + out_test2).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref.grad.a, a_test.grad.a)
        self.assertEqual(a_ref.grad.b, a_test.grad.b)
        self.assertEqual(b_ref.grad.a, b_test.grad.a)
        self.assertEqual(b_ref.grad.b, b_test.grad.b)

    @torch._functorch.config.patch(
        {
            "disable_guess_zero_tangent_for_mutated_input_subclass": True,
        }
    )
    def test_aot_dispatch_input_mutation(self):
        def f(a, b):
            a.mul_(2)
            b.mul_(3)
            return a + b

        b1_ref = torch.ones(3, 3, requires_grad=True)
        b2_ref = torch.ones(3, 3, requires_grad=True)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = torch.ones(3, 3, requires_grad=True)
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        # NOTE: we need to use b in our gradient compute. Otherwise we will need to recompile the backward.
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    # NB: Metadata mutation for subclasses is currently broken and disabled
    # See https://github.com/pytorch/pytorch/issues/114975
    @unittest.expectedFailure
    def test_aot_dispatch_input_metadata_mutation(self):
        def f(a, b):
            a.t_()
            b.unsqueeze_(0)
            return a + b

        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()
            .requires_grad_(True)
        )
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        # NOTE: we need to use b in our gradient compute. Otherwise we will need to recompile the backward.
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    # NB: Metadata mutation for subclasses is currently broken and disabled
    # See https://github.com/pytorch/pytorch/issues/114975
    @unittest.expectedFailure
    def test_aot_dispatch_input_data_and_metadata_mutation(self):
        def f(a, b):
            a.t_()
            b.unsqueeze_(0)
            a.mul_(2)
            b.mul_(3)
            return a + b

        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()
            .requires_grad_(True)
        )
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref = f(a_ref, b_ref)
        out_test = compiled_f(a_test, b_test)
        self.assertEqual(out_ref.a, out_test.a)
        self.assertEqual(out_ref.b, out_test.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        # NOTE: we need to use b in our gradient compute. Otherwise we will need to recompile the backward.
        (b_ref * out_ref).sum().backward()
        (b_test * out_test).sum().backward()
        # Both grad_inputs are TwoTensor
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)
        self.assertEqual(b_ref_base.grad.a, b_test_base.grad.a)
        self.assertEqual(b_ref_base.grad.b, b_test_base.grad.b)

    @torch._functorch.config.patch(
        {
            "disable_guess_zero_tangent_for_mutated_input_subclass": True,
        }
    )
    def test_aot_dispatch_input_mutation_and_output_alias(self):
        def f(a, b):
            a.mul_(2)
            b.mul_(3)
            return b.view(b.shape), a + b

        b1_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b2_ref = torch.arange(9, requires_grad=True, dtype=torch.float32).reshape(3, 3)
        b_ref_base = TwoTensor(b1_ref, b2_ref)
        a_ref_base = (
            torch.arange(9, dtype=torch.float32)
            .reshape(3, 3)
            .detach()
            .requires_grad_(True)
        )
        b_ref = b_ref_base + 1
        a_ref = a_ref_base + 1

        b1_test = b1_ref.detach().clone().requires_grad_(True)
        b2_test = b2_ref.detach().clone().requires_grad_(True)
        b_test_base = TwoTensor(b1_test, b2_test)
        a_test_base = a_ref_base.detach().clone().requires_grad_(True)
        b_test = b_test_base + 1
        a_test = a_test_base + 1

        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=min_cut_rematerialization_partition,
        )
        out_ref1, out_ref2 = f(a_ref, b_ref)
        out_test1, out_test2 = compiled_f(a_test, b_test)
        self.assertEqual(out_ref1.a, out_test1.a)
        self.assertEqual(out_ref1.b, out_test1.b)
        self.assertEqual(out_ref2.a, out_test2.a)
        self.assertEqual(out_ref2.b, out_test2.b)

        # confirm input mutations worked
        self.assertEqual(a_test, a_ref)
        self.assertEqual(b_test.a, b_ref.a)
        self.assertEqual(b_test.b, b_ref.b)

        (out_ref1 * out_ref2).sum().backward()
        (out_test1 * out_test2).sum().backward()
        # Both grad_inputs are TwoTensors
        self.assertEqual(a_ref_base.grad.a, a_test_base.grad.a)
        self.assertEqual(a_ref_base.grad.b, a_test_base.grad.b)

    def test_aot_dispatch_output_requires_grad_in_no_grad(self):
        def fn(x):
            out1 = x.sin()
            with torch.enable_grad():
                out2 = x.cos()
            return out1, out2

        inp_fns = [
            lambda: torch.ones(10, requires_grad=True),
            lambda: torch.ones(10, requires_grad=False),
        ]

        compiled_f = aot_function(fn, nop)
        for inp_fn in inp_fns:
            with torch.no_grad():
                ref_x = inp_fn()
                ref_out = fn(ref_x)
                x = inp_fn()
                out = compiled_f(x)
                for r, o in zip(ref_out, out):
                    self.assertEqual(r.requires_grad, o.requires_grad)
            if ref_x.requires_grad:
                with torch.enable_grad():
                    (ref_out[0] + ref_out[1]).sum().backward()
                    (out[0] + out[1]).sum().backward()
                    self.assertEqual(ref_x.grad, x.grad)
                    assert torch.allclose(ref_x.grad, x.grad, atol=1e-3, rtol=1e-3)

    def test_aot_dispatch_output_requires_grad_in_no_grad_views(self):
        # view-type ops preserve requires_grad even in no_grad.
        def fn(x):
            return x.view(-1), x.sin()

        inference_graph_cell = [None]
        inference_compiler = make_boxed_compiler(
            partial(extract_graph, graph_cell=inference_graph_cell)
        )
        compiled_fn = aot_function(fn, nop, inference_compiler=inference_compiler)

        inp_x0 = torch.ones(2, 3, requires_grad=True)
        # Clone in no_grad will make requires_grad=False tensors, keep clone outside of no_grad
        ref_x0 = inp_x0.clone()
        x0 = inp_x0.clone()
        with torch.no_grad():
            ref_out1, ref_out2 = fn(ref_x0)

            out1, out2 = compiled_fn(x0)
            # Assert that we executed inference graph
            self.assertTrue(inference_graph_cell[0] is not None)

            self.assertEqual(ref_out1.requires_grad, out1.requires_grad)
            self.assertEqual(ref_out2.requires_grad, out2.requires_grad)


class GradsNoForceContiguousContextManager(ContextDecorator):
    def __enter__(self):
        # flake8: noqa: TOR901
        self.lib = torch.library.Library("_test_aotdispatch_lib", "FRAGMENT")
        self.d = {
            torch.channels_last: 0,
            torch.contiguous_format: 0,
        }
        self.tangent_strides = []

        self.lib.define("log_tangents_memory_format(Tensor x) -> Tensor")
        self.lib.define("log_tangents_memory_format_log(Tensor x) -> Tensor")

        def log_tangents_memory_format_impl(a):
            return a.clone()

        def log_tangents_memory_format_meta(a):
            return a.clone()

        def log_tangents_memory_format_log_impl(x):
            self.d[torch._prims_common.suggest_memory_format(x)] += 1
            self.tangent_strides.append(x.stride())
            return x.clone()

        def log_tangents_memory_format_log_meta(a):
            return a.clone()

        for backend in ["CPU", "CUDA"]:
            self.lib.impl(
                "log_tangents_memory_format", log_tangents_memory_format_impl, backend
            )
            self.lib.impl(
                "log_tangents_memory_format_log",
                log_tangents_memory_format_log_impl,
                backend,
            )

        self.lib.impl(
            "log_tangents_memory_format", log_tangents_memory_format_meta, "Meta"
        )
        self.lib.impl(
            "log_tangents_memory_format_log",
            log_tangents_memory_format_log_meta,
            "Meta",
        )

        def log_tangents_memory_format_bwd(ctx, grad):
            torch.ops._test_aotdispatch_lib.log_tangents_memory_format_log(grad)
            return grad.clone()

        torch.library.register_autograd(
            "_test_aotdispatch_lib::log_tangents_memory_format",
            log_tangents_memory_format_bwd,
            lib=self.lib,
        )

        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op

        _register_effectful_op(
            torch.ops._test_aotdispatch_lib.log_tangents_memory_format.default,
            _EffectType.ORDERED,
        )
        _register_effectful_op(
            torch.ops._test_aotdispatch_lib.log_tangents_memory_format_log.default,
            _EffectType.ORDERED,
        )

        return self

    def __exit__(self, type, value, tb):
        self.lib._destroy()
        return False

    def reset_counters(self):
        self.d = {
            torch.channels_last: 0,
            torch.contiguous_format: 0,
        }


class TestAOTModuleSimplified(AOTTestCase):
    def test_aot_module_simplified(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                return (self.linear(x) + y,)

        mod = MockModule()
        mod.zero_grad()

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]
        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]

        ref = mod(*inputs)
        ref[0].sum().backward()

        compiled_f = aot_module_simplified(mod, cloned_inputs, nop)
        mod.zero_grad()
        res = compiled_f(*cloned_inputs)
        res[0].sum().backward()

        assert torch.allclose(ref[0], res[0])
        assert torch.allclose(inputs[0].grad, cloned_inputs[0].grad)
        assert torch.allclose(inputs[1].grad, cloned_inputs[1].grad)

    def test_aot_module_simplified_dynamic(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                return (self.linear(x) + y,)

        mod = MockModule()

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)

        inputs = [x, y]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)

        ref = mod(*inputs)
        ref[0].sum().backward()

        cloned_inputs = [x.detach().clone().requires_grad_(True) for x in inputs]
        res = compiled_f(*cloned_inputs)
        res[0].sum().backward()

        self.assertExpectedInline(
            shape_env.format_guards(),
            """\
 - Eq(s49, 20)
 - Eq(s70, 30)""",
        )

        assert torch.allclose(ref[0], res[0])
        assert torch.allclose(inputs[0].grad, cloned_inputs[0].grad)
        assert torch.allclose(inputs[1].grad, cloned_inputs[1].grad)

    # https://github.com/pytorch/pytorch/issues/105327
    def test_lift_fresh_copy_in_graph(self):
        class MyMod(torch.nn.Module):
            def forward(self, x):
                _tensor_constant0 = torch.tensor([1])
                lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(
                    _tensor_constant0
                )
                y = x.mul(lift_fresh_copy)
                return (y,)

        mod = MyMod()
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = torch.ones(4, requires_grad=True)
        inputs = [x]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        compiled_f = aot_module_simplified(mod, fake_inputs, nop)

        out_ref = mod(x)
        out_test = compiled_f(x)
        self.assertEqual(out_ref[0].detach(), out_test[0].detach())

    def test_inference_python_dispatcher(self):
        # Extracted from unet
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.upsample = torch.nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=True
                )

            def forward(self, x):
                return (self.upsample(x),)

        mod = MockModule()
        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        x = torch.randn(2, 512, 40, 59)  # NB: must not require grad
        inputs = [x]
        fake_inputs = [fake_mode.from_tensor(x) for x in inputs]
        aot_module_simplified(mod, fake_inputs, nop)

    def test_aot_module_simplified_preserves_stack_trace(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                z = self.linear(x)
                z = z + y
                z = z.relu()
                return (z,)

        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(MockModule())
        mod = torch.fx.GraphModule(tracer.root, graph)

        for node in mod.graph.nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(node.stack_trace is not None)
            assert "test_aotdispatch.py" in node.stack_trace

        def assert_compiler(gm: torch.fx.GraphModule, _):
            for node in gm.graph.nodes:
                if node.op == "output" or node.op == "placeholder":
                    continue
                self.assertTrue(node.stack_trace is not None)
                assert "test_aotdispatch.py" in node.stack_trace
            return gm.forward  # return a python callable

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]

        compiled_f = aot_module_simplified(
            mod, inputs, fw_compiler=assert_compiler, bw_compiler=assert_compiler
        )
        res = compiled_f(*inputs)
        res[0].sum().backward()

    def test_aot_module_simplified_preserves_stack_trace_from_mutation(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                x_view = x[0]
                x_view.mul_(2)
                return (x + x,)

        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(MockModule())
        mod = torch.fx.GraphModule(tracer.root, graph)

        for node in mod.graph.nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(node.stack_trace is not None)
            assert "test_aotdispatch.py" in node.stack_trace

        def assert_compiler(gm: torch.fx.GraphModule, _):
            assert torch.ops.aten.copy_.default in [x.target for x in gm.graph.nodes]
            for node in gm.graph.nodes:
                if node.target == torch.ops.aten.copy_.default:
                    assert "stack_trace" in node.meta
                    assert "x_view.mul_(2)" in node.meta["stack_trace"]
            return gm.forward  # return a python callable

        x = torch.randn(128, 20)
        inputs = [x]

        aot_module_simplified(
            mod,
            inputs,
            fw_compiler=assert_compiler,
            bw_compiler=assert_compiler,
            keep_inference_input_mutations=True,
        )

    def test_aot_module_simplified_fake_tensor_gm_raises(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        real_x = torch.randn(4, requires_grad=True)
        fake_x = fake_mode.from_tensor(real_x)
        real_z = torch.randn(4)
        fake_z = fake_mode.from_tensor(real_z)

        class MockModule(torch.nn.Module):
            def forward(self, x):
                # Accessing a free variable fake tensor will look like a
                # constant to make_fx, and result in the tensor being traced
                # into the graph, which is an error condition.  Make sure we
                # report adequately in this case.
                return (x + fake_z,)

        with self.assertRaisesRegex(AssertionError, "Unexpected fake"):
            aot_module_simplified(MockModule(), (fake_x,), nop)

    def test_aot_test_subclasses_with_tensor_factories(self):
        from torch.testing._internal.common_subclass import SubclassWithTensorFactory

        inp = SubclassWithTensorFactory(torch.zeros(3, 5))

        def fn(x):
            return 2 * x

        ref_out = fn(inp)
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(inp)
        self.assertEqual(ref_out, out)

    # Next several tests are related to issue:
    # https://github.com/pytorch/pytorch/issues/134644
    # AOTD tries to predict tangents for tracing ahead of time.
    # The first strategy was to coerce traced_tangents and runtime_tangents to be contiguous().
    # But for models working in channels_last memory format this will add additional contiguous() calls.
    # The fix is predicting tangents memory format to be similar to outputs memory format.
    # And coerce runtime tangents to that traced memory format.
    def test_grads_no_force_contiguous_dense(self):
        with GradsNoForceContiguousContextManager() as ctx:

            class M(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 3, 3)

                def forward(self, x, y, cont_inp):
                    z = y + 3
                    y.mul_(2)
                    r = self.conv(x)
                    r = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(r)
                    return (
                        r,
                        r.transpose(0, 1),
                        z.view(-1),
                        z.transpose(0, 1),
                        cont_inp * 2,
                    )

            m = M()
            m.to(memory_format=torch.channels_last)
            m.train()

            def dense_inps():
                return (
                    torch.randn(2, 3, 5, 5, requires_grad=True).to(
                        memory_format=torch.channels_last
                    ),
                    torch.randn(3, 2, 1, 1, requires_grad=True).to(
                        memory_format=torch.channels_last
                    ),
                    torch.randn(3, 2, 1, 1, requires_grad=True),
                )

            ref_inps = dense_inps()
            ref_outs = m(*ref_inps)
            ref_outs[0].sum().backward()

            ctx.reset_counters()
            inps = dense_inps()
            outs = torch.compile(m, backend="inductor", fullgraph=True)(*inps)
            outs[0].sum().backward()

            self.assertEqual(ctx.d[torch.channels_last], 1)
            self.assertEqual(ctx.d[torch.contiguous_format], 0)

    def test_grads_no_force_contiguous_subclass(self):
        with GradsNoForceContiguousContextManager() as ctx:

            class M(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 3, 3)

                def forward(self, x, y):
                    r = self.conv(x)
                    r = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(r)
                    return r, y + 1

            m = M()
            m.to(memory_format=torch.channels_last)
            m.train()

            def inps_fn():
                return (
                    TwoTensor(
                        torch.randn(2, 3, 5, 5, requires_grad=True).to(
                            memory_format=torch.channels_last
                        ),
                        torch.randn(2, 3, 5, 5, requires_grad=True).to(
                            memory_format=torch.channels_last
                        ),
                    ),
                    torch.randn(3, 2, requires_grad=True).clone(),
                )

            ref_outs = m(*inps_fn())
            ref_outs[0].sum().backward()

            ctx.reset_counters()
            mc = M()
            mc.to(memory_format=torch.channels_last)
            mc.train()
            outs = torch.compile(mc, backend="aot_eager", fullgraph=True)(*inps_fn())
            outs[0].sum().backward()

            self.assertEqual(ctx.d[torch.channels_last], 2)
            self.assertEqual(ctx.d[torch.contiguous_format], 0)

    def test_grads_no_force_contiguous_nested_subclass(self):
        with GradsNoForceContiguousContextManager() as ctx:

            class M(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, 3, 3)

                def forward(self, x):
                    r = self.conv(x)
                    r = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(r)
                    return r

            m = M()
            m.to(memory_format=torch.channels_last)
            m.train()

            def inps_fn(x):
                return (
                    TwoTensor(
                        TwoTensor(x.clone(), x.clone()), TwoTensor(x.clone(), x.clone())
                    ),
                )

            x = torch.randn(2, 3, 5, 5, requires_grad=True).to(
                memory_format=torch.channels_last
            )
            ref_inps = inps_fn(x)
            ref_outs = m(*ref_inps)
            ref_outs[0].sum().backward()

            ctx.reset_counters()

            mc = M()
            mc.to(memory_format=torch.channels_last)
            mc.train()

            x = torch.randn(2, 3, 5, 5, requires_grad=True).to(
                memory_format=torch.channels_last
            )
            inps = inps_fn(x)
            outs = torch.compile(mc, backend="aot_eager", fullgraph=True)(*inps)
            outs[0].sum().backward()
            self.assertEqual(ctx.d[torch.channels_last], 4)
            self.assertEqual(ctx.d[torch.contiguous_format], 0)

    def test_grads_no_force_contiguous_nested_tensor_tangent(self):
        # NestedTensor setattr could fails with AttributeError for attr "_min_seqlen_tensor"
        # Adding test to verify that it is handled.
        def fn(x):
            return x.clone()

        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)

        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(nt)
        out_buffer = out.values()
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

    def test_wrong_guess_tangent_type(self):
        def fn(x):
            return x.clone()

        ref_x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        ref_y = fn(ref_x)
        ref_y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        fn_comp = torch.compile(fn, fullgraph=True)

        x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y = fn_comp(x)
        y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        x2 = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y2 = fn_comp(x2)
        with self.assertRaisesRegex(
            RuntimeError,
            """
During the backward, we encountered a tensor subclass where we guessed its
metadata incorrectly.
""",  # noqa: F541
        ):
            y2.backward(gradient=torch.randn(2, 3))

    def test_tangent_type_coercion(self):
        def fn(x):
            return x.clone()

        ref_y = fn(WrapperSubclass(torch.randn(2, 3, requires_grad=True)))
        ref_y.sum().backward()

        fn_comp = torch.compile(fn, fullgraph=True)

        x = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y = fn_comp(x)
        y.backward(gradient=TwoTensor(torch.randn(2, 3), torch.randn(2, 3)))

        x2 = TwoTensor(
            torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        )
        y2 = fn_comp(x2)
        # Test coercion WrapperSubclass -> TwoTensor
        y2.backward(gradient=WrapperSubclass(torch.randn(2, 3)))

        y3 = torch.compile(fn, fullgraph=True)(torch.randn(2, 3, requires_grad=True))
        # Test coercion WrapperSubclass -> Tensor
        y3.backward(gradient=WrapperSubclass(torch.randn(2, 3)))

    @torch._inductor.config.patch({"freezing": True})
    def test_inductor_freezing_with_subclasses(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = TwoTensor(torch.randn(3, 4), torch.randn(3, 4))
                self.wt = torch.randn(3, 4)

            def forward(self, x):
                return (
                    x.index_select(
                        dim=0, index=torch.tensor([0, 2, 1], dtype=torch.int64)
                    )
                    + self.w
                    + self.wt
                )

        m = M()
        inp = torch.randn(3, 4)
        with torch.no_grad():
            torch.compile(m, fullgraph=True)(inp)

    def test_rrelu(self):
        def fn(x):
            return torch.rrelu(x, training=True)

        def fn_(x):
            torch.rrelu_(x, training=True)
            return x

        x = torch.randn(4, 4)
        torch.compile(fn, backend="inductor", fullgraph=True)(x)
        torch.compile(fn_, backend="inductor", fullgraph=True)(x)

    def test_layer_norm(self):
        def fn(x):
            return F.layer_norm(x, normalized_shape=(8,))

        x = torch.randn(2, 4, 8)
        eager = fn(x)
        aot_eager = torch.compile(backend="aot_eager")(fn)(x)
        self.assertEqual(eager, aot_eager, atol=0, rtol=0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_rms_norm(self):
        # Only CUDA rms norm fails to be decomposed
        def fn(x):
            return F.rms_norm(x, normalized_shape=(8,))

        x = torch.randn(2, 4, 8, device="cuda")
        eager = fn(x)
        aot_eager = torch.compile(backend="aot_eager")(fn)(x)
        self.assertEqual(eager, aot_eager, atol=0, rtol=0)

    def test_subclass_parameters(self):
        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(torch.zeros(3, 4), torch.randn(3, 4)),
                        torch.ones(3, 4),
                    )
                )

            def forward(self, x):
                return x + self.p

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        torch.ones(3, 4),
                        TwoTensor(torch.randn(3, 4), torch.randn(3, 4)),
                    )
                )
                self._m = _M()

            def forward(self, x):
                return self._m(x) + x + 2 * self.p1 + self.p2

        m = M()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ref_out.sum().backward()
        m.zero_grad()

        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        unwrap_tensor_subclass_parameters(m)

        ref_x2 = ref_x.detach().clone()
        ref_out2 = m(ref_x2)
        self.assertEqual(ref_out2, ref_out)
        ref_out2.sum().backward()
        self.assertEqual(ref_x2.grad, ref_x.grad)
        m.zero_grad()

        x = ref_x.detach().clone()
        comp_fn = torch.compile(m, backend="aot_eager", fullgraph=True)
        out = comp_fn(x)
        self.assertEqual(ref_out, out)
        out.sum().backward()
        self.assertEqual(ref_x.grad, x.grad)

    def test_subclass_parameters_torture_case(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(
                            torch.ones(3, 4),
                            TwoTensor(torch.randn(3, 4), torch.randn(3, 4)),
                        ),
                        TwoTensor(
                            TwoTensor(torch.randn(3, 4), torch.randn(3, 4)),
                            TwoTensor(torch.ones(3, 4), torch.randn(3, 4)),
                        ),
                    )
                )

            def forward(self, x):
                return x + 2 * self.p1 + self.p2.a.b

        m = M()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ref_out.sum().backward()
        m.zero_grad()

        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        unwrap_tensor_subclass_parameters(m)

        ref_x2 = ref_x.detach().clone()
        ref_out2 = m(ref_x2)
        self.assertEqual(ref_out2, ref_out)
        ref_out2.sum().backward()
        self.assertEqual(ref_x2.grad, ref_x.grad)
        m.zero_grad()

        x = ref_x.detach().clone()
        comp_fn = torch.compile(m, backend="aot_eager", fullgraph=True)
        out = comp_fn(x)
        self.assertEqual(ref_out, out)
        out.sum().backward()
        self.assertEqual(ref_x.grad, x.grad)

    def test_rrelu_with_noise_mutation(self):
        def fn_functional(x):
            noise = torch.ones_like(x)
            result, noise_out = torch.ops.aten.rrelu_with_noise_functional(
                x, noise, 0.2, 0.8, True
            )
            return result, noise_out

        def fn_mutation(x):
            noise = torch.ones_like(x)
            result = torch.ops.aten.rrelu_with_noise(x, noise, 0.2, 0.8, True)
            return result, noise

        def fn_inplace(x):
            noise = torch.ones_like(x, requires_grad=False)
            torch.ops.aten.rrelu_with_noise_(x, noise, 0.2, 0.8, True)
            return x, noise

        def _test_fn(fn, check_backward=True):
            x = -torch.abs(torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True))

            ref_y, ref_noise = fn(x)
            self.assertTrue(torch.all(ref_noise < torch.ones_like(ref_noise)).item())

            comp_y, comp_noise = torch.compile(fn, backend="inductor", fullgraph=True)(
                x
            )

            if check_backward:
                comp_y.sum().backward()
            self.assertTrue(torch.all(comp_noise < torch.ones_like(comp_noise)).item())

        _test_fn(fn_functional)
        _test_fn(fn_mutation)
        _test_fn(fn_inplace, check_backward=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @parametrize("dynamic_shapes", [True, False])
    @parametrize("test_subclasses", [True, False])
    @parametrize("device", ["cuda", "cpu"])
    @patch("torch._functorch.config.guess_tangent_strides_as_outputs", True)
    def test_noncontig_nonmemformat_tangents(
        self, dynamic_shapes, test_subclasses, device
    ):
        B = 2
        T = 4
        E = 6

        def fn(x):
            x = x + 1
            return x.transpose(1, 2)

        def _inp_dense():
            t = torch.randn(B, T, E, device=device, requires_grad=True)
            if dynamic_shapes:
                for i in range(t.ndim):
                    torch._dynamo.mark_dynamic(t, i)
            return t

        def _inp_sc():
            return TwoTensor(_inp_dense(), _inp_dense())

        _inp = _inp_dense if not test_subclasses else _inp_sc

        comp_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        def _tg3(y):
            t = torch.randn(
                2 * y.shape, dtype=y.dtype, layout=y.layout, device=y.device
            )
            return t.as_strided(y.shape, tuple(s * 2 for s in y.stride()))

        TEST_CASES = [
            (_inp, lambda y: torch.ones(y.shape, dtype=y.dtype, device=y.device)),
            # Memory overlap, dense tangent
            (
                _inp,
                lambda y: torch.tensor([1], dtype=y.dtype, device=y.device).as_strided(
                    y.shape, (0,) * y.ndim
                ),
            ),
            # No memory overlap, not-dense tangent
            (_inp, _tg3),
        ]

        for inp_fn, tg_fn in TEST_CASES:
            ref_x = inp_fn()
            x = ref_x.detach().clone().requires_grad_()

            ref_y = fn(ref_x)

            y = comp_fn(x)
            self.assertEqual(ref_y, y)

            ref_tg = (
                tg_fn(ref_y)
                if not test_subclasses
                else TwoTensor(tg_fn(ref_y), tg_fn(ref_y))
            )
            tg = ref_tg.clone()

            ref_y.backward(ref_tg)
            y.backward(tg)

            self.assertEqual(ref_x.grad, x.grad)

    @patch("torch._functorch.config.guess_tangent_strides_as_outputs", True)
    def test_flex_attn_noncontiguous_tangents(self):
        with GradsNoForceContiguousContextManager() as ctx:
            E = 16  # embedding dim
            H = 4  # number of heads

            @torch.compile(backend="aot_eager", fullgraph=True)
            def attn_fn(q, k, v):
                y = flex_attention(query=q, key=k, value=v)
                y = torch.ops._test_aotdispatch_lib.log_tangents_memory_format(y)
                return y

            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.c_attn = torch.nn.Linear(E, 3 * E)

                def forward(self, x):
                    B, T, E = x.size()
                    q, k, v = self.c_attn(x).split(E, dim=2)
                    k = k.view(B, T, H, E // H).transpose(1, 2)  # (B, nh, T, hs)
                    q = q.view(B, T, H, E // H).transpose(1, 2)  # (B, nh, T, hs)
                    v = v.view(B, T, H, E // H).transpose(1, 2)  # (B, nh, T, hs)

                    y = attn_fn(q, k, v)

                    return y.transpose(1, 2).contiguous().view(B, T, E)

            m = M()
            B = 1
            T = 8

            def _inp():
                return torch.randn(B, T, E, requires_grad=True)

            x = _inp()
            y = m(x)
            y.backward(torch.ones_like(y).contiguous())

            self.assertEqual(1, len(ctx.tangent_strides))
            self.assertEqual((128, 4, 16, 1), ctx.tangent_strides[0])

    def _test_pack_hooks(
        self,
        fn,
        inp_fn,
        hooks,
        symbolic_tracing=True,
        pre_compile_fn=None,
        backend="inductor",
    ):
        ctx = torch.autograd.graph.saved_tensors_hooks
        torch._dynamo.reset()
        with ExitStack() as stack:
            # All hooks in eager to get ref
            for hook, _ in hooks:
                pack, unpack = hook
                stack.enter_context(ctx(pack, unpack))
            ref_x = inp_fn()

            def _f(t):
                if t.dtype.is_floating_point:
                    return t.detach().clone().requires_grad_()

                return t

            x = pytree.tree_map_only(torch.Tensor, _f, ref_x)

            ref_y = fn(*ref_x)
            ref_y.sum().backward()
        if pre_compile_fn:
            pre_compile_fn()

        with ExitStack() as stack:
            for hook, inline in hooks:
                pack, unpack = hook
                if inline:
                    if symbolic_tracing:
                        stack.enter_context(
                            ctx(
                                *saved_tensors_hooks_to_gm(
                                    pack,
                                    unpack,
                                    "pack_hash",
                                    "unpack_hash",
                                )
                            )
                        )
                    else:
                        stack.enter_context(
                            ctx(
                                *saved_tensors_hooks_to_gm(
                                    pack, unpack, "pack_hash", "unpack_hash"
                                )
                            )
                        )
                else:
                    stack.enter_context(ctx(pack, unpack))
            y = torch.compile(fn, backend=backend, fullgraph=True)(*x)
            y.sum().backward()
            self.assertEqual(ref_y, y, atol=1e-2, rtol=1e-2)
            ref_x_grad = pytree.tree_map_only(torch.Tensor, lambda t: t.grad, ref_x)
            x_grad = pytree.tree_map_only(torch.Tensor, lambda t: t.grad, x)
            self.assertEqual(ref_x_grad, x_grad, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @parametrize("saved_tensors_hooks_filtering_mode", ["donated", "no_static", "all"])
    def test_saved_tensors_hooks_base(self, saved_tensors_hooks_filtering_mode):
        with patch(
            "torch._functorch.config.saved_tensors_hooks_filtering_mode",
            saved_tensors_hooks_filtering_mode,
        ):
            # y argument is expected to test saving of int tensor,
            # to check filtering functionality to not apply hooks for e.g. is_floating_point
            class SAF(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    ctx.save_for_backward(x, y)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    (saved_x, saved_y) = ctx.saved_tensors
                    return gx + saved_x + saved_y, None

            class AF(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    ctx.d1 = x.size(1)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    (saved_x,) = ctx.saved_tensors
                    d1 = ctx.d1
                    return gx + saved_x * d1

            def fn(x, y):
                x = x.relu()
                x = x + 1
                x = x.relu()
                x = 2 * x
                x = AF.apply(x)
                return x

            def simple_fn(x, y):
                x = x + 1
                x = x.t()
                x = x.relu()
                x = x.t()
                x = SAF.apply(x, y)
                return x

            device = torch.device("cuda:0")

            def inp_fn():
                x = torch.ones(2, 2, device=device, requires_grad=True)
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                y = torch.zeros(2, 2, device=device, dtype=torch.int64)
                return x, y

            def pack_dev_sym_cpu(x):
                return x.dtype, x.device, x.size(1), x.cpu()

            def unpack_dev_sym_cpu(packed):
                dtype, device, dim1, x = packed
                x = x.to(device=device)
                return x.to(dtype)

            def pack_tensor(x):
                return x.device, x.cpu()

            def unpack_tensor(packed):
                device, t_cpu = packed
                return t_cpu.to(device)

            def pack_bf16(x):
                return x.dtype, x.to(dtype=torch.bfloat16)

            def unpack_bf16(packed):
                dtype, x = packed
                return x.to(dtype)

            def pack_mul2(x):
                return x.dtype, x * 2

            def unpack_mul2(x):
                dtype, x = x
                x = x / 2
                return x.to(dtype)

            def pack_wrapper_sc(x):
                return WrapperSubclass(x)

            def unpack_wrapper_sc(x):
                return x.a

            def pack_wrapper_two_tensor(x):
                return TwoTensor(x, x)

            def unpack_wrapper_two_tensor(x):
                return x.a + x.b

            def pack_mul2_eager(x):
                return x * 2

            def unpack_mul2_eager(x):
                return x / 2

            def pack_cpu(x):
                return x.to(device="cpu")

            def unpack_cpu(x):
                return x.to(device=device)

            for test_fn in [simple_fn, fn]:
                self._test_pack_hooks(
                    test_fn,
                    inp_fn,
                    [((pack_cpu, unpack_cpu), True)],
                    symbolic_tracing=False,
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_bf16, unpack_bf16), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_mul2, unpack_mul2), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_tensor, unpack_tensor), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_dev_sym_cpu, unpack_dev_sym_cpu), True)]
                )
                self._test_pack_hooks(
                    test_fn, inp_fn, [((pack_mul2_eager, unpack_mul2_eager), False)]
                )
                self._test_pack_hooks(
                    test_fn,
                    inp_fn,
                    [((pack_fp8, unpack_fp8), True)],
                )
                self._test_pack_hooks(
                    test_fn,
                    inp_fn,
                    [((pack_fp8_with_scale, unpack_fp8_with_scale), True)],
                )
                # Disable testing of Subclasses for now
                # self._test_pack_hooks(test_fn, inp_fn, [(pack_wrapper_sc, unpack_wrapper_sc)])
                # self._test_pack_hooks(
                #     test_fn, inp_fn, [(pack_wrapper_two_tensor, unpack_wrapper_two_tensor)]
                # )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    def test_saved_tensors_hooks_params(self):
        lib = torch.library.Library("_test_aotdispatch_lib", "FRAGMENT")
        logged_shapes = []
        logged_dtypes = []
        lib.define("log(Tensor x) -> Tensor")

        def log_impl(x):
            logged_shapes.append(list(x.shape))
            logged_dtypes.append(x.dtype)
            return x.clone()

        def log_meta(x):
            return x.clone()

        for backend in ["CPU", "CUDA"]:
            lib.impl(
                "log",
                log_impl,
                backend,
            )
        lib.impl("log", log_meta, "Meta")

        def pack_fp8_with_scale_and_log(x):
            torch.ops._test_aotdispatch_lib.log(x)
            return _pack_fp8_with_scale_wrap(x)

        def unpack_fp8_with_scale_and_log(packed):
            return _unpack_fp8_with_scale_wrap(packed)

        def m_inp_fn():
            x = torch.ones(
                2, 2, 2, device=device, dtype=torch.float64, requires_grad=True
            )
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return (x,)

        class SAF0(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gx):
                (saved_x,) = ctx.saved_tensors
                return gx + saved_x

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 2)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(2, 2)

            def forward(self, x):
                x = SAF0.apply(x)
                x = x.to(dtype=torch.float32)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        def _reset_logged():
            logged_shapes.clear()
            logged_dtypes.clear()

        device = torch.device("cuda:0")
        m = M().to(device=device)

        def _test_m():
            self._test_pack_hooks(
                m,
                m_inp_fn,
                [
                    (
                        (
                            pack_fp8_with_scale_and_log,
                            unpack_fp8_with_scale_and_log,
                        ),
                        True,
                    )
                ],
                pre_compile_fn=_reset_logged,
                backend="aot_eager",
            )

        with patch(
            "torch._functorch.config.saved_tensors_hooks_filtering_mode", "donated"
        ):
            _reset_logged()
            _test_m()
            # Check that hooks were not applied to Parameters
            # parameters excluded
            self.assertFalse([2, 2] in logged_shapes)
            self.assertTrue([2, 2, 2] in logged_shapes)
            # input excluded
            self.assertFalse(torch.float64 in logged_dtypes)

        with patch(
            "torch._functorch.config.saved_tensors_hooks_filtering_mode", "no_static"
        ):
            _reset_logged()
            _test_m()
            # Check that hooks were not applied to Parameters
            # parameters excluded
            self.assertFalse([2, 2] in logged_shapes)
            self.assertTrue([2, 2, 2] in logged_shapes)
            self.assertTrue(torch.float64 in logged_dtypes)

        with patch("torch._functorch.config.saved_tensors_hooks_filtering_mode", "all"):
            _reset_logged()
            _test_m()
            # Check that hooks were applied to all saved tensors
            self.assertTrue([2, 2] in logged_shapes)
            self.assertTrue([2, 2, 2] in logged_shapes)
            self.assertTrue(torch.float64 in logged_dtypes)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @torch._functorch.config.patch(saved_tensors_hooks_filtering_mode="all")
    def test_saved_tensors_hooks_recompile(self):
        ctx = torch.autograd.graph.saved_tensors_hooks

        def pack_bf16(x):
            return x.to(dtype=torch.bfloat16)

        def unpack_bf16(x):
            return x.to(dtype=torch.float)

        def pack_mul2(x):
            return x * 2

        def unpack_mul2(x):
            return x / 2

        def _test(hooks, inline, expected_compile_count):
            class SAF(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    (saved_x,) = ctx.saved_tensors
                    return gx + saved_x

            class AF(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    ctx.d1 = x.size(1)
                    return x

                @staticmethod
                def backward(ctx, gx):
                    (saved_x,) = ctx.saved_tensors
                    d1 = ctx.d1
                    return gx + saved_x * d1

            def fn(x):
                x = x.relu()
                x = x + 1
                x = 2 * x
                x = AF.apply(x)
                return x

            device = torch.device("cuda:0")

            def inp_fn():
                x = torch.ones(2, 3, device=device, requires_grad=True)
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                return x

            from torch._dynamo.testing import CompileCounter

            cnt = CompileCounter()
            x = inp_fn()
            y = torch.compile(fn, backend=cnt, fullgraph=True)(x)
            y.sum().backward()

            def _test_with_hooks(hooks):
                with ExitStack() as stack:
                    pack, unpack = hooks
                    if inline:
                        stack.enter_context(
                            ctx(
                                *saved_tensors_hooks_to_gm(
                                    pack, unpack, "pack_hash", "unpack_hash"
                                )
                            )
                        )
                    else:
                        stack.enter_context(ctx(pack, unpack))

                    x = inp_fn()
                    y = torch.compile(fn, backend=cnt, fullgraph=True)(x)
                    y.sum().backward()

            _test_with_hooks(hooks[0])
            _test_with_hooks(hooks[1])
            self.assertEqual(cnt.frame_count, expected_compile_count)

        _test(
            ((pack_bf16, unpack_bf16), (pack_mul2, unpack_mul2)),
            inline=False,
            expected_compile_count=1,
        )
        _test(
            ((pack_bf16, unpack_bf16), (pack_mul2, unpack_mul2)),
            inline=True,
            expected_compile_count=3,
        )

    @torch._functorch.config.patch(donated_buffer=True)
    @torch._functorch.config.patch(saved_tensors_hooks_filtering_mode="no_static")
    def test_saved_tensors_hooks_donated_buffers(self):
        pack_gm, unpack_gm = saved_tensors_hooks_to_gm(
            pack_fp8,
            unpack_fp8,
            "pack_hash",
            "unpack_hash",
        )
        logger_name = "torch._functorch._aot_autograd.graph_compile"

        class SAF(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gx):
                (saved_x,) = ctx.saved_tensors
                return gx + saved_x

        def fn(x):
            x0 = x
            x = SAF.apply(x)
            return x0, torch.nn.functional.relu(x)

        inp = torch.rand([3, 3], requires_grad=True)
        # 1. No donated buffers without hooks, as relu saves input which is also user output.
        with self.assertLogs(logger_name, level="INFO") as captured:
            out = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=False)(
                inp
            )
            out[1].sum().backward()
            expected_msg = "bw_donated_idxs=[]"

        FileCheck().check(expected_msg).run("\n".join(captured.output))

        # 2. Hooks applied for all saved, as we set saved_tensors_hooks_no_filtering=True
        # Results of the hooks become donated buffers.
        inp = torch.rand([3, 3], requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack_gm, unpack_gm):
            with self.assertLogs(logger_name, level="INFO") as captured:
                out = torch.compile(
                    fn, backend="aot_eager", fullgraph=True, dynamic=False
                )(inp)
                out[1].sum().backward()
                expected_msg = "bw_donated_idxs=[0, 1]"

        FileCheck().check(expected_msg).run("\n".join(captured.output))


# entries in here don't work and need to be fixed.
# Each one of these is a bug (or needs to be investigated)
aot_autograd_failures = {
    # data-dependent control flow
    xfail("cov"),
    xfail("nn.functional.gaussian_nll_loss"),
    xfail("tensor_split"),
    xfail("corrcoef"),
    xfail("quantile"),
    xfail("nanquantile"),
    skip("narrow"),
    xfail("istft"),
    xfail("linalg.eig"),
    skip("as_strided_scatter"),
    skip("as_strided", "partial_views"),  # flaky
    # Given input size: (s0xs1x2). Calculated output size: ...
    skip("max_pool2d_with_indices_backward"),
    # Misc
    xfail("to_sparse"),
    xfail("corrcoef"),
    xfail("cov"),
    xfail("chalf"),  # RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'
    xfail("sparse.sampled_addmm"),
    xfail("sparse.mm", "reduce"),
    skip("nn.functional.binary_cross_entropy_with_logits"),  # seems to fail sometimes?
    skip("nn.functional.margin_ranking_loss"),  # seems flaky
    skip("linalg.lu_solve"),  # flaky
    decorate("matmul", decorator=unittest.skipIf(IS_ARM64, "flaky")),
    decorate("__rmatmul__", decorator=unittest.skipIf(IS_ARM64, "flaky")),
    # overrides atol=1e-4, rtol=1e-5 would do as well
    decorate(
        "svd_lowrank",
        decorator=toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-05)}),
    ),
    decorate(
        "linalg.householder_product",
        decorator=unittest.skipIf(IS_MACOS and IS_X86, "flaky"),
    ),
    decorate(
        "linalg.pinv",
        "singular",
        # This delta is coming entirely from the clone() on tangents
        # in AOTDispatcher to make them contiguous
        decorator=toleranceOverride({torch.float32: tol(atol=1e-02, rtol=1e-02)}),
    ),
    decorate(
        "nn.functional.interpolate",
        "bicubic",
        decorator=toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-05)}),
    ),
    # conv2d sometimes nondeterministic in this config?
    decorate("nn.functional.conv2d", decorator=unittest.skipIf(IS_ARM64, "flaky")),
}

if not TEST_MKL:
    aot_autograd_failures.update(
        {
            decorate(
                "matmul",
                decorator=toleranceOverride(
                    {torch.float32: tol(atol=6e-05, rtol=4e-06)}
                ),
            ),
            decorate(
                "__rmatmul__",
                decorator=toleranceOverride(
                    {torch.float32: tol(atol=6e-05, rtol=4e-06)}
                ),
            ),
        }
    )

symbolic_aot_autograd_failures = {
    xfail("combinations", ""),  # aten.masked_select.default
    xfail(
        "index_fill", ""
    ),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail(
        "linalg.lstsq", ""
    ),  # aten.linalg_lstsq.default - couldn't find symbolic meta function/decomposition
    xfail(
        "linalg.lstsq", "grad_oriented"
    ),  # aten.linalg_lstsq.default - couldn't find symbolic meta funct...
    xfail(
        "linalg.lu_solve", ""
    ),  # aten.linalg_lu_solve.default - couldn't find symbolic meta function/deco...
    skip(
        "nn.functional.batch_norm", ""
    ),  # '0 is not tracked with proxy for <torch.fx.experimental.proxy_te..
    xfail(
        "nn.functional.binary_cross_entropy", ""
    ),  # aten.fill_.Scalar - couldn't find symbolic meta funct...
    xfail(
        "nn.functional.cross_entropy", ""
    ),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail(
        "nn.functional.ctc_loss", ""
    ),  # aten._ctc_loss.Tensor - couldn't find symbolic meta function/deco...
    xfail(
        "nn.functional.fractional_max_pool3d", ""
    ),  # rand() received an invalid combination of arguments - g...
    xfail("trace", ""),  # Cannot call sizes() on tensor with symbolic sizes/strides
    decorate(
        "linalg.householder_product",
        decorator=unittest.skipIf(IS_MACOS and IS_X86, "flaky"),
    ),
}


def _test_aot_autograd_helper(
    self,
    device,
    dtype,
    op,
    dynamic=False,
    disable_functionalization=False,
):
    if not op.supports_autograd:
        self.skipTest("Op does not support autograd")

    # aot_autograd_check is able to check data specialization by
    # randomizing the inputs. Here's a list of ops that really do not
    # like random inputs for which we want to disable that.
    cant_check_data_specialization = set(
        {
            "nn.functional.max_unpool1d",
            "nn.functional.max_unpool2d",
            "nn.functional.max_unpool3d",
        }
    )
    try_check_data_specialization = op.name not in cant_check_data_specialization

    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
    for sample_input in sample_inputs_itr:
        t_args = [sample_input.input] + list(sample_input.args)
        t_kwargs = sample_input.kwargs
        try:
            aot_autograd_check(
                op.op,
                t_args,
                t_kwargs,
                dynamic,
                self.assertRaisesRegex,
                self.assertEqual,
                check_gradients=True,
                try_check_data_specialization=try_check_data_specialization,
                skip_correctness_check=op.skip_correctness_check_compile_vs_eager,
                disable_functionalization=disable_functionalization,
            )
        except DynamicOutputShapeException:
            self.skipTest("Dynamic output shape operation in trace")
        except GuardOnDataDependentSymNode:
            # Carveout for getitem; I don't want to xfail the entire test
            # because that will reject known to be good tests see
            # https://github.com/pytorch/pytorch/issues/94705
            if op.name == "__getitem__":
                self.skipTest("Dynamic output shape operation in trace")
            else:
                raise


def _test_aot_autograd_module_helper(
    self, device, dtype, training, module_info, *, dynamic=False
):
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(
        module_info, device=device, dtype=dtype, requires_grad=True, training=training
    )
    for module_input in module_inputs:
        if module_input.forward_input is None:
            continue

        args, kwargs = (
            module_input.constructor_input.args,
            module_input.constructor_input.kwargs,
        )
        m = module_cls(*args, **kwargs)
        m.to(device).to(dtype)
        m.train(training)

        # Lazy modules need to see an input first to initialize params.
        args, kwargs = (
            module_input.forward_input.args,
            module_input.forward_input.kwargs,
        )
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # PackedSequence is only used for RNNs. It might be possible to fake-ify if they're pytrees but
        # torchdynamo already doesn't support RNNs
        if any(tuple(isinstance(flat_arg, PackedSequence) for flat_arg in flat_args)):
            continue

        if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
            with torch.no_grad():
                m(*args, **kwargs)

        sentinel_val = -42
        is_tensor_spec = [
            sentinel_val if isinstance(arg, torch.Tensor) else arg for arg in flat_args
        ]
        args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

        def f(params_buffers_args):
            named_params, named_buffers, args = params_buffers_args
            cur_flat_args = list(is_tensor_spec)
            args = iter(args)
            for idx, v in enumerate(cur_flat_args):
                if v == sentinel_val:
                    cur_flat_args[idx] = next(args)
            c_args, c_kwargs = pytree.tree_unflatten(cur_flat_args, args_spec)
            params_and_buffers = {**named_params, **named_buffers}
            return torch.func.functional_call(m, params_and_buffers, c_args, c_kwargs)

        named_params = dict(m.named_parameters(remove_duplicate=False))
        named_buffers = dict(m.named_buffers(remove_duplicate=False))
        num_params_buffers = len(named_params) + len(named_buffers)
        compiled_f = aot_function(
            f, nop, num_params_buffers=num_params_buffers, dynamic=dynamic
        )
        params_buffers_args = [named_params, named_buffers, args]
        _test_aot_autograd_forwards_backwards_helper(
            f,
            compiled_f,
            params_buffers_args,
            self.assertRaisesRegex,
            self.assertEqual,
            True,
        )


class TestEagerFusionOpInfo(AOTTestCase):
    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestEagerFusionOpInfo", "test_aot_autograd_exhaustive", aot_autograd_failures
    )
    def test_aot_autograd_exhaustive(self, device, dtype, op):
        _test_aot_autograd_helper(self, device, dtype, op)

    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @patch("functorch.compile.config.debug_assert", True)
    @skipOps(
        "TestEagerFusionOpInfo",
        "test_aot_autograd_symbolic_exhaustive",
        aot_autograd_failures | symbolic_aot_autograd_failures,
    )
    def test_aot_autograd_symbolic_exhaustive(self, device, dtype, op):
        _test_aot_autograd_helper(self, device, dtype, op, dynamic=True)

    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @skipOps(
        "TestEagerFusionOpInfo",
        "test_aot_autograd_disable_functionalization_exhaustive",
        aot_autograd_failures,
    )
    def test_aot_autograd_disable_functionalization_exhaustive(self, device, dtype, op):
        _test_aot_autograd_helper(
            self, device, dtype, op, disable_functionalization=True
        )

    @ops(op_db + hop_db, allowed_dtypes=(torch.float,))
    @patch("functorch.compile.config.debug_assert", True)
    @skipOps(
        "TestEagerFusionOpInfo",
        "test_aot_autograd_disable_functionalization_symbolic_exhaustive",
        aot_autograd_failures | symbolic_aot_autograd_failures,
    )
    def test_aot_autograd_disable_functionalization_symbolic_exhaustive(
        self, device, dtype, op
    ):
        _test_aot_autograd_helper(
            self,
            device,
            dtype,
            op,
            dynamic=True,
            disable_functionalization=True,
        )


aot_autograd_module_failures = set(
    {
        torch.nn.CTCLoss,  # torch._subclasses.fake_tensor.DynamicOutputShapeException: aten._ctc_loss.default
        torch.nn.GaussianNLLLoss,  # RuntimeError: It appears that you're trying to get value out
        # of a tracing tensor with aten._local_scalar_dense.default -
        # erroring out! It's likely that this is caused by data-dependent
        # control flow or similar.
        torch.nn.MultiLabelMarginLoss,  # AssertionError: The values for attribute 'shape' do not match:
        # torch.Size([1]) != torch.Size([]). Outputs of the operator are different in
        # eager-mode PyTorch vs AOTAutograd. This means the operator will have incorrect
        # output underneath torch.compile. This could be because the operator's
        # implementation not traceable or that there is a bug in AOTAutograd.
        torch.nn.TransformerEncoder,  # DataDependentOutputException: aten.eq compares a mask input
        # to a causal mask tensor, to see if Boolean is_causal should be set
        # for TransformerEncoder layers, MHA and sdp custom kernels
        torch.nn.Transformer,  # DataDependentOutputException: aten.equal compares a mask input
        # to a causal mask tensor, to see if Boolean is_causal should be set
        # for TransformerEncoder layers, MHA and sdp custom kernels
        # (this bubbles up to Transformer)
    }
)

symbolic_aot_autograd_module_failures = {
    torch.nn.Transformer,  # DataDependentOutputException: aten.equal compares a mask input to a mask producing a bool
    torch.nn.TransformerEncoder,  # DataDependentOutputException: aten.equal compares a mask input to a mask producing a bool
    torch.nn.GaussianNLLLoss,  # NotImplementedError: local_scalar_dense/item NYI for torch.bool
    torch.nn.FractionalMaxPool3d,  # int() argument must be a string, a bytes-like object or a number, not 'SymFloat'
    torch.nn.BCELoss,  # new_size = _infer_size(target.size(), weight.size())
    # RuntimeError: expected int at position 0, but got: SymInt
}


class TestEagerFusionModuleInfo(AOTTestCase):
    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(unittest.expectedFailure, aot_autograd_module_failures)
    def test_aot_autograd_module_exhaustive(self, device, dtype, training, module_info):
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info)

    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(
        unittest.expectedFailure,
        aot_autograd_module_failures | symbolic_aot_autograd_module_failures,
    )
    def test_aot_autograd_symbolic_module_exhaustive(
        self, device, dtype, training, module_info
    ):
        _test_aot_autograd_module_helper(
            self, device, dtype, training, module_info, dynamic=True
        )


instantiate_parametrized_tests(TestAOTAutograd)
instantiate_parametrized_tests(TestAOTModuleSimplified)
only_for = "cpu"
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)
instantiate_device_type_tests(TestEagerFusionModuleInfo, globals(), only_for=only_for)


@xfail_inherited_tests(
    [
        "test_set__and_data_mutation_bad",
        "test_subclass_metadata_mutation_req_grad_True",
        "test_subclass_metadata_mutation_req_grad_False",
    ]
)
class TestAOTAutogradWithDynamo(TestAOTAutograd):
    """
    These are the same as TestAOTAutograd tests, but we run dynamo first to get a graph module.
    """

    def assertExpectedInline(self, *args, **kwargs):
        # These will have different outputs because dynamo returns a different graph module
        # But we don't really care about that assertion when testing with dynamo,
        # only that the outputs match, etc.
        pass

    def make_compiler(self, graph_cell):
        return make_boxed_compiler(partial(extract_graph, graph_cell=graph_cell))

    # Compiler to passes to dynamo
    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: list[Optional[Callable]],
        decompositions: Optional[dict],
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        """
        Runs dynamo and aot_autograd with the specified settings
        """

        def dynamo_compiler(gm, inputs, **kwargs):
            result = aot_module_simplified(
                gm,
                inputs,
                fw_compiler=self.make_compiler(fw_graph_cell),
                bw_compiler=self.make_compiler([None]),
                decompositions=decompositions,
                keep_inference_input_mutations=keep_input_mutations,
                # Dynamic is calculated from whether the inputs have fake tensors
            )
            return result

        def torch_compile_wrapper(*args, **kwargs):
            torch._dynamo.reset()
            fn = torch.compile(f, backend=dynamo_compiler)
            try:
                result = fn(*args, **kwargs)
            except torch._dynamo.exc.BackendCompilerFailed as e:
                # So that assertRaises works properly
                raise e.inner_exception from e
            return result

        return torch_compile_wrapper

    def test_inputs_overlapping_unsqueeze_with_mutation(self):
        def f(x, y):
            x.add_(1)
            y.add_(1)
            return x

        def run(f):
            base = torch.ones(10)
            inputs = [base.unsqueeze(0), base.unsqueeze(0)]
            return f(*inputs)

        optf = torch.compile(backend="aot_eager", dynamic=True)(f)

        out = run(f)
        optout = run(optf)

        self.assertEqual(out, optout)

    def test_inputs_overlapping_with_mutation_guard_base(self):
        def f(x, y):
            x.add_(1)
            y.add_(1)
            return x

        def run(f):
            base = torch.ones(10)
            inputs = [base[1:], base[1:]]
            return f(*inputs)

        optf = torch.compile(backend="aot_eager", dynamic=True)(f)

        out = run(f)
        optout = run(optf)

        self.assertEqual(out, optout)

    def test_mutations_in_bw_detached_from_tangent(self):
        class AF(torch.autograd.Function):
            @staticmethod
            def forward(ctx, dummy, inplace_tensor):
                ctx.inplace_tensor = inplace_tensor
                return dummy.clone()

            @staticmethod
            def backward(ctx, grad_output):
                inplace_tensor = ctx.inplace_tensor
                gradient_attachment = grad_output * 0 + 1
                inplace_tensor.add_(1 * gradient_attachment)
                return grad_output, None, None

        def fn(dummy, inplace_tensor):
            return AF.apply(dummy, inplace_tensor)

        def _inps():
            dummy = torch.zeros((2,), requires_grad=True)
            inplace_tensor = torch.zeros((2,), requires_grad=False)
            return dummy, inplace_tensor

        inps = _inps()
        out = fn(*inps)
        ref_inps_after_fw = [x.clone().detach() for x in inps]
        out.sum().backward()
        ref_inps_after_bw = [x.clone().detach() for x in inps]

        inps = _inps()
        out = torch.compile(fn, backend="aot_eager", fullgraph=True)(*inps)
        inps_after_fw = [x.clone().detach() for x in inps]
        out.sum().backward()
        inps_after_bw = [x.clone().detach() for x in inps]

        self.assertEqual(ref_inps_after_fw, inps_after_fw)
        self.assertEqual(ref_inps_after_bw, inps_after_bw)

    def test_mutation_of_input_in_fw_and_bw(self):
        class AF(torch.autograd.Function):
            @staticmethod
            def forward(ctx, dummy, inplace_tensor):
                inplace_tensor.add_(1)

                ctx.inplace_tensor = inplace_tensor
                return dummy.clone()

            @staticmethod
            def backward(ctx, grad_output):
                inplace_tensor = ctx.inplace_tensor
                inplace_tensor.add_(1)
                return grad_output, None, None

        def fn(dummy, inplace_tensor):
            return AF.apply(dummy, inplace_tensor)

        def inps():
            dummy = torch.randn((2,), requires_grad=True)
            inplace_tensor = torch.zeros((2,), requires_grad=False)
            return dummy, inplace_tensor

        def sc_inps():
            dummy = TwoTensor(
                torch.randn((2,), requires_grad=True),
                torch.randn((2,), requires_grad=True),
            )
            inplace_tensor = TwoTensor(
                torch.zeros((2,), requires_grad=False),
                torch.zeros((2,), requires_grad=False),
            )
            return dummy, inplace_tensor

        for _inps in [inps, sc_inps]:
            dummy, inplace = _inps()
            y = fn(dummy, inplace)
            ref0 = inplace.clone().detach()
            y.sum().backward()
            ref = inplace.clone().detach()

            dummy, inplace = _inps()
            y = torch.compile(fn, backend="aot_eager", fullgraph=True)(dummy, inplace)
            self.assertEqual(ref0, inplace)
            y.sum().backward()
            self.assertEqual(ref, inplace)


class MockFXGraphCache:
    """
    In memory version of FXGraphCache so we can isolate testing for FXGraphCache
    """

    def __init__(self) -> None:
        self.cache = {}

    def save(self, key, gm):
        self.cache[key] = gm

    def load(self, gm, inputs):
        key, _ = compiled_fx_graph_hash(gm, inputs, {}, [])
        if key not in self.cache:
            self.cache[key] = gm
        gm, _ = self.load_with_key(key, [], inputs, None, None, None, None, None)
        return gm

    def load_with_key(
        self,
        key,
        debug_lines,
        inputs,
        local,
        remote_cache,
        is_backward,
        constants,
        evaluate_guards,
    ):
        gm = self.cache.get(key)
        if gm is not None:
            gm = make_boxed_func(gm)
            gm = MockFXGraphCacheOutput(gm)
            gm._fx_graph_cache_key = key  # (cache_key, debug lines)
            gm._fx_graph_cache_debug_lines = []
            gm._time_taken_ns = 0
        return gm, {}


# The following tests fail in strict caching mode (i.e. they bypass or
# cache miss instead of cache hitting). They will be fixed in the PRs above this.
FAILING_CACHE_TESTS = (
    # BypassAOTAutogradCache: unsupported nodes
    "test_backward_mutation_data",  # Custom Autograd Function
    "test_backward_mutation_metadata",  # Custom Autograd Function
    "test_input_output_aliase_custom_autograd_function",
)


@xfail_inherited_tests(FAILING_CACHE_TESTS)
class TestAOTAutogradWithCache(TestAOTAutogradWithDynamo):
    """
    In memory version of FXGraphCache so we can isolate testing for FXGraphCache
    """

    def make_compiler(self, fw_graph_cell):
        mock_inductor_cache = self.inductor_cache

        def compiler(gm, example_inputs):
            nonlocal mock_inductor_cache, fw_graph_cell
            result = mock_inductor_cache.load(gm, example_inputs)
            fw_graph_cell[0] = gm
            return result

        compiler = SerializableAOTDispatchCompiler(MockFXGraphCacheOutput, compiler)
        return compiler

    def run_autograd(
        self,
        f: Callable,
        fw_graph_cell: list[Optional[Callable]],
        decompositions: Optional[dict],
        keep_input_mutations: bool,
        dynamic: bool,
    ):
        return super().run_autograd(
            f,
            fw_graph_cell,
            decompositions,
            keep_input_mutations,
            dynamic,
        )

    @torch._functorch.config.patch(
        {
            "enable_autograd_cache": True,
            "strict_autograd_cache": True,
        }
    )
    @torch._inductor.config.patch("fx_graph_cache", True)
    def verify_aot_autograd(
        self,
        f,
        inp_: Union[Callable, list[Any]],
        *,
        test_mutation: bool = False,
        keep_inp_mutations: bool = False,
        decompositions: Optional[dict] = None,
        dynamic: bool = False,
        # Only active when inp_ is Callable.
        # TODO: probably consolidate all tests to make inp a Callable.
        make_inputs_subclasses: bool = False,
    ):
        self.inductor_cache = MockFXGraphCache()
        AOTAutogradCache.clear()
        with patch(
            "torch._inductor.codecache.FxGraphCache.load_with_key",
            new=self.inductor_cache.load_with_key,
        ):
            return super().verify_aot_autograd(
                f,
                inp_,
                test_mutation=test_mutation,
                keep_inp_mutations=keep_inp_mutations,
                decompositions=decompositions,
                dynamic=dynamic,
                make_inputs_subclasses=make_inputs_subclasses,
            )

    def test_input_mutation_false_aliasing(self):
        # This test is disabled because it fails in strict cache mode
        # But also can't be xfailed because it causes undefined behavior for
        # ASAN
        self.skipTest("Skipping because it fails in strict cache mode")


if __name__ == "__main__":
    run_tests()
