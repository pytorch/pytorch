# Owner(s): ["module: functorch"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Callable, List, Any, Optional, Dict
from unittest.mock import patch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    IS_ARM64,
    IS_WINDOWS,
    compare_equal_outs_and_grads,
    outs_and_grads
)
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import unittest
import warnings
import itertools
from functools import partial
from torch.nn.utils import stateless
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, wrapper_set_seed
from torch.testing._internal.common_modules import module_db, modules
from functorch import (
    grad, vjp, vmap, jacrev,
    make_fx
)
from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import (
    nnc_jit, compiled_function, compiled_module,
    min_cut_rematerialization_partition, aot_function, aot_module,
    nop, default_partition, default_decompositions,
    memory_efficient_fusion, get_aot_compilation_context
)
from torch._decomp import decomposition_table

from torch.testing._internal.common_device_type import ops
from common_utils import (
    decorate,
    xfail,
    skip,
    skipOps,
    decorateForModules,
)
from torch._subclasses.fake_tensor import DynamicOutputShapeException, FakeTensorMode
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._functorch.named_members_polyfill import _named_buffers, _named_parameters

USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)

USE_NETWORKX = False
try:
    import networkx  # noqa: F401
    USE_NETWORKX = True
except ImportError:
    warnings.warn("Some tests use networkx but it was not installed",
                  UserWarning)

try:
    import sympy  # noqa: F401
    # TODO(jansel): these tests fail on windows
    HAS_SYMPY = not IS_WINDOWS
except ImportError:
    HAS_SYMPY = False
skipIfNoSympy = unittest.skipIf(not HAS_SYMPY, "no sympy")

# NB: numpy is a testing dependency!

class AOTTestCase(TestCase):
    def setUp(self):
        super().setUp()

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
        ops = set([i.target for i in fx_f.graph.nodes])

        self.assertEqual(torch.ops.aten.tanh_backward in ops, True)

        fx_f = make_fx(grad(f), decomposition_table)(torch.randn(5))
        ops = set([i.target for i in fx_f.graph.nodes])
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
            x['a'] = x['a'] * 2
            return x
        inp = ({'a': torch.randn(3), 'b': torch.randn(3)},)
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

class TestAOTAutograd(AOTTestCase):
    # test_mutation will:
    # - Ensure that inputs are non-leaves, so our graphs can mutate them
    # - try to mutate outputs of the graph (to ensure that autograd meta is set properly on outputs)
    def verify_aot_autograd(
        self,
        f,
        inp: Union[Callable, List[Any]],
        *,
        test_mutation: bool = False,
        return_fw_graph: bool = False,
        decompositions: Optional[Dict] = None,
    ):
        # Some tests pass in a callable for inp, to generate the inputs
        # (useful if we want to generate complicated aliasing inputs)
        if isinstance(inp, Callable):
            inp_callable = inp
            # The callable should return a tuple of f_inputs, f_graph_inputs
            # (The idea is that we might want to compile a function with the graph inputs,
            # but test autograd backprop all the way through the actual inputs)
            inp_copy, graph_inps_copy = inp_callable()
            inp, graph_inps = inp_callable()
        else:
            inp_copy = []
            # Our input clones need to mimic when inputs are duplicates of one another
            dupes_map = {}
            for i, x in enumerate(inp):
                if x in dupes_map:
                    x_dupe_idx = dupes_map[x]
                    inp_copy.append(inp_copy[x_dupe_idx])
                else:
                    dupes_map[x] = i
                    x_copy = x.clone().detach().requires_grad_(x.requires_grad)
                    if x.requires_grad and not x.is_leaf:
                        x_copy = x_copy.clone()
                    inp_copy.append(x_copy)

            if test_mutation:
                # For graphs where we mutate inputs, need our test to make sure inputs aren't leaves
                graph_inps = [x.add(1) for x in inp]
                graph_inps_copy = [x.add(1) for x in inp_copy]
            else:
                graph_inps = inp
                graph_inps_copy = inp_copy

        # Create a copy of inputs, so we can test input mutation correctness.

        fw_graph_cell = [None]
        if isinstance(f, nn.Module):
            compiled_f = aot_module(
                f, fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell), bw_compiler=nop, decompositions=decompositions)
        else:
            compiled_f = aot_function(
                f, fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell), bw_compiler=nop, decompositions=decompositions)
        ref_out, ref_grad = outs_and_grads(f, graph_inps, inp)
        test_out, test_grad = outs_and_grads(compiled_f, graph_inps_copy, inp_copy)
        self.assertEqual(ref_grad, test_grad)

        if isinstance(ref_out, torch.Tensor):
            self.assertTrue(isinstance(test_out, torch.Tensor))
            ref_out, test_out = [ref_out], [test_out]
        for ref_o, test_o in zip(ref_out, test_out):
            if isinstance(ref_o, torch.Tensor):
                self.assertEqual(ref_o.requires_grad, test_o.requires_grad)
                self.assertEqual(ref_o.is_leaf, test_o.is_leaf)
                if ref_o.requires_grad:
                    # _is_view() should probably unconditionally be the same,
                    # but in practice I don't think this matters for tensors that don't require grad
                    self.assertEqual(ref_o._is_view(), test_o._is_view())
                self.assertEqual(ref_o, test_o)
                if test_mutation:
                    # This tests that autograd meta is set properly on the output we can
                    # mutate it.
                    ref_o.mul_(2)
                    test_o.mul_(2)
                    self.assertEqual(ref_o, test_o)
        for ref_i, test_i in zip(inp, inp_copy):
            if isinstance(ref_i, torch.Tensor):
                self.assertEqual(ref_i.requires_grad, test_i.requires_grad)
                self.assertEqual(ref_i, test_i)
        if return_fw_graph:
            return fw_graph_cell[0]

    def test_single_output(self):
        def f(a, b):
            return a + b
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output(self):
        def f(a, b):
            return a + b, a - b
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    def test_multi_output_list(self):
        def f(a, b):
            return [a + b, a - b]
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3)]
        self.verify_aot_autograd(f, inp)

    # Test for bug occurring at the intersection of fake tensors & functionalization.
    @patch("torch._functorch.config.use_dynamic_shapes", True)
    @patch("torch._functorch.config.use_fake_tensor", True)
    def test_squeeze_mutation(self):
        def f(a):
            b = a.clone().squeeze(-1)
            b.add_(1.)
            return a + b

        inp = [torch.randn(3, 1, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

    @patch("torch._functorch.config.use_dynamic_shapes", True)
    @patch("torch._functorch.config.use_fake_tensor", True)
    def test_embedding_bag_view(self):
        # Backwards pass tries to wrap a sparse tensor in a FunctionalTensorWrapper;
        # test that this works even though the sparse tensor has no storage.

        class F(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.EmbeddingBag(100, 8, sparse=True)

            def forward(self, x, y):
                return self.emb(x, y).view(-1)

        x = torch.arange(3)
        y = torch.arange(3)
        self.verify_aot_autograd(F(), [x, y])

    def test_input_mutation_simple(self):
        def f(a):
            a.mul_(2)
            return a * 3
        inp = [torch.ones(3, 3, requires_grad=True)]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        # Things to note:
        # - the extra clone is because we need to pass the pre-mutated input to grad(),
        #   but autograd operates above functionalization so we need to manually clone.
        #   Hopefully backends can optimize this easily.
        # - The extra return arg is because the compiled forward returns (mutated inputs + outputs)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, 3)
    return [mul, mul_1]""")

    def test_input_mutation_simple_with_none_and_nontensor(self):
        # Tensor, None, int
        def f(a, b, c):
            return a * c
        inp = [torch.ones(3, 3, requires_grad=True), None, 3]

        f_compiled = aot_function(f, nop)
        out_ref = f(*inp)
        out_test = f_compiled(*inp)
        self.assertEqual(out_ref, out_test)

    def test_input_mutation_is_output(self):
        def f(a):
            a.mul_(2)
            return a
        inp = [torch.ones(3, 3, requires_grad=True)]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    return [mul]""")

    def test_input_mutation_multiple(self):
        def f(a, b, c):
            a.mul_(2)
            c.mul_(2)
            return a + b + c

        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 2);  clone_1 = None
    add = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
    add_1 = torch.ops.aten.add.Tensor(add, mul_1);  add = None
    return [mul, mul_1, add_1]""")

    def test_input_mutation_metadata(self):
        def f(a, b):
            a.transpose_(1, 0)
            return a + b
        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True),
        ]

        self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)

    def test_input_mutation_metadata2(self):
        def f(a):
            a.transpose_(1, 0)
            a.mul_(2)
            return a + 1
        inp = [torch.ones(3, 3, requires_grad=True)]

        self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)

    def test_input_mutation_resize_smaller(self):
        def f(a, b):
            a.resize_(2, 2)
            return a + b
        # tenors that require gradients cannot be resized, so only test requires_grad=False case
        inp = [
            torch.ones(3, 3),
            torch.ones(2, 2, requires_grad=True),
        ]

        self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)

    def test_input_mutation_batchnorm(self):
        def f(inpt, weight, bias, running_mean, running_var):
            # This is additionally a good test, because the input tensors that we mutate
            # are *also* saved for backwards.
            # This tests that what we save for the backward is actually cloned inputs,
            # and not the original inputs that got mutated.
            return torch._native_batch_norm_legit(inpt, weight, bias, running_mean, running_var, True, 0.5, 1e-5)
        inp = [
            torch.ones(2, 5, 5, 5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5, requires_grad=True),
            torch.ones(5),
            torch.ones(5),
        ]

        from torch._decomp import get_decompositions
        # This simulates what inductor does (running the fw + bw decompositions)
        decompositions = get_decompositions([
            torch.ops.aten._native_batch_norm_legit_functional,
            torch.ops.aten.native_batch_norm_backward,
        ])
        self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True, decompositions=decompositions)

    def test_input_output_view_simple(self):
        def f(a):
            return a.view(-1)
        inp = [
            torch.ones(2, 2, requires_grad=True).add(1),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        # Outputs that alias inputs are pulled out of the graph entirely, so we don't compile anything here
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    return [4, 1, 0]""")

    def test_input_output_view_mutate_multiple(self):
        def f(a, b, c):
            a.mul_(2)
            c.mul_(3)
            return b.view(2, 2), c.view(2, 2)
        inp = [
            torch.ones(2, 2, requires_grad=True).add(1),
            torch.ones(2, 2, requires_grad=True).add(1),
            torch.ones(2, 2, requires_grad=True).add(1),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        # The original function returned two outputs, both of which aliased inputs.
        # We expect two outputs in the functional graph, a_updated and c_updated.
        # The actual aliased outputs themselves aren't in the compiled forward graph;
        # Instead, they're generated outside of  the graph.
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    clone_1 = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    mul = torch.ops.aten.mul.Tensor(clone, 2);  clone = None
    mul_1 = torch.ops.aten.mul.Tensor(clone_1, 3);  clone_1 = None
    return [mul, mul_1, 2, 2, 2, 1, 0, 2, 2, 2, 1, 0]""")

    def test_input_output_view_metadata_mutate_multiple(self):
        def f(a, b, c):
            b.mul_(3)
            c.t_()
            return a.view(2, 2), b.view(2, 2), c.view(2, 2)
        inp = [
            torch.ones(2, 2, requires_grad=True).add(1),
            torch.ones(2, 2, requires_grad=True).add(1),
            torch.ones(2, 2, requires_grad=True).add(1),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        # Important thing to check here: of the three inputs:
        # Only the b.mul_(3) should show up in the graph (we functionalize it and return it).
        # Everything else that does not show up in the graph includes:
        # - The metadata mutation on c (we do it outside the graph)
        # - All 3 original fw outputs, which are aliases of inputs (we regenerate them outside of the graph)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_2);  primals_2 = None
    mul = torch.ops.aten.mul.Tensor(clone, 3);  clone = None
    return [mul, 2, 2, 1, 2, 0, 2, 2, 2, 1, 0, 2, 2, 2, 1, 0, 2, 2, 1, 2, 0]""")

    def test_input_mutation_and_output_view(self):
        def f(a):
            a.add_(1)
            return a.view(-1)
        inp = [
            torch.ones(2, 2, requires_grad=True).add(1),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        # Here, total # of outputs is 1 because:
        # - num_mutated_inps = 1 (a_updated)
        # - num_fw_outputs = 0 (the output is an alias of the input, so we move it outside the compiled fw)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    return [add, 4, 1, 0]""")


    def test_input_mutation_output_view_multiple(self):
        def f(a, b, c, d):
            b.transpose_(1, 0)
            c.add_(1)
            return d + 1, b.diagonal(), a + c
        inp = [
            torch.arange(4, requires_grad=True, dtype=torch.float32).view(2, 2).add(1),
            torch.arange(4, requires_grad=True, dtype=torch.float32).view(2, 2).add(1),
            torch.ones(2, 2, requires_grad=True).add(1),
            torch.ones(2, 2, requires_grad=True).add(1),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2, primals_3, primals_4):
    clone = torch.ops.aten.clone.default(primals_3);  primals_3 = None
    add = torch.ops.aten.add.Tensor(clone, 1);  clone = None
    add_1 = torch.ops.aten.add.Tensor(primals_4, 1);  primals_4 = None
    add_2 = torch.ops.aten.add.Tensor(primals_1, add);  primals_1 = None
    return [add, add_1, add_2, 2, 2, 1, 2, 0, 2, 3, 0]""")


    def test_input_data_and_metadata_mutation(self):
        def f(a):
            a.t_()
            a[0].mul_(2)
            return a.view(a.shape)
        inp = [torch.ones(3, 3, requires_grad=True)]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    t = torch.ops.aten.t.default(clone)
    select = torch.ops.aten.select.int(t, 0, 0);  t = None
    mul = torch.ops.aten.mul.Tensor(select, 2);  select = None
    t_1 = torch.ops.aten.t.default(clone);  clone = None
    select_scatter = torch.ops.aten.select_scatter.default(t_1, mul, 0, 0);  t_1 = mul = None
    t_2 = torch.ops.aten.t.default(select_scatter);  select_scatter = None
    t_3 = torch.ops.aten.t.default(t_2);  t_2 = None
    return [t_3, 3, 3, 1, 3, 0]""")

    def test_view_and_inplace_view(self):
        def f(a, b):
            a.t_()
            return b.view(b.shape), a.view(a.shape)
        inp = [
            torch.ones(3, 3, requires_grad=True),
            torch.ones(3, 3, requires_grad=True)
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2):
    return [3, 3, 1, 3, 0, 3, 3, 3, 1, 0, 3, 3, 1, 3, 0]""")

    def test_view_detach(self):
        def f(a):
            tmp = a.detach()
            a.mul_(2)
            return a, tmp
        inp = [
            torch.ones(3, 3, requires_grad=True),
        ]

        self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)

    def test_input_inplace_requires_grad_true(self):
        def f(a, b):
            a.requires_grad_(True)
            return a.mul(3), b.mul(4)
        inp = [
            # First inp doesnt require grad, but we switch it on
            torch.ones(3, 3, requires_grad=False),
            torch.ones(3, 3, requires_grad=True),
        ]

        fw_graph = self.verify_aot_autograd(f, inp, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2):
    mul = torch.ops.aten.mul.Tensor(primals_1, 3);  primals_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(primals_2, 4);  primals_2 = None
    return [mul, mul_1]""")

    def test_input_data_and_metadata_mutation_aliases_other_input(self):
        # a and b are aliased
        def f(a, b):
            a.t_()
            b.mul_(2)
            return a.mul(3)

        def inp_callable():
            base = torch.ones(2, 2, requires_grad=True)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.add(1)
            inp1 = x.view(-1)
            inp2 = x.view(-1)
            return [base], [inp1, inp2]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        # Important parts of the graph:
        # - the compiled graph takes in a base, and we generate a and b (the views) off of the base
        # - clone() is still in the graph, because we need to call grad() on the original (non-mutated) inputs
        # - We re-generate the views *after* the clone, to preserve view relationships.
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = None
    as_strided_5 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0);  as_strided_scatter = None
    t_1 = torch.ops.aten.t.default(as_strided_5);  as_strided_5 = None
    mul_1 = torch.ops.aten.mul.Tensor(t_1, 3);  t_1 = None
    return [mul, mul_1, 4, 1, 0]""")

    def test_input_mutation_aliases_other_input(self):
        def f(a, b):
            a.add_(1)
            return a + b

        def inp_callable():
            base = torch.ones(2, 2, requires_grad=True)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            x = base.add(1)
            inp1 = x[0]
            inp2 = x[1]
            return [base], [inp1, inp2]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        # Important parts of the graph:
        # - the compiled graph takes in a base, and we generate a and b (the views) off of the base
        # - clone() is still in the graph, because we need to call grad() on the original (non-mutated) inputs
        # - We re-generate the views *after* the clone, to preserve view relationships.
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = None
    as_strided_4 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [1], 2);  as_strided_scatter = None
    add_1 = torch.ops.aten.add.Tensor(add, as_strided_4);  as_strided_4 = None
    return [add, add_1]""")

    def test_input_mutation_aliases_other_input2(self):
        def f(a, b):
            a.add_(1)
            return a + b

        def inp_callable():
            base = torch.ones(2, 2, requires_grad=True)
            x = base.add(1)
            inp1 = x[0]
            # Here, one of the aliased inputs is the base itself
            inp2 = x
            return [base], [inp1, inp2]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [2], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [2], [1], 0);  clone = None
    as_strided_4 = torch.ops.aten.as_strided.default(as_strided_scatter, [2, 2], [2, 1], 0);  as_strided_scatter = None
    add_1 = torch.ops.aten.add.Tensor(add, as_strided_4);  as_strided_4 = None
    return [add, add_1]""")

    def test_input_mutation_aliases_and_output_alias(self):
        def f(a, b):
            # Here, we need to take care:that because and b are aliased
            # (1) since a and b are aliased, we generate a view off of "updated b"
            # (2) We're returning a view, which doesn't show up in the graph
            a.add_(1)
            return b.view(b.shape)

        def inp_callable():
            base = torch.ones(2, 2, requires_grad=True)
            x = base.add(1)
            return [base], [x.view(-1), x.view(-1)]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0);  clone = None
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    return [add, 4, 1, 0]""")

    def test_input_aliased_with_mutation_output_alias(self):
        def f(a, b, c):
            # a and c alias
            c.mul_(2)
            # The main thing we're testing here is that
            # (1) We need to reconstruct c.view(-1) from the 3rd input to the forward
            # (2) But we need to be careful to do this *before* converting aliased inputs into synthetic bases.
            #     The original fw takes in 3 args, but the compiled fw takes in only 2 args.
            return b.add(1), c.view(-1)

        def inp_callable():
            base1 = torch.ones(2, 2, requires_grad=True)
            base2 = torch.ones(2, 2, requires_grad=True)
            x = base1.add(1)
            y = base2.add(1)
            return [base1, base2], [x.view(-1), y, x.view(-1)]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0);  clone = None
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    add = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    return [mul, add, 4, 1, 0]""")

    def test_input_metadata_mutation_aliases(self):
        def f(a, b):
            # a and b alias, and we do a metadata mutation on a
            # Since we're not mutating data, then b isn't affected at all.
            # We expect aot autograd to not bother with constructing a synthetic base.
            a.t_()
            return a + b

        def inp_callable():
            base = torch.ones(2, 2, requires_grad=True)
            x = base.add(1)
            return [base], [x.view(-1), x.view(-1)]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        # Expectation: fwd() takes in 2 args, and we don't construct a synthetic base.
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2):
    view = torch.ops.aten.view.default(primals_1, [4]);  primals_1 = None
    t = torch.ops.aten.t.default(view);  view = None
    add = torch.ops.aten.add.Tensor(t, primals_2);  t = primals_2 = None
    return [add, 4, 1, 0]""")

    def test_input_mutation_aliases_and_none_require_gradients(self):
        def f(a, b, c):
            # a and b alias, but neither require gradients (so they don't have a _base)
            # aot autograd should construct the synthetic base from `torch.Tensor(a.storage())`
            a.mul_(2)
            return b + 1, c + 1

        def inp_callable():
            base = torch.ones(2, 2)
            c_arg = torch.ones(2, 2, requires_grad=True)
            x = base.add(1)
            return [base, c_arg], [x.view(-1), x.view(-1), c_arg]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided, 2);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = None
    as_strided_2 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0);  as_strided_scatter = None
    add = torch.ops.aten.add.Tensor(as_strided_2, 1);  as_strided_2 = None
    add_1 = torch.ops.aten.add.Tensor(primals_2, 1);  primals_2 = None
    return [mul, add, add_1]""")

    def test_input_mutation_aliases_bases_out_of_order(self):
        # This tests our calling convention: if b and d are aliased, then the outer calling convention
        # that we send to the compiled forward becomes:
        # (b_d_base, a, c)
        # Importantly, even though a and c alias in our test, neither inputs are mutated,
        # So we don't need to do the base construction / deconstruction
        def f(a, b, c, d):
            b.add_(1)
            d.t_()
            return a + c + d, b.view(-1)

        def inp_callable():
            base1 = torch.ones(2, 2, requires_grad=True)
            base2 = torch.ones(2, 2, requires_grad=True)
            x1 = base1.add(1)
            x2 = base2.add(1)
            # a and c alias, b and d alias
            return [base1, base2], [x1.view(-1), x2.view(-1), x1.view(-1), x2.view(-1)]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        # 3 graph inputs: (b_d_base, a, c)
        # 2 returns: (b_updated, a+c+d)
        # (there are 2 original fw outs, but one is a view of b so it's not part of the graph)
        # (there are also 2 input mutations, but one is a metadata-only mutation so the compiled forward doesn't return it)
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2, primals_3):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    add_1 = torch.ops.aten.add.Tensor(primals_2, primals_3);  primals_2 = primals_3 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, add, [4], [1], 0);  clone = None
    as_strided_4 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0);  as_strided_scatter = None
    t_1 = torch.ops.aten.t.default(as_strided_4);  as_strided_4 = None
    add_2 = torch.ops.aten.add.Tensor(add_1, t_1);  add_1 = t_1 = None
    return [add, add_2, 4, 1, 0, 4, 1, 0]""")

    # Mondo test that tests a combination of:
    # input is mutated, that aliases another input (so we make a synthetic base)
    # an output is an alias of another output
    # an output is an alias of an intermediate
    def test_input_mutation_alias_everything(self):
        # a and c are aliased
        def f(a, b, c):
            c.mul_(2)  # mutates c
            b.t_()  # metadata mutate b
            tmp = a + c
            # TODO: this test doesn't test "alias of an intermediate" yet,
            # delete this line later and get that to be tested
            return tmp, b.t(), a
            out1 = tmp.view(-1)
            out2 = b.t()
            out3 = out1.unsqueeze(0)
            # out1 and out3 are aliases of an intermediate, and alias each other!
            # out2 aliases an input, so we don't return it
            return out1, out2, out3

        def inp_callable():
            base1 = torch.ones(2, 2, requires_grad=True)
            base2 = torch.ones(2, 2, requires_grad=True)
            # Note: in our test, the add() is important because we need the graph inputs to be non-leaves so we can mutate them.
            base1_ = base1.add(1)
            base2_ = base2.add(1)
            a = base1_.view(-1)
            b = base2_
            c = base1_.view(-1)
            return [base1, base2], [a, b, c]

        fw_graph = self.verify_aot_autograd(f, inp_callable, test_mutation=True, return_fw_graph=True)
        # Expected:
        # - 2 inputs in the forward: synthetic_base_a_c, b
        # - 1 output in the forward: "tmp"
        #   out2 is an alias of an input, and will be generated off of b outside of the compiled fn
        #   out1 and out3 are aliases of tmp, that we generate outside of the compiled function
        self.assertExpectedInline(fw_graph.code.strip(), """\
def forward(self, primals_1, primals_2):
    clone = torch.ops.aten.clone.default(primals_1);  primals_1 = None
    as_strided_1 = torch.ops.aten.as_strided.default(clone, [4], [1], 0)
    mul = torch.ops.aten.mul.Tensor(as_strided_1, 2);  as_strided_1 = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(clone, mul, [4], [1], 0);  clone = None
    as_strided_4 = torch.ops.aten.as_strided.default(as_strided_scatter, [4], [1], 0);  as_strided_scatter = None
    add = torch.ops.aten.add.Tensor(as_strided_4, mul);  as_strided_4 = None
    return [mul, add, 2, 2, 1, 2, 0, 2, 2, 2, 1, 0]""")

    def test_no_grad_input_output(self):
        def f(a, b):
            return a.cos(), b.cos(), a * b

        inp_thunks = [lambda: torch.randn(5, requires_grad=True), lambda: torch.randn(5, requires_grad=False)]
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
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

    def test_some_outputs_dont_require_grad_non_view(self):
        def f(a, b):
            return a.add(1).detach(), b
        inp = [torch.randn(3, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True)]
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
            return {'a': x, 'b': x}
        inp = [torch.randn(3, 3, requires_grad=True)]
        self.verify_aot_autograd(f, inp)

        def f(x, y):
            return {'a': x, 'b': y + x}
        inp = [torch.randn(3, requires_grad=True), torch.randn(3)]
        self.verify_aot_autograd(f, inp)

        def f(x):
            new_d = {}
            for k in x:
                new_d[k] = x[k] * 2
            return new_d

        def inp_callable():
            inps = [{'a': torch.randn(3, requires_grad=True), 'b': torch.randn(3, requires_grad=True)}]
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

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
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
        self.assertEqual(count, [(['0_forward'], 4), (['1_inference'], 4), (['0_backward'], 8)])

    def test_dupe_arg(self):
        def f(x, y):
            return x + y

        x = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(f, [x, x])

    def test_dupe_arg_torture(self):
        def f(x, y):
            x.t_()
            y.t_()
            return x + y

        x = torch.randn(3, 3, requires_grad=True).clone()
        self.verify_aot_autograd(f, [x, x])

    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
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
            AssertionError, lambda: fxx(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."""  # noqa: B950
        )


    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe(self, counter):
        self._test_invalid_dupe(counter, fake=False)

    # See Note: Dynamo recompilation guarding invalid grad for why this test exists
    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_dupe_fake(self, counter):
        self._test_invalid_dupe(counter, fake=True)


    def _test_invalid_dupe(self, counter, fake):
        class F(torch.nn.Module):
            def forward(self, x, y):
                x.t_()
                y.t_()
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
        fxy(x, x)  # is ok!

        if fake:
            fxx = aot_module_simplified(F(), (fake_x, fake_x), nop)
        else:
            fxx = aot_module_simplified(F(), (x, x), nop)

        fxx(x, x)
        # Note This should not raise! Once we have guards in place here,
        # we will have this working correctly, as it should recompile.
        self.assertExpectedRaisesInline(
            AssertionError, lambda: fxx(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would be a duplicate of input 0, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."""  # noqa: B950
        )


    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
    @patch("torch._functorch.config.debug_assert", True)
    def test_invalid_requires_grad(self, counter):
        self._test_invalid_requires_grad(counter, fake=False)

    # See Note: Dynamo recompilation guarding invalid grad for why this test exists
    @patch('torch._functorch.aot_autograd.AOT_COUNTER', new_callable=itertools.count)
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
            AssertionError, lambda: fxz(x, y),
            """At compilation time, graph 1 was compiled under the assumption that input 1 would not require grad, but at runtime this was not the case.  This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."""  # noqa: B950
        )

    def test_resize_input(self):
        def f(x, y):
            y.resize_(4)
            y.zero_()
            self.assertEqual(x.shape, (4,))
            return y

        # NB: don't use verify_aot_autograd as the inputs get
        # mutated and I don't trust verify to do it right

        compiled_f = aot_function(f, nop)
        ref_x = torch.randn(0)
        ref_out = f(ref_x, ref_x)

        test_x = torch.randn(0)
        test_out = compiled_f(test_x, test_x)

        self.assertEqual(ref_out, test_out)

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
        guard = torch._C._DisableAutocast()
        try:
            x = torch.rand([4, 4]).cuda()
            y = x @ x
            self.assertEqual(y.dtype, torch.float32)
        finally:
            del guard

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_nonidempotent_amp(self):
        def f(self_s_emb, add_3):
            einsum_2 = torch.functional.einsum('ah,th->t', self_s_emb, add_3)
            log_softmax_2 = einsum_2.log_softmax(-1)
            return (log_softmax_2,)

        args = [torch.rand((1, 256), dtype=torch.float32, device='cuda'), torch.rand((30, 256), dtype=torch.float16, device='cuda')]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

        args = [e.requires_grad_(True) for e in args]
        with torch.cuda.amp.autocast(enabled=True):
            self.verify_aot_autograd(f, args)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_batch_norm_amp(self):
        device = "cuda"
        input_dtype = torch.float16
        param_dtype = torch.float32
        weight, bias = [torch.ones(64, device=device, dtype=param_dtype, requires_grad=True) for _ in range(2)]
        running_mean, running_var = [torch.ones(64, device=device, dtype=param_dtype) for _ in range(2)]

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
        inp = torch.ones(torch.Size([16, 64, 112, 112]), dtype=input_dtype, device=device)

        ref = bn(inp)
        cudnn_batch_norm_decomp = torch._decomp.get_decompositions({torch.ops.aten.cudnn_batch_norm})
        aot_fn = make_fx(bn, decomposition_table=cudnn_batch_norm_decomp)(inp)
        res = aot_fn(inp)
        for a, b in zip(ref, res):
            assert torch.allclose(a, b)

    @patch("functorch.compile.config.use_dynamic_shapes", True)
    @patch("functorch.compile.config.use_fake_tensor", True)
    @skipIfNoSympy
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
        af = aot_function(f, nop, partition_fn=partial(min_cut_rematerialization_partition, compiler="inductor"))
        out = af(inp)
        self.assertEqual(out, f(inp))

    @patch("functorch.compile.config.use_dynamic_shapes", True)
    @patch("functorch.compile.config.use_fake_tensor", True)
    @skipIfNoSympy
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
        inp = torch.randn(2, 2, requires_grad=True)

        def f(a):
            b = a[0]
            c = torch.ones_like(b, dtype=torch.bool)
            d = b.masked_fill_(c, 0)
            return d

        compiled_f = aot_function(f, nop)
        inp_ref = torch.ones(2, 2, requires_grad=True)
        inp_test = torch.ones(2, 2, requires_grad=True)

        out_ref = f(inp_ref.clone())
        out_test = compiled_f(inp_test.clone())

        self.assertEqual(out_ref, out_test)

        out_ref.sum().backward()
        out_test.sum().backward()

        self.assertEqual(inp_ref.grad, inp_test.grad)

    def test_real_weights_in_symbolic_mode(self):
        from functorch.experimental import functionalize

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = M().eval()

        inp = torch.randn(2, 5)

        gm = make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)
        self.assertEqual(gm(torch.ones(2, 5)), m(torch.ones(2, 5)))

        gm_functionalized = make_fx(functionalize(gm,), tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)
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

        with self.assertRaisesRegex(Exception, "Please convert all Tensors to FakeTensors"):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=False)(torch.randn(2, 5))

    def test_real_weights_in_symbolic_mode_with_inplace_ops(self):

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer", torch.ones(4, 5))

            def forward(self, x):
                y = self.buffer.add_(3)
                y.resize_([20])
                assert(y.shape == self.buffer.shape)
                return x.sum() + self.buffer.sum()

        m = M().eval()
        inp = torch.randn(2, 5)
        # inplace mutation on attr is not allowed
        with self.assertRaisesRegex(Exception, "Can't call metadata"):
            make_fx(m, tracing_mode="symbolic", _allow_non_fake_inputs=True)(inp)


def extract_graph(fx_g, _, graph_cell):
    graph_cell[0] = fx_g
    return fx_g


def get_ins_outs(fx_g):
    ins = []
    outs = []
    for n in fx_g.graph.nodes:
        if n.op == 'placeholder':
            ins.append(n)
        elif n.op == 'output':
            outs = tuple(n.args[0])
    return ins, outs


def get_num_ins_outs(fx_g):
    return tuple(len(i) for i in get_ins_outs(fx_g))


def get_fw_bw_graph(f, inps, partitioner=min_cut_rematerialization_partition):
    fw_graph_cell = [None]
    bw_graph_cell = [None]
    aot_function(f,
                 fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
                 bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
                 partition_fn=partitioner,
                 decompositions=default_decompositions)(*inps).sum().backward()
    return (fw_graph_cell[0], bw_graph_cell[0])


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
        res_a = ref_a.clone().detach().requires_grad_(True)
        res_b = ref_b.clone().detach().requires_grad_(True)

        def compile_fn(x, _):
            return x

        compiled_fn = compiled_function(fn, compile_fn, compile_fn, min_cut_rematerialization_partition)
        res = compiled_fn(res_a, res_b)
        res.sum().backward()
        assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)
        assert torch.allclose(ref_a.grad, res_a.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(ref_b.grad, res_b.grad, atol=1e-3, rtol=1e-3)

    def test_meta_tensor_inplace_op(self):
        # Following module results in inplace ops while tracing. The test checks
        # that the meta tensor information is stored for inplace ops.
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(3072, 768, requires_grad=True))
                self.bias = torch.nn.Parameter(torch.randn(3072, requires_grad=True))

            def forward(self, add_4):
                linear_4 = torch.nn.functional.linear(add_4, self.weight, bias=self.bias)
                gelu = torch.nn.functional.gelu(linear_4)
                return gelu

        def check_meta_tensor(fx_g, _):
            for node in fx_g.graph.nodes:
                if node.op != 'output':
                    assert 'tensor_meta' in node.meta
            return fx_g

        inp0 = torch.randn(16, 128, 768, requires_grad=True)
        inputs = [inp0, ]
        mod = MockModule().to(device="cpu")
        aot_mod = aot_module(mod, fw_compiler=check_meta_tensor)
        aot_mod(*inputs)

    def test_default_partitioner_getitem(self):
        mod = nn.LayerNorm([10])

        def f(x, mod_weight, mod_bias):
            return torch.nn.functional.layer_norm(x, [10], mod_weight, mod_bias, eps=1e-6)

        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(3, 10, requires_grad=True), mod.weight, mod.bias],
                                             partitioner=default_partition)
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 6))
        self.assertEqual(get_num_ins_outs(bw_graph), (6, 3))

    @patch("functorch.compile.config.use_dynamic_shapes", True)
    @patch("functorch.compile.config.use_fake_tensor", True)
    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    @skipIfNoSympy
    def test_min_cut_partitioner_save_shape(self):

        def f(x):
            s = x.sum(dim=1)
            return s

        inp = [torch.ones([10, 10], requires_grad=True)]
        fw_graph, bw_graph = get_fw_bw_graph(f, inp)
        _, fw_output = get_ins_outs(fw_graph)
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))
        self.assertEqual(str(fw_output[0]), "sum_1")
        # make sure we don't do the suboptimal thing of saving the bigger primals input to sum,
        # rather than saving the sizes of the primals input for use in backward expand
        self.assertEqual(str(fw_output[1]), "sym_size")
        self.assertEqual(str(fw_output[2]), "sym_size_1")

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
        fw_graph, bw_graph = get_fw_bw_graph(f, inp)
        self.assertEqual(get_num_ins_outs(fw_graph), (3, 5))
        self.assertEqual(get_num_ins_outs(bw_graph), (5, 3))
        _, outs = get_ins_outs(fw_graph)
        self.assertTrue(all([is_sym_node(n) for n in outs[1:]]))

    @patch("functorch.compile.config.use_dynamic_shapes", True)
    @patch("functorch.compile.config.use_fake_tensor", True)
    @skipIfNoSympy
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
            mm2 = torch.mm(mm, a.view(mm.size(1), a.size(0)))  # this saves 4 new ints for backward. why?
            # and what do i have to do to make it save a tensor for backward?
            return cat, sb, c, mm2

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=default_partition,
            decompositions=default_decompositions)(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]

        # 12 outs because:
        # - 5 original outputs -> 4 graph outputs (the 3rd output is an input alias, gets moved outside)
        # - 8 saved outputs for backward: 5 tensors, 3 symints
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 12))
        self.assertEqual(get_num_ins_outs(bw_graph), (12, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            #
            # TODO(whc)- are the saved-tensors/saved-symints correct here?
            # i just made the test pass based on what default partition did
            # Of the 5 original forward outputs, the 4th (c) is an input,
            # which won't show up in the compiled forward graph
            [False, True, True, False] + [False] * 4 + [True] * 4,
            [is_sym_node(n) for n in fw_graph_out_nodes]
        )

        real_outs = f(*inp)
        self.assertEqual(compiled_outs, real_outs)
        self.assertTrue(isinstance(real_outs[1], torch.Size))

        # TODO(whc) we should learn to return torch.Sizes
        self.assertFalse(isinstance(compiled_outs[1], torch.Size))

    @patch("functorch.compile.config.use_dynamic_shapes", True)
    @patch("functorch.compile.config.use_fake_tensor", True)
    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    @skipIfNoSympy
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
            mm2 = torch.mm(mm, a.view(mm.size(1), a.size(0)))  # this saves 4 new ints for backward. why?
            # and what do i have to do to make it save a tensor for backward?
            return cat, sb, c, mm2

        fw_graph_cell = [None]
        bw_graph_cell = [None]
        compiled_outs = aot_function(
            f,
            fw_compiler=partial(extract_graph, graph_cell=fw_graph_cell),
            bw_compiler=partial(extract_graph, graph_cell=bw_graph_cell),
            partition_fn=min_cut_rematerialization_partition,
            decompositions=default_decompositions)(*inp)
        fw_graph = fw_graph_cell[0]
        (compiled_outs[0].sum() + compiled_outs[2].sum()).backward()
        bw_graph = bw_graph_cell[0]

        self.assertEqual(get_num_ins_outs(fw_graph), (4, 12))
        self.assertEqual(get_num_ins_outs(bw_graph), (12, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            # then 4 tensors (transposes of matricies used for mm) are saved
            # finally 4 symints are saved
            [False, True, True, False] + [False] * 4 + [True] * 4,
            [is_sym_node(n) for n in fw_graph_out_nodes]
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

        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True) for _ in range(4)])
        self.assertEqual(get_num_ins_outs(fw_graph), (4, 2))
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 4))

        def f(x):
            return torch.mm(x, torch.ones(x.shape)).tanh().tanh()
        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(5, 5, requires_grad=True)])
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))

        ins, outs = get_ins_outs(fw_graph)
        self.assertEqual(outs[1].target, torch.ops.aten.mm.default)

    @unittest.skipIf(not USE_NETWORKX, "networkx not available")
    def test_min_cut_partitioner_recomputable_ops(self):
        def f(x):
            return x * x * x

        recomputable_ops = []
        partition_fn = partial(min_cut_rematerialization_partition, recomputable_ops=recomputable_ops)

        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)], partition_fn)
        # Expected forward graph:
        # opcode         name       target           args                        kwargs
        # -------------  ---------  ---------------  --------------------------  --------
        # placeholder    primals_1  primals_1        ()                          {}
        # call_function  mul        aten.mul.Tensor  (primals_1, primals_1)      {}
        # call_function  mul_1      aten.mul.Tensor  (mul, primals_1)            {}
        # output         output     output           ([mul_1, primals_1, mul],)  {}
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 3))
        # Expected backward graph:
        # opcode         name        target           args                     kwargs
        # -------------  ----------  ---------------  -----------------------  --------
        # placeholder    primals_1   primals_1        ()                       {}
        # placeholder    mul         mul              ()                       {}
        # placeholder    tangents_1  tangents_1       ()                       {}
        # call_function  mul_2       aten.mul.Tensor  (tangents_1, mul)        {}
        # call_function  mul_3       aten.mul.Tensor  (tangents_1, primals_1)  {}
        # call_function  mul_4       aten.mul.Tensor  (mul_3, primals_1)       {}
        # call_function  add         aten.add.Tensor  (mul_2, mul_4)           {}
        # call_function  add_1       aten.add.Tensor  (add, mul_4)             {}
        # output         output      output           ([add_1],)               {}
        self.assertEqual(get_num_ins_outs(bw_graph), (3, 1))

        recomputable_ops = [torch.ops.aten.mul]
        partition_fn = partial(min_cut_rematerialization_partition, recomputable_ops=recomputable_ops)
        fw_graph, bw_graph = get_fw_bw_graph(f, [torch.randn(3, requires_grad=True)], partition_fn)
        # Expected forward graph:
        # opcode         name       target           args                    kwargs
        # -------------  ---------  ---------------  ----------------------  --------
        # placeholder    primals_1  primals_1        ()                      {}
        # call_function  mul        aten.mul.Tensor  (primals_1, primals_1)  {}
        # call_function  mul_1      aten.mul.Tensor  (mul, primals_1)        {}
        # output         output     output           ([mul_1, primals_1],)   {}
        self.assertEqual(get_num_ins_outs(fw_graph), (1, 2))
        # Expected backward graph:
        # opcode         name        target           args                     kwargs
        # -------------  ----------  ---------------  -----------------------  --------
        # placeholder    primals_1   primals_1        ()                       {}
        # placeholder    tangents_1  tangents_1       ()                       {}
        # call_function  mul         aten.mul.Tensor  (primals_1, primals_1)   {} # RECOMPUTED
        # call_function  mul_2       aten.mul.Tensor  (tangents_1, mul)        {}
        # call_function  mul_3       aten.mul.Tensor  (tangents_1, primals_1)  {}
        # call_function  mul_4       aten.mul.Tensor  (mul_3, primals_1)       {}
        # call_function  add         aten.add.Tensor  (mul_2, mul_4)           {}
        # call_function  add_1       aten.add.Tensor  (add, mul_4)             {}
        # output         output      output           ([add_1],)               {}
        self.assertEqual(get_num_ins_outs(bw_graph), (2, 1))

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

class TestAOTModuleSimplified(AOTTestCase):
    def test_aot_module_simplified(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                return (self.linear(x) + y, )

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
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                return (self.linear(x) + y, )

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

        self.assertExpectedInline(shape_env.format_guards(), """\
 - Eq(s1, 20)
 - Eq(s2, 30)""")

        assert torch.allclose(ref[0], res[0])
        assert torch.allclose(inputs[0].grad, cloned_inputs[0].grad)
        assert torch.allclose(inputs[1].grad, cloned_inputs[1].grad)


    def test_aot_module_simplified_preserves_stack_trace(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(20, 30)

            def forward(self, x, y):
                z = self.linear(x)
                z = z + y
                z = z.relu()
                return (z, )

        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(MockModule())
        mod = torch.fx.GraphModule(tracer.root, graph)

        for node in mod.graph.nodes:
            if node.op == 'output':
                continue
            self.assertTrue(node.stack_trace is not None)
            assert 'test_aotdispatch.py' in node.stack_trace

        def assert_compiler(gm: torch.fx.GraphModule, _):
            for node in gm.graph.nodes:
                if node.op == 'output' or node.op == 'placeholder':
                    continue
                self.assertTrue(node.stack_trace is not None)
                assert 'test_aotdispatch.py' in node.stack_trace
            return gm.forward  # return a python callable

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]

        compiled_f = aot_module_simplified(mod, inputs, fw_compiler=assert_compiler, bw_compiler=assert_compiler)
        res = compiled_f(*inputs)
        res[0].sum().backward()

    def test_aot_module_simplified_fake_tensor_gm_raises(self):
        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        real_x = torch.randn(4, requires_grad=True)
        fake_x = fake_mode.from_tensor(real_x)
        real_z = torch.randn(4)
        fake_z = fake_mode.from_tensor(real_z)

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # Accessing a free variable fake tensor will look like a
                # constant to make_fx, and result in the tensor being traced
                # into the graph, which is an error condition.  Make sure we
                # report adequately in this case.
                return (x + fake_z, )

        with self.assertRaisesRegex(
            AssertionError, "Unexpected fake buffer"
        ):
            aot_module_simplified(MockModule(), (fake_x,), nop)


# entries in here don't work and need to be fixed.
# Each one of these is a bug (or needs to be investigated)
aot_autograd_failures = {
    # data-dependent control flow
    xfail('cov'),
    xfail('istft'),
    xfail('nn.functional.gaussian_nll_loss'),
    xfail('tensor_split'),
    xfail('corrcoef'),
    xfail('quantile'),
    xfail('nanquantile'),
    xfail('narrow'),
    xfail('index_reduce'),
    xfail('istft'),
    xfail('linalg.eig'),
    xfail('scatter_reduce', 'prod'),

    skip('as_strided_scatter'),
    skip('as_strided', 'partial_views'),  # flaky

    # Too annoying to generate random inputs
    xfail('cholesky'),
    xfail('linalg.cholesky'),

    # Given input size: (s0xs1x2). Calculated output size: ...
    skip('max_pool2d_with_indices_backward'),

    # Worked with real but not with fake
    xfail('cholesky_inverse'),
    xfail('segment_reduce', 'lengths'),
    xfail('nn.functional.embedding_bag'),
    skip('nn.functional.nll_loss', ''),  # UBSAN failure!

    # Misc
    xfail('to_sparse'),
    xfail('corrcoef'),
    xfail('cov'),
    xfail('chalf'),  # RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'
    xfail('sparse.sampled_addmm'),
    skip('nn.functional.binary_cross_entropy_with_logits'),  # seems to fail sometimes?
    skip('nn.functional.margin_ranking_loss'),  # seems flaky
    skip('linalg.lu_solve'),  # flaky
    skip('linalg.householder_product'),  # flaky
    decorate('matmul', decorator=unittest.skipIf(IS_ARM64, 'flaky')),
    decorate('__rmatmul__', decorator=unittest.skipIf(IS_ARM64, 'flaky')),
}

symbolic_aot_autograd_failures = {
    xfail('addmv', ''),  # aten.addmv.default - couldn't find symbolic meta function/decomposition
    xfail('amax', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('amin', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('baddbmm', ''),  # aten.baddbmm.default - couldn't find symbolic meta function/decomposition
    xfail('block_diag', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('cartesian_prod', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('cdist', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('cholesky_inverse', ''),  # could not find kernel
    xfail('cholesky_solve', ''),  # could not find kernel
    xfail('column_stack', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('combinations', ''),  # aten.masked_select.default
    xfail('cross', ''),  # aten.linalg_cross.default - couldn't find symbolic meta function/decomposition
    xfail('cummax', ''),  # aten.cummax.default - couldn't find symbolic meta function/decomposition
    xfail('cummin', ''),  # aten.cummin.default - couldn't find symbolic meta function/decomposition
    xfail('cumprod', ''),  # aten.cumprod.default - couldn't find symbolic meta function/decomposition
    xfail('cumsum', ''),  # aten.cumsum.default - couldn't find symbolic meta function/decomposition
    xfail('cumulative_trapezoid', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('diff', ''),  # aten.zeros_like.default - couldn't find symbolic meta function/decomposition
    xfail('digamma', ''),  # aten.polygamma.default - couldn't find symbolic meta function/decomposition
    xfail('dsplit', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.fft2', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.fft', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.fftn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.fftshift', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.hfft2', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.hfft', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.hfftn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.ifft2', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.ifft', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.ifftn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.ifftshift', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.ihfft2', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.ihfft', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.ihfftn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.irfft2', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.irfft', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.irfftn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.rfft2', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.rfft', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('fft.rfftn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('frexp', ''),  # aten.frexp.Tensor - couldn't find symbolic meta function/decomposition
    xfail('gradient', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('hsplit', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('i0', ''),  # aten.i0.default - couldn't find symbolic meta function/decomposition
    xfail('index_put', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('inner', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('kron', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('kthvalue', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.cholesky_ex', ''),  # could not find kernel for aten.linalg_solve_triangular.default
    xfail('linalg.cond', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('linalg.cross', ''),  # aten.linalg_cross.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.det', ''),  # aten._linalg_det.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.det', 'singular'),  # aten._linalg_det.default - couldn't find symbolic meta function/deco...
    xfail('linalg.eigh', ''),  # aten._linalg_eigh.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.eigvals', ''),  # aten.linalg_eig.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.eigvalsh', ''),  # aten._linalg_eigh.default - couldn't find symbolic meta function/decompo...
    xfail('linalg.householder_product', ''),  # aten.linalg_householder_product.default - couldn't find symbo...
    xfail('linalg.inv', ''),  # aten.linalg_inv_ex.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.inv_ex', ''),  # aten.linalg_inv_ex.default - couldn't find symbolic meta function/decompos...
    xfail('linalg.lstsq', ''),  # aten.linalg_lstsq.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.lstsq', 'grad_oriented'),  # aten.linalg_lstsq.default - couldn't find symbolic meta funct...
    xfail('linalg.lu', ''),  # aten.linalg_lu.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.lu_factor', ''),  # aten.linalg_lu_factor_ex.default - couldn't find symbolic meta function...
    xfail('linalg.lu_factor_ex', ''),  # aten.linalg_lu_factor_ex.default - couldn't find symbolic meta funct...
    xfail('linalg.lu_solve', ''),  # aten.linalg_lu_solve.default - couldn't find symbolic meta function/deco...
    xfail('linalg.matrix_norm', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.matrix_power', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.multi_dot', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.norm', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.norm', 'subgradients_at_zero'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.pinv', ''),  # aten.linalg_pinv.atol_rtol_tensor - couldn't find symbolic meta function/dec...
    xfail('linalg.pinv', 'hermitian'),  # aten.linalg_pinv.atol_rtol_tensor - couldn't find symbolic meta fu...
    xfail('linalg.qr', ''),  # aten.linalg_qr.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.slogdet', ''),  # aten._linalg_slogdet.default - couldn't find symbolic meta function/decom...
    xfail('linalg.solve', ''),  # aten._linalg_solve_ex.default - couldn't find symbolic meta function/decomp...
    xfail('linalg.solve_ex', ''),  # aten._linalg_solve_ex.default - couldn't find symbolic meta function/dec...
    xfail('linalg.solve_triangular', ''),  # aten.linalg_solve_triangular.default - couldn't find symbolic me...
    xfail('linalg.svd', ''),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('linalg.svdvals', ''),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposi...
    xfail('linalg.tensorinv', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.tensorsolve', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('linalg.vander', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('logaddexp2', ''),  # aten.logaddexp2.default - couldn't find symbolic meta function/decomposition
    xfail('logaddexp', ''),  # aten.logaddexp.default - couldn't find symbolic meta function/decomposition
    xfail('logcumsumexp', ''),  # aten.logcumsumexp.default - couldn't find symbolic meta function/decomposition
    xfail('logdet', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('logsumexp', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('lu', ''),  # aten.linalg_lu_factor_ex.default - couldn't find symbolic meta function/decomposition
    xfail('lu_solve', ''),  # aten.linalg_lu_solve.default - couldn't find symbolic meta function/decomposition
    xfail('lu_unpack', ''),  # aten.lu_unpack.default - couldn't find symbolic meta function/decomposition
    xfail('masked.amax', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked.amin', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked.cumprod', ''),  # aten.cumprod.default - couldn't find symbolic meta function/decomposition
    xfail('masked.cumsum', ''),  # aten.cumsum.default - couldn't find symbolic meta function/decomposition
    xfail('masked_fill', ''),  # could not find kernel
    xfail('masked.logaddexp', ''),  # aten.logaddexp.default - couldn't find symbolic meta function/decomposi...
    xfail('masked.logsumexp', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked.prod', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked_scatter', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked_select', ''),  # aten.masked_select.default - couldn't find symbolic meta function/decompos...
    xfail('matrix_exp', ''),  # aten.linalg_matrix_exp.default - couldn't find symbolic meta function/decompo...
    xfail('median', ''),  # could not find kernel
    xfail('meshgrid', 'list_of_tensors'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('meshgrid', 'variadic_tensors'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('min', 'reduction_with_dim'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('mode', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional._scaled_dot_product_attention', ''),  # Cannot call sizes() on tensor with symbolic ...
    xfail('nn.functional.adaptive_avg_pool3d', ''),  # aten._adaptive_avg_pool3d_backward.default - couldn't ...
    xfail('nn.functional.adaptive_max_pool1d', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.adaptive_max_pool2d', ''),  # aten.adaptive_max_pool2d.default - couldn't find symbo...
    xfail('nn.functional.adaptive_max_pool3d', ''),  # argument 'output_size' (position 2...
    xfail('nn.functional.avg_pool3d', ''),  # aten.avg_pool3d.default - couldn't find symbolic meta function/...
    skip('nn.functional.batch_norm', ''),  # '0 is not tracked with proxy for <torch.fx.experimental.proxy_te..
    xfail('nn.functional.bilinear', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.binary_cross_entropy', ''),  # aten.fill_.Scalar - couldn't find symbolic meta funct...
    xfail('nn.functional.cosine_similarity', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.cross_entropy', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.ctc_loss', ''),  # aten._ctc_loss.Tensor - couldn't find symbolic meta function/deco...
    xfail('nn.functional.embedding_bag', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.fractional_max_pool2d', ''),  # rand() received an invalid combination of arguments - g...
    xfail('nn.functional.fractional_max_pool3d', ''),  # rand() received an invalid combination of arguments - g...
    xfail('nn.functional.grid_sample', ''),  # RuntimeError: aten.grid_sampler_3d.default - couldn't find sym ...
    xfail('nn.functional.group_norm', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'area'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'bicubic'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'linear'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'nearest'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'trilinear'),  # Cannot call sizes() on tensor with symbolic sizes/st...
    xfail('nn.functional.max_pool1d', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.max_pool2d', ''),  # aten.max_pool2d_with_indices_backward.default - couldn't find s...
    xfail('nn.functional.max_pool3d', ''),  # aten.max_pool3d_with_indices.default - couldn't find symbolic m...
    xfail('nn.functional.max_unpool1d', ''),  # aten.max_unpool2d.default - couldn't find symbolic meta funct...
    xfail('nn.functional.max_unpool1d', 'grad'),  # aten.max_unpool2d.default - couldn't find symbolic meta ...
    xfail('nn.functional.max_unpool2d', ''),  # aten.max_unpool2d.default - couldn't find symbolic meta funct...
    xfail('nn.functional.max_unpool2d', 'grad'),  # aten.max_unpool2d.default - couldn't find symbolic meta ...
    xfail('nn.functional.max_unpool3d', ''),  # aten.max_unpool3d.default - couldn't find symbolic meta funct...
    xfail('nn.functional.max_unpool3d', 'grad'),  # aten.max_unpool3d.default - couldn't find symbolic meta ...
    xfail('nn.functional.multi_margin_loss', ''),  # could not find kernel
    xfail('nn.functional.multilabel_margin_loss', ''),  # could not find kernel
    xfail('nn.functional.nll_loss', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.pad', 'reflect'),  # aten.reflection_pad1d.default - couldn't find symbolic meta fu...
    xfail('nn.functional.pad', 'replicate'),  # aten.replication_pad1d.default - couldn't find symbolic meta...
    xfail('nn.functional.pdist', ''),  # could not find kernel
    xfail('nn.functional.pixel_shuffle', ''),  # aten.pixel_shuffle.default - couldn't find symbolic meta fun...
    xfail('nn.functional.pixel_unshuffle', ''),  # aten.pixel_unshuffle.default - couldn't find symbolic meta...
    xfail('nn.functional.prelu', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.rrelu', ''),  # aten.rrelu_with_noise.default - couldn't find symbolic meta function...
    xfail('nn.functional.smooth_l1_loss', ''),  # could not find kernel
    xfail('nn.functional.unfold', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.upsample_nearest', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('norm', 'nuc'),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('normal', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('normal', 'number_mean'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('ormqr', ''),  # aten.ormqr.default - couldn't find symbolic meta function/decomposition
    xfail('pca_lowrank', ''),  # could not find kernel
    xfail('pinverse', ''),  # aten.linalg_pinv.atol_rtol_tensor - couldn't find symbolic meta function/decomp...
    xfail('polar', ''),  # could not find kernel
    xfail('polygamma', 'polygamma_n_0'),  # aten.polygamma.default - couldn't find symbolic meta function/de...
    xfail('polygamma', 'polygamma_n_1'),  # aten.polygamma.default - couldn't find symbolic meta function/de...
    xfail('polygamma', 'polygamma_n_2'),  # aten.polygamma.default - couldn't find symbolic meta function/de...
    xfail('polygamma', 'polygamma_n_3'),  # aten.polygamma.default - couldn't find symbolic meta function/de...
    xfail('polygamma', 'polygamma_n_4'),  # aten.polygamma.default - couldn't find symbolic meta function/de...
    xfail('prod', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('put', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('qr', ''),  # aten.linalg_qr.default - couldn't find symbolic meta function/decomposition
    xfail('renorm', ''),  # aten.renorm.default - couldn't find symbolic meta function/decomposition
    xfail('repeat_interleave', ''),  # aten.repeat_interleave.Te...
    xfail('reshape_as', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('roll', ''),  # narrow() received an invalid combination of arguments - got (FakeTensor, int, torch._C...
    xfail('segment_reduce', 'lengths'),  # aten.segment_reduce.default - couldn't find symbolic meta functio...
    xfail('segment_reduce', 'offsets'),  # aten.segment_reduce.default - couldn't find symbolic meta functio...
    xfail('sgn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('special.i1', ''),  # aten.i0.default - couldn't find symbolic meta function/decomposition
    xfail('special.polygamma', 'special_polygamma_n_0'),  # aten.polygamma.default - couldn't find symbolic ...
    xfail('std', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('std', 'unbiased'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('std_mean', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('std_mean', 'unbiased'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('stft', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('sum_to_size', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('svd', ''),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('svd_lowrank', ''),  # could not find kernel
    xfail('symeig', ''),  # aten.symeig.default - couldn't find symbolic meta function/decomposition
    xfail('take_along_dim', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('take', ''),  # aten.take.default - couldn't find symbolic meta function/decomposition
    xfail('tensordot', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('trace', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('trapezoid', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('trapz', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('triangular_solve', ''),  # aten.triangular_solve.default - couldn't find symbolic meta function/de...
    xfail('unflatten', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('var', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('var', 'unbiased'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('var_mean', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('var_mean', 'unbiased'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('view_as', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('vsplit', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
}

def _test_aot_autograd_forwards_backwards_helper(self, f, compiled_f, args):
    # Verify grads are equal between compiled and non-compiled versions of f.

    def call_forwards_backwards(f):
        out = wrapper_set_seed(f, args)
        if not isinstance(out, torch.Tensor):
            flat_out, _ = pytree.tree_flatten(out)
            sm = 0
            for i in flat_out:
                sm += i.sum()
            sm.backward()
        else:
            out.sum().backward()

    def reset_grads():
        def f(x):
            x.grad = None
        pytree.tree_map(f, args)

    def get_grads(args):
        return pytree.tree_map(lambda x: x.grad, args)

    try:
        reset_grads()
        call_forwards_backwards(compiled_f)
        compiled_grad = get_grads(args)

        reset_grads()
        call_forwards_backwards(f)
        orig_grad = get_grads(args)
        self.assertEqual(orig_grad, compiled_grad)

        def create_new_arg(x):
            if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
                return x.detach().uniform_(0, 1).requires_grad_(x.requires_grad)
            return x

        args = pytree.tree_map(create_new_arg, args)

        reset_grads()
        call_forwards_backwards(compiled_f)
        compiled_grad = get_grads(args)

        reset_grads()
        call_forwards_backwards(f)
        orig_grad = get_grads(args)
        self.assertEqual(orig_grad, compiled_grad)
    except DynamicOutputShapeException:
        self.skipTest("Dynamic output shape operation in trace")

def _test_aot_autograd_helper(self, device, dtype, op):
    if not op.supports_autograd:
        self.skipTest("Op does not support autograd")

    sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
    for sample_input in sample_inputs_itr:
        t_args = [sample_input.input] + list(sample_input.args)
        t_kwargs = sample_input.kwargs
        flat_args, args_spec = pytree.tree_flatten((t_args, t_kwargs))
        sentinel_val = -42
        is_tensor_spec = [sentinel_val if isinstance(arg, torch.Tensor) else arg for arg in flat_args]
        args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

        def f(args):
            cur_flat_args = list(is_tensor_spec)
            args = iter(args)
            for idx, v in enumerate(cur_flat_args):
                if v == sentinel_val:
                    cur_flat_args[idx] = next(args)
            c_args, c_kwargs = pytree.tree_unflatten(cur_flat_args, args_spec)
            return op.op(*c_args, **c_kwargs)

        compiled_f = compiled_function(f, nop, nop)
        _test_aot_autograd_forwards_backwards_helper(self, f, compiled_f, args)

def _test_aot_autograd_module_helper(self, device, dtype, training, module_info):
    module_cls = module_info.module_cls
    module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                   requires_grad=True, training=training)
    for module_input in module_inputs:
        if module_input.forward_input is None:
            continue

        args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
        m = module_cls(*args, **kwargs)
        m.to(device).to(dtype)
        m.train(training)

        # Lazy modules need to see an input first to initialize params.
        args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
        if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
            with torch.no_grad():
                m(*args, **kwargs)

        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        sentinel_val = -42
        is_tensor_spec = [sentinel_val if isinstance(arg, torch.Tensor)
                          else arg for arg in flat_args]
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
            return stateless.functional_call(m, params_and_buffers, c_args, c_kwargs)

        named_params = dict(_named_parameters(m, remove_duplicate=False))
        named_buffers = dict(_named_buffers(m, remove_duplicate=False))
        num_params_buffers = len(named_params) + len(named_buffers)
        compiled_f = aot_function(f, nop, num_params_buffers=num_params_buffers)
        params_buffers_args = [named_params, named_buffers, args]
        _test_aot_autograd_forwards_backwards_helper(self, f, compiled_f, params_buffers_args)


class TestEagerFusionOpInfo(AOTTestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestEagerFusionOpInfo', 'test_aot_autograd_exhaustive', aot_autograd_failures)
    def test_aot_autograd_exhaustive(self, device, dtype, op):
        _test_aot_autograd_helper(self, device, dtype, op)

    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipIfNoSympy
    @patch("functorch.compile.config.use_dynamic_shapes", True)
    @patch("functorch.compile.config.use_fake_tensor", True)
    @patch("functorch.compile.config.use_functionalize", True)
    @skipOps('TestEagerFusionOpInfo', 'test_aot_autograd_symbolic_exhaustive',
             aot_autograd_failures | symbolic_aot_autograd_failures)
    def test_aot_autograd_symbolic_exhaustive(self, device, dtype, op):
        _test_aot_autograd_helper(self, device, dtype, op)


aot_autograd_module_failures = set({
    torch.nn.RNN,  # See https://github.com/pytorch/pytorch/issues/90500
    torch.nn.LSTM,  # See https://github.com/pytorch/pytorch/issues/90500
    torch.nn.GRU,  # See https://github.com/pytorch/pytorch/issues/90500
    torch.nn.GaussianNLLLoss,  # RuntimeError: It appears that you're trying to get value out
                               # of a tracing tensor with aten._local_scalar_dense.default -
                               # erroring out! It's likely that this is caused by data-dependent
                               # control flow or similar.
    torch.nn.CrossEntropyLoss,  # RuntimeError: It appears that you're trying to get value out
                                # of a tracing tensor with aten._local_scalar_dense.default -
                                # erroring out! It's likely that this is caused by data-dependent
                                # control flow or similar.
})

symbolic_aot_autograd_module_failures = {
    torch.nn.TransformerEncoderLayer,  # RuntimeError: tried to get Double out of SymFloat
    torch.nn.TransformerDecoderLayer,  # RuntimeError: tried to get Double out of SymFloat
    torch.nn.RNN,  # Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode
                   # is not yet supported. Please convert all Tensors to FakeTensors first.
    torch.nn.LSTM,  # Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode
                    # is not yet supported. Please convert all Tensors to FakeTensors first.
    torch.nn.GRU,  # Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode
                   # is not yet supported. Please convert all Tensors to FakeTensors first.
    torch.nn.GaussianNLLLoss,  # NotImplementedError: local_scalar_dense/item NYI for torch.bool
    torch.nn.CrossEntropyLoss,  # Cannot call sizes() on tensor with symbolic sizes/strides
    torch.nn.Bilinear,  # Cannot call sizes() on tensor with symbolic sizes/strides
}


class TestEagerFusionModuleInfo(AOTTestCase):
    @modules(module_db, allowed_dtypes=(torch.float,))
    @decorateForModules(unittest.expectedFailure, aot_autograd_module_failures)
    def test_aot_autograd_module_exhaustive(self, device, dtype, training, module_info):
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info)

    @modules(module_db, allowed_dtypes=(torch.float,))
    @skipIfNoSympy
    @patch("functorch.compile.config.use_dynamic_shapes", True)
    @patch("functorch.compile.config.use_fake_tensor", True)
    @patch("functorch.compile.config.use_functionalize", True)
    @decorateForModules(unittest.expectedFailure,
                        aot_autograd_module_failures | symbolic_aot_autograd_module_failures)
    def test_aot_autograd_symbolic_module_exhaustive(self, device, dtype, training, module_info):
        _test_aot_autograd_module_helper(self, device, dtype, training, module_info)


only_for = ("cpu")
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)
instantiate_device_type_tests(TestEagerFusionModuleInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
