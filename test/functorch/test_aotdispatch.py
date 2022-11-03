# Owner(s): ["module: functorch"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch
from torch.testing._internal.common_utils import TestCase, run_tests, IS_ARM64, IS_WINDOWS
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import unittest
import warnings
import itertools
from functools import partial
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db, wrapper_set_seed
from functorch import (
    grad, vjp, vmap, jacrev,
    make_fx
)
from functorch._src.aot_autograd import aot_module_simplified
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
)
from torch._subclasses.fake_tensor import DynamicOutputShapeException
from torch.fx.experimental.proxy_tensor import is_sym_node

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


def _outs_and_grads(fn, inps):
    outs = fn(*inps)
    for out in pytree.tree_flatten(outs)[0]:
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward(retain_graph=True)
    grads = [inp.grad for inp in pytree.tree_flatten(inps)[0]]
    for inp in pytree.tree_flatten(inps)[0]:
        inp.grad = None
    return outs, grads


class TestAOTAutograd(AOTTestCase):
    def verify_aot_autograd(self, f, inp):
        if isinstance(f, nn.Module):
            compiled_f = aot_module(f, nop)
        else:
            compiled_f = aot_function(f, nop)
        ref_out, ref_grad = _outs_and_grads(f, inp)
        test_out, test_grad = _outs_and_grads(compiled_f, inp)
        self.assertEqual(ref_out, test_out)
        self.assertEqual(ref_grad, test_grad)

        if isinstance(ref_out, torch.Tensor):
            self.assertTrue(isinstance(test_out, torch.Tensor))
            ref_out, test_out = [ref_out], [test_out]
        for ref_o, test_o in zip(ref_out, test_out):
            if isinstance(ref_o, torch.Tensor):
                self.assertEqual(ref_o.requires_grad, test_o.requires_grad)

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

    def test_no_grad_input_output(self):
        def f(a, b):
            return a.cos(), b.cos(), a * b

        inp_thunks = [lambda: torch.randn(5, requires_grad=True), lambda: torch.randn(5, requires_grad=False)]
        for inps in itertools.product(inp_thunks, repeat=2):
            inps = [i() for i in inps]
            self.verify_aot_autograd(f, inps)

    def test_some_outputs_dont_require_grad(self):
        def f(a, b):
            return a.detach(), b
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
        inp = [{'a': torch.randn(3, requires_grad=True), 'b': torch.randn(3, requires_grad=True)}]
        self.verify_aot_autograd(f, inp)

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

    def test_compilation_context(self):
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
        self.assertEqual(count, [(['forward'], 4), (['inference'], 4), (['backward'], 8)])

    def test_dupe_arg(self):
        def f(x, y):
            return x + y

        x = torch.randn(3, 3, requires_grad=True)
        self.verify_aot_autograd(f, [x, x])

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
            def forward(ctx, grad_output):
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

        self.assertEqual(get_num_ins_outs(fw_graph), (4, 13))
        self.assertEqual(get_num_ins_outs(bw_graph), (13, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            #
            # TODO(whc)- are the saved-tensors/saved-symints correct here?
            # i just made the test pass based on what default partition did
            [False, True, True, False, False] + [False] * 5 + [True] * 3,
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

        self.assertEqual(get_num_ins_outs(fw_graph), (4, 13))
        self.assertEqual(get_num_ins_outs(bw_graph), (13, 4))
        _, fw_graph_out_nodes = get_ins_outs(fw_graph)
        self.assertEqual(
            # fw outputs include b.size() which expands to 2 symints,
            # then 4 tensors (transposes of matricies used for mm) are saved
            # finally 4 symints are saved
            [False, True, True, False, False] + [False] * 4 + [True] * 4,
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

        aot_mod = aot_module_simplified(mod, nop)
        aot_mod.zero_grad()
        res = aot_mod(*cloned_inputs)
        res[0].sum().backward()

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

        aot_mod = aot_module_simplified(mod, fw_compiler=assert_compiler, bw_compiler=assert_compiler)

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]
        res = aot_mod(*inputs)
        res[0].sum().backward()

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

    # non-deterministic
    skip('as_strided_scatter'),

    # Too annoying to generate random inputs
    xfail('cholesky'),
    xfail('linalg.cholesky'),

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
    xfail('__rmatmul__', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('addmv', ''),  # aten.addmv.default - couldn't find symbolic meta function/decomposition
    xfail('addr', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('amax', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('amin', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('baddbmm', ''),  # aten.baddbmm.default - couldn't find symbolic meta function/decomposition
    xfail('bernoulli', ''),  # aten.bernoulli.default - couldn't find symbolic meta function/decomposition
    xfail('block_diag', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('cartesian_prod', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('cdouble'),  # RuntimeError: aten.view_as_real.default - couldn't find symbolic meta function/decomposition
    xfail('cfloat'),  # RuntimeError: aten.view_as_real.default - couldn't find symbolic meta function/decomposition
    xfail('cdist', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('cholesky_inverse', ''),  # could not find kernel
    xfail('cholesky_solve', ''),  # could not find kernel
    xfail('column_stack', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('combinations', ''),  # aten.masked_select.default
    xfail('complex', ''),  # aten.view_as_real.default - couldn't find symbolic meta function/decomposition
    xfail('cross', ''),  # aten.linalg_cross.default - couldn't find symbolic meta function/decomposition
    xfail('cummax', ''),  # aten.cummax.default - couldn't find symbolic meta function/decomposition
    xfail('cummin', ''),  # aten.cummin.default - couldn't find symbolic meta function/decomposition
    xfail('cumprod', ''),  # aten.cumprod.default - couldn't find symbolic meta function/decomposition
    xfail('cumsum', ''),  # aten.cumsum.default - couldn't find symbolic meta function/decomposition
    xfail('cumulative_trapezoid', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('deg2rad', ''),  # aten.deg2rad.default - couldn't find symbolic meta function/decomposition
    xfail('diff', ''),  # aten.zeros_like.default - couldn't find symbolic meta function/decomposition
    xfail('digamma', ''),  # aten.polygamma.default - couldn't find symbolic meta function/decomposition
    xfail('dist', ''),  # aten.dist.default - couldn't find symbolic meta function/decomposition
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
    xfail('linalg.cholesky_ex', ''),  # aten.linalg_cholesky_ex.default - couldn't find symbolic meta functio...
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
    xfail('linalg.vector_norm', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
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
    xfail('masked.log_softmax', ''),  # argument 'size' (position 2) must be tuple of ints, not ...
    xfail('masked.logaddexp', ''),  # aten.logaddexp.default - couldn't find symbolic meta function/decomposi...
    xfail('masked.logsumexp', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked.mean', ''),  # ones() received an invalid combination of arguments - got (torch.Size, device=t...
    xfail('masked.median', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked.norm', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked.prod', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked_scatter', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked_select', ''),  # aten.masked_select.default - couldn't find symbolic meta function/decompos...
    xfail('masked.softmax', ''),  # argument 'size' (position 2) must be tuple of ints, not torc...
    xfail('masked.softmin', ''),  # argument 'size' (position 2) must be tuple of ints, not torc...
    xfail('masked.std', ''),  # ones() received an invalid combination of arguments - got (torch.Size, device=to...
    xfail('masked.sum', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('masked.var', ''),  # ones() received an invalid combination of arguments - got (torch.Size, device=to...
    xfail('matmul', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('matrix_exp', ''),  # aten.linalg_matrix_exp.default - couldn't find symbolic meta function/decompo...
    xfail('median', ''),  # could not find kernel
    xfail('nan_to_num', ''),  # Cannot cast FakeTensor(FakeTensor(..., device='meta', size=()), cpu) to number
    xfail('meshgrid', 'list_of_tensors'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('meshgrid', 'variadic_tensors'),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('min', 'reduction_with_dim'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('mode', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('mv', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('mvlgamma', 'mvlgamma_p_1'),  # aten.digamma_.default - couldn't find symbolic meta function/decom...
    xfail('mvlgamma', 'mvlgamma_p_3'),  # aten.digamma_.default - couldn't find symbolic meta function/decom...
    xfail('mvlgamma', 'mvlgamma_p_5'),  # aten.digamma_.default - couldn't find symbolic meta function/decom...

    # Deleting this in a followup
    xfail('nn.functional.feature_alpha_dropout', 'with_train'),
    xfail('nn.functional.pad', 'circular'),
    xfail('nn.functional.poisson_nll_loss', ''),

    xfail('nn.functional._scaled_dot_product_attention', ''),  # Cannot call sizes() on tensor with symbolic ...
    xfail('nn.functional.adaptive_avg_pool3d', ''),  # aten._adaptive_avg_pool3d_backward.default - couldn't ...
    xfail('nn.functional.adaptive_max_pool1d', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.adaptive_max_pool2d', ''),  # aten.adaptive_max_pool2d.default - couldn't find symbo...
    xfail('nn.functional.adaptive_max_pool3d', ''),  # argument 'output_size' (position 2...
    xfail('nn.functional.avg_pool3d', ''),  # aten.avg_pool3d.default - couldn't find symbolic meta function/...
    skip('nn.functional.batch_norm', ''),  # '0 is not tracked with proxy for <torch.fx.experimental.proxy_te..
    xfail('nn.functional.bilinear', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.binary_cross_entropy', ''),  # aten.fill_.Scalar - couldn't find symbolic meta funct...
    xfail('nn.functional.cosine_embedding_loss', ''),  # Cannot call sizes() on tensor with symbolic sizes/st...
    xfail('nn.functional.cosine_similarity', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.cross_entropy', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.ctc_loss', ''),  # aten._ctc_loss.Tensor - couldn't find symbolic meta function/deco...
    xfail('nn.functional.embedding_bag', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.fractional_max_pool2d', ''),  # rand() received an invalid combination of arguments - g...
    xfail('nn.functional.fractional_max_pool3d', ''),  # rand() received an invalid combination of arguments - g...
    xfail('nn.functional.grid_sample', ''),  # prims::arange() Expected a value of type 'number' for argument...
    xfail('nn.functional.group_norm', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.hinge_embedding_loss', ''),  # aten.zeros_like.default - couldn't find symbolic meta...
    xfail('nn.functional.huber_loss', ''),  # Unable to cast Python instance to C++ type (#define PYBIND11_DE...
    xfail('nn.functional.interpolate', 'area'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'bicubic'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'linear'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'nearest'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.interpolate', 'trilinear'),  # Cannot call sizes() on tensor with symbolic sizes/st...
    xfail('nn.functional.max_pool1d', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
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
    xfail('nn.functional.normalize', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.pad', 'reflect'),  # aten.reflection_pad1d.default - couldn't find symbolic meta fu...
    xfail('nn.functional.pad', 'replicate'),  # aten.replication_pad1d.default - couldn't find symbolic meta...
    xfail('nn.functional.pairwise_distance', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.pdist', ''),  # could not find kernel
    xfail('nn.functional.pixel_shuffle', ''),  # aten.pixel_shuffle.default - couldn't find symbolic meta fun...
    xfail('nn.functional.pixel_unshuffle', ''),  # aten.pixel_unshuffle.default - couldn't find symbolic meta...
    xfail('nn.functional.prelu', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('nn.functional.rrelu', ''),  # aten.rrelu_with_noise.default - couldn't find symbolic meta function...
    xfail('nn.functional.smooth_l1_loss', ''),  # could not find kernel
    xfail('nn.functional.upsample_nearest', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('norm', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('norm', 'nuc'),  # aten._linalg_svd.default - couldn't find symbolic meta function/decomposition
    xfail('normal', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('normal', 'number_mean'),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('ormqr', ''),  # aten.ormqr.default - couldn't find symbolic meta function/decomposition
    xfail('outer', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
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
    xfail('rad2deg', ''),  # aten.rad2deg.default - couldn't find symbolic meta function/decomposition
    xfail('renorm', ''),  # aten.renorm.default - couldn't find symbolic meta function/decomposition
    xfail('repeat_interleave', ''),  # aten.repeat_interleave.Te...
    xfail('reshape_as', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('roll', ''),  # narrow() received an invalid combination of arguments - got (FakeTensor, int, torch._C...
    xfail('round', ''),  # aten.round.default - couldn't find symbolic meta function/decomposition
    xfail('round', 'decimals_0'),  # aten.round.decimals - couldn't find symbolic meta function/decomposition
    xfail('round', 'decimals_3'),  # aten.round.decimals - couldn't find symbolic meta function/decomposition
    xfail('round', 'decimals_neg_3'),  # aten.round.decimals - couldn't find symbolic meta function/decompos...
    xfail('scatter_reduce', 'amax'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decom...
    xfail('scatter_reduce', 'amin'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decom...
    xfail('scatter_reduce', 'mean'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decom...
    xfail('scatter_reduce', 'sum'),  # aten.scatter_reduce.two - couldn't find symbolic meta function/decomp...
    xfail('segment_reduce', 'lengths'),  # aten.segment_reduce.default - couldn't find symbolic meta functio...
    xfail('segment_reduce', 'offsets'),  # aten.segment_reduce.default - couldn't find symbolic meta functio...
    xfail('sgn', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('sort', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('special.i1', ''),  # aten.i0.default - couldn't find symbolic meta function/decomposition
    xfail('special.polygamma', 'special_polygamma_n_0'),  # aten.polygamma.default - couldn't find symbolic ...
    xfail('std', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('std_mean', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
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
    xfail('var_mean', ''),  # Cannot call numel() on tensor with symbolic sizes/strides
    xfail('view_as_complex', ''),  # aten.view_as_complex.default - couldn't find symbolic meta function/deco...
    xfail('view_as', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
    xfail('vsplit', ''),  # Cannot call sizes() on tensor with symbolic sizes/strides
}

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

        def call_forwards_backwards(f):
            out = wrapper_set_seed(f, args)
            if isinstance(out, tuple):
                sm = 0
                for i in out:
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

        compiled_f = compiled_function(f, nop, nop)

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

only_for = ("cpu")
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
