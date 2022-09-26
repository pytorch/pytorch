# Owner(s): ["module: functorch"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import unittest
import warnings
import itertools
from functools import partial
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import op_db
from functorch import (
    grad, vjp, vmap, jacrev,
    make_fx
)
from functorch._src.aot_autograd import aot_module_simplified
from functorch.compile import (
    nnc_jit, compiled_function, compiled_module,
    min_cut_rematerialization_partition, aot_function, aot_module,
    nop, num_of_recompilations, default_partition, default_decompositions,
    memory_efficient_fusion, clear_compile_cache, get_aot_compilation_context
)
from torch._decomp import decomposition_table

from torch.testing._internal.common_device_type import ops
from functorch_additional_op_db import additional_op_db
from common_utils import (
    xfail,
    skip,
    skipOps,
)

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

# NB: numpy is a testing dependency!

class AOTTestCase(TestCase):
    def setUp(self):
        super().setUp()
        # NB: We cache on function id, which is unreliable
        # Can fix by using weakrefs, but not sure if it matters
        clear_compile_cache()

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

        start_recompilations = num_of_recompilations()
        f = aot_function(foo, nop, get_graph_size)
        with torch.set_grad_enabled(False):
            f(*inps)
        self.assertIsNone(graph_size)

        with torch.set_grad_enabled(True):
            out = f(*inps)
            self.assertIsNone(graph_size)
            out.sum().backward()
            self.assertTrue(graph_size > 2)
        self.assertEqual(num_of_recompilations() - start_recompilations, 2)

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



class TestEagerFusionOpInfo(AOTTestCase):
    @ops(op_db + additional_op_db, allowed_dtypes=(torch.float,))
    # entries in here need don't work and need to be fixed.
    # Each one of these is a bug (or needs to be investigated)
    @skipOps('TestEagerFusionOpInfo', 'test_aot_autograd_exhaustive', {
        xfail('linalg.cholesky'),
        skip('msort'),
        xfail('nn.functional.dropout'),
        xfail('to_sparse'),
        xfail('addcdiv'),
        xfail('cholesky'),
        xfail('cumulative_trapezoid'),
        xfail('diag_embed'),
        xfail('logit'),
        xfail('trapezoid'),
        xfail('trapz'),
        xfail('corrcoef'),
        xfail('cov'),
        xfail('chalf'),  # RuntimeError: "sum_cpu" not implemented for 'ComplexHalf'
        skip('nn.functional.binary_cross_entropy_with_logits'),  # seems to fail sometimes?
        skip('nn.functional.margin_ranking_loss'),  # seems flaky
    })
    def test_aot_autograd_exhaustive(self, device, dtype, op):
        def f(args, kwargs):
            return op.op(*args, **kwargs)
        if not op.supports_autograd:
            return
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        for sample_input in sample_inputs_itr:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            if not all([isinstance(i, torch.Tensor) and i.dtype == torch.float for i in args]):
                self.skipTest("not all inputs are float tensors")
            if not all([isinstance(i, torch.Tensor) and i.dtype == torch.float for i in kwargs.values()]):
                self.skipTest("not all inputs are float tensors")
                continue
            t = f(args, kwargs)
            if isinstance(t, tuple):
                self.skipTest("output is a tuple")
                continue

            def reset_grads():
                def f(x):
                    x.grad = None
                pytree.tree_map(f, args)

            def get_grads(args):
                return pytree.tree_map(lambda x: x.grad, args)

            compiled_f = compiled_function(f, nop, nop)

            reset_grads()
            compiled_f(args, kwargs).sum().backward()
            compiled_grad = get_grads(args)

            reset_grads()
            f(args, kwargs).sum().backward()
            orig_grad = get_grads(args)
            self.assertEqual(orig_grad, compiled_grad)

            def create_new_arg(x):
                return x.detach().uniform_(0, 1).requires_grad_(x.requires_grad)

            args = pytree.tree_map(create_new_arg, args)

            reset_grads()
            compiled_f(args, kwargs).sum().backward()
            compiled_grad = get_grads(args)

            reset_grads()
            f(args, kwargs).sum().backward()
            orig_grad = get_grads(args)
            self.assertEqual(orig_grad, compiled_grad)


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
            assert 'test_pythonkey.py' in node.stack_trace

        def assert_compiler(gm: torch.fx.GraphModule, _):
            for node in gm.graph.nodes:
                if node.op == 'output' or node.op == 'placeholder':
                    continue
                self.assertTrue(node.stack_trace is not None)
                assert 'test_pythonkey.py' in node.stack_trace
            return gm.forward  # return a python callable

        aot_mod = aot_module_simplified(mod, fw_compiler=assert_compiler, bw_compiler=assert_compiler)

        x = torch.randn(128, 20, requires_grad=True)
        y = torch.randn(128, 30, requires_grad=True)
        inputs = [x, y]
        res = aot_mod(*inputs)
        res[0].sum().backward()


only_for = ("cpu")
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
