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
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from functorch import (
    grad, vjp, vmap, jacrev,
    make_fx
)
from functorch.compile import (
    nnc_jit, compiled_function, compiled_module,
    partition_with_recompute_fwd_in_bwd, aot_function, aot_module, decomposition_table, nop,
    num_of_recompilations
)

from torch.testing._internal.common_device_type import ops
from functorch_lagging_op_db import functorch_lagging_op_db
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

# NB: numpy is a testing dependency!


class TestPythonKey(TestCase):
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

    def test_make_fx_no_decompose(self, device):
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


make_fx_failures = {
    xfail('to_sparse'),
    xfail('allclose'),
    xfail('nn.functional.dropout'),
    xfail('linalg.eigvals'),
    xfail('nn.functional.ctc_loss'),
    xfail('nn.functional.fractional_max_pool3d', device_type='cpu'),
    xfail('randn_like'),  # randomness
    xfail('rand_like'),  # randomness
    xfail('randint_like'),  # randomness
    skip('new_empty'),  # nondeterministic
    skip('empty_like'),  # nondeterministic
    skip('linalg.lstsq', 'grad_oriented'),  # flaky
}


class TestPythonKeyOperatorsOpInfo(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestPythonKeyOperatorsOpInfo', 'test_make_fx_exhaustive', make_fx_failures
             )
    def test_make_fx_exhaustive(self, device, dtype, op):

        def f(args, kwargs):
            return op.op(*args, **kwargs)
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        new_f = None
        for sample_input in sample_inputs_itr:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            t = f(args, kwargs)
            # just since pytrees with torch.return_types doesn't work
            if isinstance(t, tuple):
                self.skipTest("output is a tuple that pytree doesn't work with")

            new_f = make_fx(f)(args, kwargs)
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
                    arg.uniform_(0, 1)
            try:
                old_out = f(args, kwargs)
            except Exception:
                continue
            new_out = new_f(args, kwargs)
            self.assertEqual(new_out, old_out)
            pass


def _outs_and_grads(fn, inps):
    outs = fn(*inps)
    for out in pytree.tree_flatten(outs)[0]:
        if isinstance(out, torch.Tensor) and out.requires_grad:
            out.sum().backward(retain_graph=True)
    grads = [inp.grad for inp in pytree.tree_flatten(inps)[0]]
    for inp in pytree.tree_flatten(inps)[0]:
        inp.grad = None
    return outs, grads


class TestAOTAutograd(TestCase):
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

    @unittest.expectedFailure
    def test_grad_context(self):
        def foo(x):
            return x * 2
        inps = [torch.randn((), requires_grad=True)]
        graph_size = None

        def assert_graph_empty(fx_g, _):
            nonlocal graph_size
            graph_size = len(fx_g.graph.nodes)
            return fx_g

        start_recompilations = num_of_recompilations()
        f = aot_function(foo, nop, assert_graph_empty)
        with torch.set_grad_enabled(False):
            f(*inps)
        self.assertEqual(graph_size, 2)
        with torch.set_grad_enabled(True):
            f(*inps)
        self.assertTrue(graph_size > 2)
        self.assertEqual(num_of_recompilations - start_recompilations, 2)

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

    @unittest.expectedFailure
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

    @unittest.expectedFailure
    def test_batchnorm(self):
        mod = compiled_module(nn.BatchNorm2d(4), nop, nop)
        x = torch.ones(1, 4, 2, 2)
        mod(x).sum().backward()


class TestEagerFusionOpInfo(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    # entries in here need don't work and need to be fixed.
    # Each one of these is a bug (or needs to be investigated)
    @skipOps('TestEagerFusionOpInfo', 'test_aot_autograd_exhaustive', {
        xfail('__rmatmul__'),
        xfail('linalg.cholesky'),
        xfail('matmul'),
        xfail('msort'),
        xfail('nn.functional.linear'),
        xfail('nn.functional.dropout'),
        xfail('polar'),
        xfail('special.zeta', 'grad'),
        xfail('to_sparse'),
        xfail('addcdiv'),
        xfail('cholesky'),
        xfail('cumulative_trapezoid'),
        xfail('diag_embed'),
        xfail('linalg.householder_product'),
        xfail('logit'),
        xfail('matrix_exp'),
        xfail('trapezoid'),
        xfail('trapz'),
        xfail('trace'),
        skip('nn.functional.binary_cross_entropy_with_logits')  # seems to fail sometimes?
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


class TestPartitioning(TestCase):
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

        compiled_fn = compiled_function(fn, compile_fn, compile_fn, partition_with_recompute_fwd_in_bwd)
        res = compiled_fn(res_a, res_b)
        res.sum().backward()
        assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)
        assert torch.allclose(ref_a.grad, res_a.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(ref_b.grad, res_b.grad, atol=1e-3, rtol=1e-3)


only_for = ("cpu")
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestPythonKeyOperatorsOpInfo, globals(), only_for=only_for)
instantiate_device_type_tests(TestEagerFusionOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
