# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree
import unittest
import functools
import itertools
import warnings
import math
from typing import Callable, Type
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma, onlyOnCPUAndCUDA, onlyCPU
import types
from functools import partial, wraps

import functorch
from functorch import (
    grad, vjp, vmap, jacrev, grad_and_value,
    make_functional_deprecated_v1, make_functional_with_buffers_deprecated_v1, make_fx, nnc_jit, compiled_function
)

from torch.testing._internal.common_device_type import ops, onlyCPU
from functorch_lagging_op_db import functorch_lagging_op_db
from functorch_additional_op_db import additional_op_db
from common_utils import (
    parameterized,
    parameterized_with_device,
    instantiate_parameterized_methods,
    get_fallback_and_vmap_exhaustive,
    opinfo_in_dict,
    xfail,
    skipOps,
)

# NB: numpy is a testing dependency!
import numpy as np

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

    def test_make_fx_jvp(self, device):
        def f(x):
            return torch.sin(x).sum()

        primals = torch.randn(3)
        _, vjp_fn = vjp(f, primals)
        cotangent = torch.randn(())
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    def test_nnc_jit(self, device):
        def f(x):
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(3)
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_jit_warns_on_recompilation(self, device):
        def f(x):
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(3)
        jit_f(inp)
        inp2 = torch.randn(5)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            jit_f(inp2)

        self.assertEqual(len(warns), 1)
        self.assertTrue("Recompiling" in str(warns[-1].message))

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

class TestPythonKeyOperatorsOpInfo(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestPythonKeyOperatorsOpInfo', 'test_make_fx_exhaustive', {
    xfail('to_sparse'),
    xfail('rsub', 'rsub_scalar'),
    xfail('linalg.matrix_power'),
    xfail('linalg.inv'),
    xfail('linalg.cholesky'),
    xfail('linalg.eigvals'),
    xfail('nn.functional.pad', 'circular'),
    })
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
            except:
                continue
            new_out = new_f(args, kwargs)
            self.assertEqual(new_out, old_out)
            pass

class TestEagerFusionOpInfo(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    # entries in here need don't work and need to be fixed.
    # Each one of these is a bug (or needs to be investigated)
    @skipOps('TestEagerFusionOpInfo', 'test_eager_compilation_exhaustive', {
        xfail('__rmatmul__'),
        xfail('expand_as'),
        xfail('fmod', ''),
        xfail('remainder', ''),
        xfail('linalg.cholesky'),
        xfail('linalg.cond'),
        xfail('linalg.det'),
        xfail('linalg.inv'),
        xfail('matmul'),
        xfail('nn.functional.gelu'),
        xfail('nn.functional.linear'),
        xfail('polar'),
        xfail('reshape_as'),
        xfail('special.zeta', 'grad'),
        xfail('to_sparse'),
        xfail('view_as'),
        xfail('addcdiv'),
        xfail('atanh'),
        xfail('addcdiv'),
        xfail('atanh'),
        xfail('cholesky'),
        xfail('cumulative_trapezoid'),
        xfail('diag_embed'),
        xfail('linalg.householder_product'),
        xfail('logit'),
        xfail('matrix_exp'),
        xfail('max', 'reduction_no_dim'),
        xfail('min', 'reduction_no_dim'),
        xfail('trapezoid'),
        xfail('trapz'),
    })
    def test_eager_compilation_exhaustive(self, device, dtype, op):

        def f(args, kwargs):
            return op.op(*args, **kwargs)
        if not op.supports_autograd:
            return
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=True)
        new_f = None
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

            compiled_f = compiled_function(f, lambda x,_: x, lambda x,_: x)

            reset_grads()
            compiled_f(args, kwargs).sum().backward()
            compiled_grad = get_grads(args)

            reset_grads()
            f(args, kwargs).sum().backward()
            orig_grad = get_grads(args)
            self.assertEqual(orig_grad, compiled_grad)

            args = pytree.tree_map(lambda x: x.detach().uniform_(0, 1), args)
            for arg in args:
                arg.requires_grad = True

            reset_grads()
            compiled_f(args, kwargs).sum().backward()
            compiled_grad = get_grads(args)

            reset_grads()
            f(args, kwargs).sum().backward()
            orig_grad = get_grads(args)
            self.assertEqual(orig_grad, compiled_grad)



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
