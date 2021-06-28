# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import functools
import itertools
import warnings
import math
from typing import Callable, Type
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma, onlyOnCPUAndCUDA, onlyCPU
import types
from functools import partial

import functorch
from functorch import (
    grad, vjp, vmap, jacrev, grad_and_value,
    make_functional, make_functional_with_buffers,
    functional_init, functional_init_with_buffers,
)

# NB: numpy is a testing dependency!
import numpy as np

USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)


class TestGradTransform(TestCase):
    def test_primitive(self, device):
        x = torch.randn([], device=device)
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_composite_simple(self, device):
        x = torch.randn(2, 3, 4, device=device)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

    def test_fn_with_kwargs(self, device):
        def foo(x, y):
            return (x * y).sum()

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected = grad(foo)(x, y)
        result = grad(foo)(x, y=y)
        self.assertEqual(result, expected)

    def test_composite_complicated(self, device):
        x = torch.randn(3, device=device)
        y = torch.randn(3, 5, device=device)

        def foo(x, y):
            result = x @ y
            return result.sum()

        result = grad(foo)(x, y)

        x.requires_grad_()
        out = foo(x, y)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_composite_two_ops(self, device):
        N, C = 2, 5
        y = torch.randn(N, C, device=device)
        targets = torch.randint(0, C, (N,), device=device)

        def foo(y, targets):
            return F.cross_entropy(y, targets)

        result = grad(foo)(y, targets)

        y.requires_grad_()
        expected, = torch.autograd.grad(foo(y, targets), y)

        self.assertEqual(result, expected)

    def _test_attributes(self, get_attr_lambda, device):
        x = torch.randn(2, 3, 5, dtype=torch.double, device=device)
        expected = get_attr_lambda(x)

        def foo(x):
            self.assertEqual(get_attr_lambda(x), expected)
            return x.sum()

        grad(foo)(x)

    def test_shape(self, device):
        self._test_attributes(lambda x: x.shape, device)

    def test_dtype(self, device):
        self._test_attributes(lambda x: x.dtype, device)

    def test_is_cuda(self, device):
        self._test_attributes(lambda x: x.is_cuda, device)

    def test_numel(self, device):
        self._test_attributes(lambda x: x.numel(), device)

    def test_inplace(self, device):
        x = torch.randn([], device=device)

        def foo(x):
            return x.clone().sin_()

        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_inplace_on_view(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y0.sin_()
            return y.sum()

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_view_base(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y.sin_()
            return y0

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_nesting_simple(self, device):
        x = torch.randn([], device=device)
        result = grad(grad(torch.sin))(x)
        self.assertEqual(result, -torch.sin(x))

    def test_escaped_wrappers_are_marked_as_dead(self, device):
        x = torch.randn([], device=device)
        escaped = []
        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        result = grad(foo)(x)
        self.assertEqual(functorch._C.dlevel(escaped[0]), -1)

    def test_escaped_wrappers_are_ignored(self, device):
        x = torch.randn([], device=device)
        escaped = []
        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        result = grad(foo)(x)

        something = escaped[0].sum()
        self.assertEqual(functorch._C.dlevel(something), 0)
        self.assertEqual(something, x.sin().sum())

    def test_vjp(self, device):
        x = torch.randn([], device=device)
        out, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(out, x.sin())

        v = torch.randn([], device=device)
        result, = vjp_fn(v)
        self.assertEqual(result, v * x.cos())

    def test_vjp_two_outputs(self, device):
        def f(x):
            return x, x
        result, vjp_fn = vjp(f, torch.tensor(1.))
        vjp_fn(result)

    def test_composed_with_autograd(self, device):
        x = torch.randn([], requires_grad=True, device=device)

        y = grad(torch.sin)(x)
        result, = torch.autograd.grad(y, x)
        self.assertEqual(result, -x.sin())

    def test_grad_of_vjp_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            out, vjp_fn = vjp(torch.sin, x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_vjp_of_grad_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            out, vjp_fn = vjp(grad(torch.sin), x)
            return vjp_fn(y)[0]

        result = foo(x, y)
        expected = -y * x.sin()
        self.assertEqual(result, expected)

    def test_grad_of_vjp_of_grad_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            df, vjp_fn = vjp(grad(lambda x: -torch.cos(x)), x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_views(self, device):
        x = torch.randn([], requires_grad=True, device=device)
        y = torch.randn([], requires_grad=True, device=device)

        def silly_sin(x):
            x = x.view([])
            x = x.sin()
            return x

        def foo(x, y):
            z1 = grad(silly_sin)(x)
            z2 = torch.cos(y)
            return z1 + z2

        result = foo(x, y)
        grads = torch.autograd.grad(result, [x, y])
        self.assertEqual(grads[0], -x.sin())
        self.assertEqual(grads[1], -y.sin())

    def test_view_inplace_simple(self, device):
        def foo(x):
            x = x.clone()
            x.view([]).sin_()
            return x

        x = torch.randn([], requires_grad=True, device=device)
        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_invalid_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, 'but only'):
            grad(torch.mul, argnums=-1)(x, y)
        with self.assertRaisesRegex(RuntimeError, 'but only'):
            grad(torch.mul, argnums=2)(x, y)
        with self.assertRaisesRegex(RuntimeError, 'int or Tuple'):
            grad(torch.mul, argnums=[0])(x, y)
        with self.assertRaisesRegex(RuntimeError, 'must be int'):
            grad(torch.mul, argnums=('0',))(x, y)

    def test_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        gx = grad(torch.mul, argnums=0)(x, y)
        self.assertEqual(gx, y)

        gy = grad(torch.mul, argnums=1)(x, y)
        self.assertEqual(gy, x)

        gx, = grad(torch.mul, argnums=(0,))(x, y)
        self.assertEqual(gx, y)

        gx, gy = grad(torch.mul, argnums=(0, 1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_zero_grad(self, device):
        def f(x):
            return (x['a']**2.0).sum()
        inps = ({'a':torch.randn(10, device=device) + 3, 'b':torch.randn(10, device=device)})
        grads = grad(f)(inps)
        self.assertNotEqual(grads['a'].sum(), 0.0)
        self.assertEqual(grads['b'].sum(), 0.0)

    def test_unrelated_grad(self, device):
        x = torch.tensor(1., device=device)
        y = torch.tensor(2., device=device)

        def unrelated(x):
            return y

        result = grad(unrelated)(x)
        self.assertEqual(result, torch.zeros_like(x))

    def test_unrelated_vjp(self, device):
        x = torch.tensor(1., device=device)
        y = torch.tensor(2., device=device)
        v = torch.tensor(1., device=device)

        def unrelated(x):
            return y

        out, vjp_fn = vjp(unrelated, x)
        result = vjp_fn(v)
        expected = (torch.zeros_like(x),)
        self.assertEqual(result, expected)

    def test_unrelated_vjp_multiple_inputs_outputs(self, device):
        w = torch.tensor(3., device=device)
        x = torch.tensor(4., device=device)
        y = torch.tensor(2., device=device)
        v = torch.tensor(1., device=device)

        def unrelated(w, x):
            return y, y, x

        out, vjp_fn = vjp(unrelated, w, x)
        result = vjp_fn((v, v, v))
        expected = (torch.zeros_like(x), torch.ones_like(x))
        self.assertEqual(result, expected)

    # TODO: https://github.com/zou3519/functorch/issues/12
    @onlyCPU
    def test_unrelated_hessian(self, device):
        N = 5
        M = 3
        W = torch.randn(N, M, device=device)

        def f(x):
            return W @ x

        x = torch.randn(M)
        result = jacrev(jacrev(f))(x)
        expected = torch.zeros(N, M, M, device=device)
        self.assertEqual(result, expected)

    def test_vjp_pytree_input(self, device):
        def f(x):
            return x[0] * x[1][0]

        x = torch.randn([], device=device)
        v = torch.randn([], device=device)
        out, vjp_fn = vjp(f, (x, (x, x)))
        self.assertEqual(out, x * x)
        result = vjp_fn(v)
        self.assertEqual(result, ((x * v, (x * v, 0.)),))

    def test_vjp_pytree_output(self, device):
        def f(x):
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        result, = vjp_fn((v1, (v2, v3)))
        self.assertEqual(result, v1 + v2 + v3)

    def test_vjp_pytree_error(self, device):
        def f(x):
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        with self.assertRaisesRegex(RuntimeError, 'Expected pytree structure'):
            result, = vjp_fn(((v1, (v2, v3)),))

    def test_functional_init(self, device):
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, fn, _ = functional_init(MLPClassifier, (B,))(32, 2)
        inputs = torch.randn(B, 7, 2)
        vmap(fn)(weights, (inputs,))

    def test_functional_init_with_buffers(self, device):
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.bn = nn.BatchNorm1d(self.hidden_dim, affine=True)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.bn(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, buffers, fn, _, _ = \
            functional_init_with_buffers(MLPClassifier, [B])(32, 2)
        inputs = torch.randn(B, 7, 2)
        vmap(fn)(weights, buffers, (inputs,))

    def test_advanced_indexing(self, device):
        def f(value):
            log_prob = torch.ones((), device=device)
            val = (torch.zeros(()) > 0)
            log_prob[val] = 0
            return value

        result = grad(f)(torch.randn((), device=device))
        self.assertEqual(result, torch.ones_like(result))

        def f2(value):
            value = value.clone()
            value[value > 0] = 0
            return value.sum()

        x = torch.randn(100, device=device)
        result = grad(f2)(x)
        self.assertEqual(result, (x <= 0).type_as(x))

    @unittest.expectedFailure
    def test_tensor_ctor_inside_grad(self, device):
        def foo(x):
            return x * torch.tensor(2., device=device)

        x = torch.tensor(3.14, device=device)
        functorch.grad(foo)(x)


class TestVmapOfGrad(TestCase):
    def test_per_sample_grads_inplace_view(self, device):
        def compute_loss(weight, x, t):
            x = x.mm(weight)
            y = x.squeeze_(0)
            return (y - t).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 1, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_new_zeros_materializes_tensor(self, device):
        N = 3
        C = 5

        def foo(y, x):
            result = x.new_zeros((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_new_empty_materializes_tensor(self, device):
        N = 3
        C = 5

        def foo(y, x):
            result = x.new_empty((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_per_sample_grads_simple(self, device):
        def compute_loss(weight, x, t):
            y = x @ weight
            return ((y - t) ** 2).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_per_sample_grads_embeddingnet(self, device):
        class SampleNet(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 16)
                self.fc1 = nn.Linear(16, 16)
                self.fc2 = nn.Linear(16, 2)

            def forward(self, x):
                x = self.emb(x)
                x = torch.transpose(x, -1, -2)
                x = torch.mean(x, -1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return x

            def name(self):
                return "SampleNet"

        # Create our inputs...
        vocab_size = 1000
        batch_shape = [64]
        words_per_sentence = 5
        data = torch.randint(0, vocab_size, (*batch_shape, words_per_sentence), device=device)
        targets = torch.randint(0, 1, (*batch_shape,), device=device)

        # Construct our module
        net = SampleNet(vocab_size).to(device=device)
        criterion = nn.CrossEntropyLoss()

        net_func, weights = make_functional(net)

        def compute_loss(weights, data, target):
            output = net_func(weights, data)
            result = criterion(output, target)
            return result

        expected = [grad(compute_loss)(weights, data[i], targets[i]) for i in range(64)]
        expected = zip(*expected)
        expected = tuple(torch.stack(shards) for shards in expected)

        result = vmap(partial(grad(compute_loss), weights))(data, targets)
        for r, e in zip(result, expected):
            # TODO: Check if the rtol is a problem
            self.assertEqual(r, e, atol=0, rtol=1e-4)

    def test_log_softmax(self, device):
        x = torch.randn(3, 5)
        v = torch.randn(5)

        def foo(x, v):
            _, vjp_fn = vjp(partial(torch.log_softmax, dim=-1), x)
            return vjp_fn(v)[0]

        result = vmap(foo, (0, None))(x, v)

        v = v.expand_as(x)
        x.requires_grad_()
        output = torch.log_softmax(x, dim=-1)
        output.backward(v)
        self.assertEqual(result, x.grad)


class TestJacrev(TestCase):
    def test_simple(self, device):
        x = torch.randn(3, device=device)
        y = jacrev(torch.sin)(x)
        expected = torch.diagflat(x.cos())
        assert torch.allclose(y, expected)

    def test_simple_not_flat(self, device):
        x = torch.randn(2, 3, device=device)
        y = jacrev(torch.sin)(x)
        expected = torch.diagflat(x.view(-1).cos())
        expected = expected.view(2, 3, 2, 3)
        assert torch.allclose(y, expected)

    def test_vmap_on_jacrev_simple(self, device):
        x = torch.randn(2, 3, device=device)
        y = vmap(jacrev(torch.sin))(x)
        expected = torch.stack([torch.diagflat(x[i].cos()) for i in range(2)])
        assert torch.allclose(y, expected)

    def test_hessian_simple(self, device):
        def foo(x):
            return x.sin().sum()

        x = torch.randn(3, device=device)
        y = jacrev(jacrev(foo))(x)
        expected = torch.diagflat(-x.sin())
        assert torch.allclose(y, expected)


class TestComposability(TestCase):
    def test_grad_grad(self, device):
        x = torch.randn([], device=device)
        y = grad(grad(torch.sin))(x)
        self.assertEqual(y, -x.sin())

    def test_grad_vmap(self, device):
        def foo(x):
            y = vmap(torch.sin)(x)
            return y.sum()

        x = torch.randn(3)
        y = grad(foo)(x)
        self.assertEqual(y, x.cos())

    def test_grad_vjp(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)[0].sum()

        y = grad(foo)(x)
        expected = grad(lambda x: (x * x.cos()).sum())(x)
        self.assertEqual(y, expected)

    def test_vmap_grad(self, device):
        x = torch.randn(3, device=device)
        y = vmap(grad(torch.sin))(x)
        self.assertEqual(y, x.cos())

    def test_vmap_vmap(self, device):
        x = torch.randn(2, 3, device=device)
        y = vmap(vmap(torch.sin))(x)
        self.assertEqual(y, x.sin())

    def test_vmap_vjp(self, device):
        x = torch.randn(3, device=device)
        _, vjp_fn = vjp(torch.sin, x)

        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)

        y = vmap(foo)(x)
        self.assertEqual(y, vjp_fn(x))

        # TODO: there's a very interesting error message when the following
        # is on CPU
        xs = torch.randn(5, 3, device=device)
        expected = torch.stack([vjp_fn(x)[0] for x in xs])
        result = vmap(lambda x: vjp_fn(x)[0])(xs)
        self.assertEqual(result, expected)

    def test_vjp_grad(self, device):
        x = torch.randn([], device=device)
        y, vjp_fn = vjp(grad(torch.sin), x)
        self.assertEqual(y, x.cos())

        v = torch.randn([])
        self.assertEqual(vjp_fn(v)[0], -x.sin() * v)

    def test_vjp_vmap(self, device):
        x = torch.randn(3, device=device)
        y, vjp_fn = vjp(vmap(torch.sin), x)
        self.assertEqual(y, x.sin())

        v = torch.randn(3, device=device)
        self.assertEqual(vjp_fn(v)[0], x.cos() * v)

    def test_vjp_vjp(self, device):
        x = torch.randn(3, device=device)
        y, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(y, x.sin())

        y, vjp_fn = vjp(lambda x: vjp_fn(x)[0], x)
        self.assertEqual(y, x * x.cos())

        y = vjp_fn(x)[0]
        # Honestly IDK what the result here is... but at least it runs


class TestExamplesCorrectness(TestCase):
    def test_maml_regression(self, device):
        class ThreeLayerNet(nn.Module):
            def __init__(self):
                super(ThreeLayerNet, self).__init__()
                self.fc1 = nn.Linear(1, 40)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(40, 40)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(40, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.fc3(x)
                return x

        # The prototype doesn't like F.mse_loss.
        def mse_loss(x, y):
            return torch.mean((x - y) ** 2)

        net, params = make_functional(ThreeLayerNet().to(device))
        K = 20
        losses = []
        num_tasks = 4
        alpha = 0.1

        def sample_tasks(outer_batch_size, inner_batch_size):
            # Select amplitude and phase for the task
            As = []
            phases = []
            for _ in range(outer_batch_size):
                As.append(np.random.uniform(low=0.1, high=.5))
                phases.append(np.random.uniform(low=0., high=np.pi))
            def get_batch():
                xs, ys = [], []
                for A, phase in zip(As, phases):
                    x = np.random.uniform(low=-5., high=5., size=(inner_batch_size, 1))
                    y = A * np.sin(x + phase)
                    xs.append(x)
                    ys.append(y)
                return torch.tensor(xs, dtype=torch.float, device=device), \
                    torch.tensor(ys, dtype=torch.float, device=device)
            x1, y1 = get_batch()
            x2, y2 = get_batch()
            return x1, y1, x2, y2

        def get_loss_for_task(use_transform, x1, y1, x2, y2):
            def inner_loss(params, x1, y1):
                f = net(params, x1)
                loss = mse_loss(f, y1)
                return loss

            if use_transform:
                grads = grad(inner_loss)(params, x1, y1)
            else:
                loss = inner_loss(params, x1, y1)
                grads = torch.autograd.grad(loss, params, create_graph=True)
            new_params = [(params[i] - alpha*grads[i]) for i in range(len(params))]

            v_f = net(new_params, x2)
            return mse_loss(v_f, y2)

        task = sample_tasks(num_tasks, K)

        # Compute with vmap+grad
        inner_losses = vmap(partial(get_loss_for_task, True))\
                            (task[0], task[1], task[2], task[3])
        loss2 = sum(inner_losses)/len(inner_losses)
        result_grads = torch.autograd.grad(loss2, params)

        # Compute without vmap+grad
        inner_losses = [
            get_loss_for_task(False, task[0][i], task[1][i], task[2][i], task[3][i])
            for i in range(num_tasks)
        ]
        loss2 = sum(inner_losses)/len(inner_losses)
        expected_grads = torch.autograd.grad(loss2, params)

        self.assertEqual(result_grads, expected_grads)

    def test_maml_omniglot(self, device):
        # TODO: there appears to be precision issues for float32
        dtype = torch.double

        # TODO: The prototype doesn't support in-place relu (and some other
        # in-place operations. That can be fixed.)
        inplace_relu = False
        n_way = 5
        n_inner_iter = 2
        num_tasks = 2
        class Flatten(nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)

        net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(64, n_way)).to(device).to(dtype)

        fnet, params, buffers = make_functional_with_buffers(net)
        net = (params, buffers, fnet)

        def loss_for_task(net, n_inner_iter, use_transform, x_spt, y_spt, x_qry, y_qry):
            params, buffers, fnet = net
            querysz = x_qry.size(0)

            def compute_loss(new_params, buffers, x, y):
                logits = fnet(new_params, buffers, x)
                loss = F.cross_entropy(logits, y)
                return loss

            new_params = params
            for _ in range(n_inner_iter):
                if use_transform:
                    grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
                else:
                    res = compute_loss(new_params, buffers, x_spt, y_spt)
                    grads = torch.autograd.grad(res, new_params, create_graph=True)
                new_params = [p - g * 1e-1 for p, g, in zip(new_params, grads)]

            qry_logits = fnet(new_params, buffers, x_qry)
            qry_loss = F.cross_entropy(qry_logits, y_qry)
            qry_acc = (qry_logits.argmax(
                dim=1) == y_qry).sum() / querysz

            return qry_loss, qry_acc

        # Get some sample inputs...
        x_spt = torch.randn(num_tasks, 25, 1, 28, 28, dtype=dtype, device=device)
        y_spt = torch.randint(0, 5, (num_tasks, 25), device=device)
        x_qry = torch.randn(num_tasks, 75, 1, 28, 28, dtype=dtype,device=device)
        y_qry = torch.randint(0, 5, (num_tasks, 75), device=device)

        # compute with vmap + grad
        compute_loss = partial(loss_for_task, net, n_inner_iter, True)
        qry_losses, _ = vmap(compute_loss)(x_spt, y_spt, x_qry, y_qry)
        result_grads = torch.autograd.grad(qry_losses.sum(), params)

        # compute without vmap + grad
        compute_loss = partial(loss_for_task, net, n_inner_iter, False)
        losses = [compute_loss(x_spt[i], y_spt[i], x_qry[i], y_qry[i])[0]
                  for i in range(num_tasks)]
        expected_grads = torch.autograd.grad(sum(losses), params)

        self.assertEqual(result_grads, expected_grads)

    def test_lennard_jones_batched_jacrev(self, device):
        sigma = 0.5
        epsilon = 4.

        def lennard_jones(r):
            return epsilon * ((sigma / r)**12 - (sigma / r)**6)

        def lennard_jones_force(r):
            """Get magnitude of LJ force"""
            return \
                -epsilon * ((-12 * sigma**12 / r**13) + (6 * sigma**6 / r**7))

        r = torch.linspace(0.5, 2 * sigma, requires_grad=True)
        drs = torch.outer(r, torch.tensor([1.0, 0, 0]))
        norms = torch.norm(drs, dim=1).reshape(-1, 1)
        training_energies = \
            torch.stack(list(map(lennard_jones, norms))).reshape(-1, 1)
        training_forces = torch.stack(
            [force * dr
             for force, dr in zip(map(lennard_jones_force, norms), drs)])

        model = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        def make_prediction(model, drs, use_functorch):
            norms = torch.norm(drs, dim=1).reshape(-1, 1)
            energies = model(norms)

            if use_functorch:
                network_derivs = vmap(jacrev(model))(norms).squeeze(-1)
                forces = -network_derivs * drs / norms
            else:
                forces = []
                for r, dr in zip(norms, drs):
                    network_deriv = torch.autograd.functional.jacobian(
                        model, r, create_graph=True)
                    force = -network_deriv * dr / r
                    forces.append(force)
                forces = torch.cat(forces)
            return energies, forces

        def loss_fn(energies, forces, predicted_energies, predicted_forces):
            return F.mse_loss(energies, predicted_energies) + \
                0.01 * F.mse_loss(forces, predicted_forces) / 3

        energies, forces = make_prediction(model, drs, use_functorch=True)
        loss = loss_fn(training_energies, training_forces, energies, forces)
        result = torch.autograd.grad(loss, model.parameters())

        energies, forces = make_prediction(model, drs, use_functorch=False)
        loss = loss_fn(training_energies, training_forces, energies, forces)
        expected = torch.autograd.grad(loss, model.parameters())

        self.assertEqual(result, expected)

    def test_ensemble_regression(self, device):
        def make_spirals(n_samples, noise_std=0., rotations=1.):
            ts = torch.linspace(0, 1, n_samples)
            rs = ts ** 0.5
            thetas = rs * rotations * 2 * math.pi
            signs = torch.randint(0, 2, (n_samples,)) * 2 - 1
            labels = (signs > 0).to(torch.long)

            xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
            ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
            points = torch.stack([xs, ys], dim=1)
            return points.to(device), labels.to(device)

        points, labels = make_spirals(100, noise_std=0.05)

        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        loss_fn = nn.NLLLoss()

        func_model, weights = make_functional(MLPClassifier().to(device))

        def train_step_fn(use_transform, weights, batch, targets, lr=0.2):
            def compute_loss(weights, batch, targets):
                output = func_model(weights, batch)
                loss = loss_fn(output, targets)
                return loss

            if use_transform:
                grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)
            else:
                loss = compute_loss(weights, batch, targets)
                grad_weights = torch.autograd.grad(loss, weights)

            new_weights = []
            with torch.no_grad():
                for grad_weight, weight in zip(grad_weights, weights):
                    new_weights.append(weight - grad_weight * lr)
            # NB: return looks weird because torch.vmap must return Tensors
            return (loss, *new_weights)

        def unpack(train_result):
            return train_result[0], train_result[1:]

        def init_fn(num_models):
            models = tuple(MLPClassifier().to(device) for _ in range(num_models))
            weights = tuple(make_functional(model)[1] for model in models)
            weights = tuple(zip(*weights))
            weights = tuple(torch.stack(shards).detach() for shards in weights)
            return weights

        def slice_weights(batched_weights, index):
            return tuple(weight[index].detach().requires_grad_() for weight in batched_weights)

        batched_weights = init_fn(num_models=2)
        parallel_train_step_fn = vmap(partial(train_step_fn, True), in_dims=(0, None, None))

        result_loss, result_weights = unpack(parallel_train_step_fn(batched_weights, points, labels))

        loss0, weights0 = unpack(train_step_fn(False, slice_weights(batched_weights, 0), points, labels))
        loss1, weights1 = unpack(train_step_fn(False, slice_weights(batched_weights, 1), points, labels))
        expected_loss = torch.stack([loss0, loss1])
        expected_weights = tuple(torch.stack([w0, w1]) for w0, w1 in zip(weights0, weights1))

        self.assertEqual(result_loss, expected_loss)
        self.assertEqual(result_weights, expected_weights)

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_per_sample_grads(self, device):
        # Straight out of opacus
        def _replace_child(
            root: nn.Module, child_name: str, converter: Callable[[nn.Module], nn.Module]
        ) -> None:
            # find the immediate parent
            parent = root
            nameList = child_name.split(".")
            for name in nameList[:-1]:
                parent = parent._modules[name]
            # set to identity
            parent._modules[nameList[-1]] = converter(parent._modules[nameList[-1]])

        def replace_all_modules(
            root: nn.Module,
            target_class: Type[nn.Module],
            converter: Callable[[nn.Module], nn.Module],
        ) -> nn.Module:
            # base case
            if isinstance(root, target_class):
                return converter(root)

            for name, obj in root.named_modules():
                if isinstance(obj, target_class):
                    _replace_child(root, name, converter)
            return root

        def _batchnorm_to_groupnorm(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
            return nn.GroupNorm(min(32, module.num_features), module.num_features, affine=True)

        def convert_batchnorm_modules(
            model: nn.Module,
            converter: Callable[
                [nn.modules.batchnorm._BatchNorm], nn.Module
            ] = _batchnorm_to_groupnorm,
        ) -> nn.Module:
            return replace_all_modules(model, nn.modules.batchnorm._BatchNorm, converter)

        import torchvision.models as models
        model = convert_batchnorm_modules(models.resnet18(num_classes=10)).to(device)
        criterion = nn.CrossEntropyLoss()

        func_model, weights = make_functional(model)

        def compute_loss(weights, image, target):
            images = image.unsqueeze(0)
            targets = target.unsqueeze(0)
            output = func_model(weights, images)
            loss = criterion(output, targets)
            return loss

        batch_size = 3
        images = torch.randn(batch_size, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        result_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(weights, images, targets)

        expected_grads = [
            torch.autograd.grad(compute_loss(weights, images[i], targets[i]), weights)
            for i in range(batch_size)
        ]
        expected_grads = [torch.stack(shards) for shards in zip(*expected_grads)]

        self.assertEqual(result_grads, expected_grads)

only_for = ("cpu", "cuda")
instantiate_device_type_tests(
    TestGradTransform,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestVmapOfGrad,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestJacrev,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestComposability,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestExamplesCorrectness,
    globals(),
    only_for=only_for,
)



if __name__ == '__main__':
    run_tests()
