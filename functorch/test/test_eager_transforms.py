from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import functools
import itertools
import warnings
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma
import types
from functools import partial

import functorch
from functorch import grad, vjp, vmap, make_functional, jacrev


class TestGradTransform(TestCase):
    def test_primitive(self):
        x = torch.randn([])
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_composite_simple(self):
        x = torch.randn(2, 3, 4)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

    def test_composite_complicated(self):
        x = torch.randn(3)
        y = torch.randn(3, 5)

        def foo(x, y):
            result = x @ y
            return result.sum()

        result = grad(foo)(x, y)

        x.requires_grad_()
        out = foo(x, y)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_composite_two_ops(self):
        N, C = 2, 5
        y = torch.randn(N, C)
        targets = torch.randint(0, C, (N,))

        def foo(y, targets):
            return F.cross_entropy(y, targets)

        result = grad(foo)(y, targets)

        y.requires_grad_()
        expected, = torch.autograd.grad(foo(y, targets), y)

        self.assertEqual(result, expected)

    def _test_attributes(self, get_attr_lambda):
        x = torch.randn(2, 3, 5, dtype=torch.double)
        expected = get_attr_lambda(x)

        def foo(x):
            self.assertEqual(get_attr_lambda(x), expected)
            return x.sum()

        grad(foo)(x)

    def test_shape(self):
        self._test_attributes(lambda x: x.shape)

    def test_dtype(self):
        self._test_attributes(lambda x: x.dtype)

    def test_is_cuda(self):
        self._test_attributes(lambda x: x.is_cuda)

    def test_numel(self):
        self._test_attributes(lambda x: x.numel())

    def test_inplace(self):
        x = torch.randn([])

        def foo(x):
            return x.clone().sin_()

        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_inplace_on_view(self):
        x = torch.randn(3)

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

    def test_inplace_on_view_base(self):
        x = torch.randn(3)

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

    def test_nesting_simple(self):
        x = torch.randn([])
        result = grad(grad(torch.sin))(x)
        self.assertEqual(result, -torch.sin(x))

    def test_escaped_wrappers_are_marked_as_dead(self):
        x = torch.randn([])
        escaped = []
        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        result = grad(foo)(x)
        self.assertEqual(escaped[0].dlevel(), -1)

    def test_escaped_wrappers_are_ignored(self):
        x = torch.randn([])
        escaped = []
        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        result = grad(foo)(x)

        something = escaped[0].sum()
        self.assertEqual(something.dlevel(), 0)
        self.assertEqual(something, x.sin().sum())

    def test_vjp(self):
        x = torch.randn([])
        out, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(out, x.sin())

        v = torch.randn([])
        result, = vjp_fn(v)
        self.assertEqual(result, v * x.cos())

    def test_composed_with_autograd(self):
        x = torch.randn([], requires_grad=True)

        y = grad(torch.sin)(x)
        result, = torch.autograd.grad(y, x)
        self.assertEqual(result, -x.sin())

    def test_grad_of_vjp_composition(self):
        x = torch.randn([])
        y = torch.randn([])

        def foo(x, y):
            out, vjp_fn = vjp(torch.sin, x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_vjp_of_grad_composition(self):
        x = torch.randn([])
        y = torch.randn([])

        def foo(x, y):
            out, vjp_fn = vjp(grad(torch.sin), x)
            return vjp_fn(y)[0]

        result = foo(x, y)
        expected = -y * x.sin()
        self.assertEqual(result, expected)

    def test_grad_of_vjp_of_grad_composition(self):
        x = torch.randn([])
        y = torch.randn([])

        def foo(x, y):
            df, vjp_fn = vjp(grad(lambda x: -torch.cos(x)), x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_views(self):
        x = torch.randn([], requires_grad=True)
        y = torch.randn([], requires_grad=True)

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

    def test_view_inplace_simple(self):
        def foo(x):
            x = x.clone()
            x.view([]).sin_()
            return x

        x = torch.randn([], requires_grad=True)
        result = grad(foo)(x)
        self.assertEqual(result, x.cos())


class TestVmapOfGrad(TestCase):
    def test_per_sample_grads_inplace_view(self):
        def compute_loss(weight, x, t):
            x = x.mm(weight)
            y = x.squeeze_(0)
            return (y - t).sum()

        weight = torch.randn(16, 2)
        x = torch.randn(64, 1, 16)
        t = torch.randn(64, 2)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_new_zeros_materializes_tensor(self):
        N = 3
        C = 5

        def foo(x, y):
            result = x.new_zeros((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N)
        y = torch.randn(N, C)
        result = vmap(grad(foo))(x, y)

    def test_per_sample_grads_simple(self):
        def compute_loss(weight, x, t):
            y = x @ weight
            return ((y - t) ** 2).sum()

        weight = torch.randn(16, 2)
        x = torch.randn(64, 16)
        t = torch.randn(64, 2)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_per_sample_grads_embeddingnet(self):
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
        data = torch.randint(0, vocab_size, (*batch_shape, words_per_sentence))
        targets = torch.randint(0, 1, (*batch_shape,))

        # Construct our module
        net = SampleNet(vocab_size)
        criterion = nn.CrossEntropyLoss()

        params = dict(net.named_parameters())
        weights, net_func, _ = make_functional(net)

        def compute_loss(weights, data, target):
            output = net_func(weights, (data,))
            result = criterion(output, target)
            return result

        expected = [grad(compute_loss)(weights, data[i], targets[i]) for i in range(64)]
        expected = zip(*expected)
        expected = tuple(torch.stack(shards) for shards in expected)

        result = vmap(partial(grad(compute_loss), weights))(data, targets)
        for r, e in zip(result, expected):
            # TODO: Check if the rtol is a problem
            self.assertEqual(r, e, atol=0, rtol=1e-4)

class TestJacrev(TestCase):
    def test_simple(self):
        x = torch.randn(3)
        y = jacrev(torch.sin)(x)
        expected = torch.diagflat(x.cos())
        assert torch.allclose(y, expected)

    def test_simple_not_flat(self):
        x = torch.randn(2, 3)
        y = jacrev(torch.sin)(x)
        expected = torch.diagflat(x.view(-1).cos())
        expected = expected.view(2, 3, 2, 3)
        assert torch.allclose(y, expected)

    def test_vmap_on_jacrev_simple(self):
        x = torch.randn(2, 3)
        y = vmap(jacrev(torch.sin))(x)
        expected = torch.stack([torch.diagflat(x[i].cos()) for i in range(2)])
        assert torch.allclose(y, expected)

    def test_hessian_simple(self):
        def foo(x):
            return x.sin().sum()

        x = torch.randn(3)
        y = jacrev(jacrev(foo))(x)
        expected = torch.diagflat(-x.sin())
        assert torch.allclose(y, expected)


class TestComposability(TestCase):
    def test_grad_grad(self):
        x = torch.randn([])
        y = grad(grad(torch.sin))(x)
        self.assertEqual(y, -x.sin())

    def test_grad_vmap(self):
        def foo(x):
            y = vmap(torch.sin)(x)
            return y.sum()

        x = torch.randn(3)
        y = grad(foo)(x)
        self.assertEqual(y, x.cos())

    def test_grad_vjp(self):
        x = torch.randn(3)

        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)[0].sum()

        y = grad(foo)(x)
        expected = grad(lambda x: (x * x.cos()).sum())(x)
        self.assertEqual(y, expected)

    def test_vmap_grad(self):
        x = torch.randn(3)
        y = vmap(grad(torch.sin))(x)
        self.assertEqual(y, x.cos())

    def test_vmap_vmap(self):
        x = torch.randn(2, 3)
        y = vmap(vmap(torch.sin))(x)
        self.assertEqual(y, x.sin())

    def test_vmap_vjp(self):
        x = torch.randn(3)
        _, vjp_fn = vjp(torch.sin, x)

        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)

        y = vmap(foo)(x)
        self.assertEqual(y, vjp_fn(x))

        xs = torch.randn(5, 3)
        expected = torch.stack([vjp_fn(x)[0] for x in xs])
        self.assertEqual(vmap(lambda x: vjp_fn(x)[0])(xs), expected)

    def test_vjp_grad(self):
        x = torch.randn([])
        y, vjp_fn = vjp(grad(torch.sin), x)
        self.assertEqual(y, x.cos())

        v = torch.randn([])
        self.assertEqual(vjp_fn(v)[0], -x.sin() * v)

    def test_vjp_vmap(self):
        x = torch.randn(3)
        y, vjp_fn = vjp(vmap(torch.sin), x)
        self.assertEqual(y, x.sin())

        v = torch.randn(3)
        self.assertEqual(vjp_fn(v)[0], x.cos() * v)

    def test_vjp_vjp(self):
        x = torch.randn(3)
        y, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(y, x.sin())

        y, vjp_fn = vjp(lambda x: vjp_fn(x)[0], x)
        self.assertEqual(y, x * x.cos())

        y = vjp_fn(x)[0]
        # Honestly IDK what the result here is... but at least it runs


if __name__ == '__main__':
    run_tests()
