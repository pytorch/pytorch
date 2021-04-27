from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import functools
import itertools
import warnings
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma, onlyOnCPUAndCUDA
import types
from functools import partial

import functorch
from functorch import grad, vjp, vmap, make_functional, jacrev, make_functional_with_buffers

# NB: numpy is a testing dependency!
import numpy as np


class TestGradTransform(TestCase):
    def test_primitive(self, device):
        x = torch.randn([], device=device)
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_composite_simple(self, device):
        x = torch.randn(2, 3, 4, device=device)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

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

        params, net, _ = make_functional(ThreeLayerNet().to(device))
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
                f = net(params, (x1,))
                loss = mse_loss(f, y1)
                return loss

            if use_transform:
                grads = grad(inner_loss)(params, x1, y1)
            else:
                loss = inner_loss(params, x1, y1)
                grads = torch.autograd.grad(loss, params, create_graph=True)
            new_params = [(params[i] - alpha*grads[i]) for i in range(len(params))]

            v_f = net(new_params, (x2,))
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
        
        params, buffers, fnet, _, _, = make_functional_with_buffers(net)
        net = (params, buffers, fnet)

        def loss_for_task(net, n_inner_iter, use_transform, x_spt, y_spt, x_qry, y_qry):
            params, buffers, fnet = net
            querysz = x_qry.size(0)

            def compute_loss(new_params, buffers, x, y):
                logits = fnet(new_params, buffers, (x,))
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

            qry_logits = fnet(new_params, buffers, (x_qry,))
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


instantiate_device_type_tests(
    TestGradTransform,
    globals(),
    None,
)
instantiate_device_type_tests(
    TestVmapOfGrad,
    globals(),
    None,
)
instantiate_device_type_tests(
    TestJacrev,
    globals(),
    None,
)
instantiate_device_type_tests(
    TestComposability,
    globals(),
    None,
)
instantiate_device_type_tests(
    TestExamplesCorrectness,
    globals(),
    None,
)



if __name__ == '__main__':
    run_tests()
