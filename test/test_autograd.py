import contextlib
import gc
import sys
import math
import torch
import unittest
import warnings
from copy import deepcopy
from collections import OrderedDict
from itertools import product
from operator import mul, itemgetter
from functools import reduce, wraps
from torch.autograd.gradcheck import gradgradcheck, gradcheck
from torch.autograd.function import once_differentiable
from torch.autograd.profiler import profile
from common import TEST_MKL, TestCase, run_tests, skipIfNoLapack, \
    suppress_warnings
from torch.autograd import Variable, Function
from torch.autograd.function import InplaceFunction
from torch.testing import make_non_contiguous, randn_like

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

PRECISION = 1e-4


class NoArgsClass(object):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration()
    next = __next__  # Python 2 compatibility

    def __len__(self):
        return 0

NO_ARGS = NoArgsClass()


class non_differentiable(object):
    def __init__(self, tensor):
        self.tensor = tensor


@contextlib.contextmanager
def backward_engine(engine):
    _prev_engine = Variable._execution_engine
    Variable._execution_engine = engine()
    try:
        yield
    finally:
        Variable._execution_engine = _prev_engine


def graph_desc(fn):
    if fn is None:
        return 'None'
    result = type(fn).__name__ + '('
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        result += graph_desc(next_fn)
        result += ', '
    if next_functions:
        result = result[:-2]
    return result + ')'


class TestAutograd(TestCase):

    def _function_test(self, cls):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones(1, requires_grad=True)
        result.sum().backward(go, create_graph=True)

        self.assertEqual(x.grad.data, y.data + torch.ones(5, 5))
        self.assertEqual(y.grad.data, x.data + torch.ones(5, 5) * 2)
        self.assertIsNotNone(x.grad.grad_fn)
        self.assertIsNotNone(y.grad.grad_fn)

        return x, y

    def test_function(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_tensors
                # NOTE: self is the test case here
                self.assertIsInstance(var1, torch.Tensor)
                self.assertIsInstance(var2, torch.Tensor)
                self.assertIsInstance(grad_output, torch.Tensor)
                return (grad_output + grad_output * var2, None,
                        grad_output * ctx.pyscalar + grad_output * var1)

        x, y = self._function_test(MyFunction)

        x_grad_desc = graph_desc(x.grad.grad_fn)
        y_grad_desc = graph_desc(y.grad.grad_fn)
        self.assertEqual(
            x_grad_desc,
            'CloneBackward(AddBackward1(ExpandBackward(AccumulateGrad()), '
            'MulBackward1(ExpandBackward(AccumulateGrad()), AccumulateGrad())))')
        self.assertEqual(
            y_grad_desc,
            'CloneBackward(AddBackward1(MulBackward0(ExpandBackward(AccumulateGrad())), '
            'MulBackward1(ExpandBackward(AccumulateGrad()), AccumulateGrad())))')

    def test_once_differentiable(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                self.assertFalse(torch.is_grad_enabled())
                t1, t2 = ctx.saved_tensors
                return (grad_output + grad_output * t2, None,
                        grad_output * ctx.pyscalar + grad_output * t1)

        x, y = self._function_test(MyFunction)
        self.assertEqual(graph_desc(x.grad.grad_fn),
                         'CloneBackward(Error(AccumulateGrad(), None, AccumulateGrad()))')
        self.assertEqual(graph_desc(y.grad.grad_fn),
                         'CloneBackward(Error(AccumulateGrad(), None, AccumulateGrad()))')

    def test_function_returns_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad * 2

        v = torch.ones(1, requires_grad=True)
        MyFunction.apply(v).backward()
        self.assertEqual(v.grad.data.tolist(), [2])

        v.grad.data.zero_()
        MyFunction.apply(v.clone()).backward()
        self.assertEqual(v.grad.data.tolist(), [2])

    def test_legacy_function_none_grad(self):
        class MyFunction(Function):
            def forward(self, x):
                return torch.zeros(2, 2, 2)

            def backward(self, grad_output):
                return None

        shape = (2, 3)
        v = torch.ones(shape, requires_grad=True)
        y = v[0, 0].expand(3, 5).t().sum()
        MyFunction()(y).sum().backward()
        self.assertEqual(v.grad.data, torch.zeros(shape))

    def test_accumulate_grad(self):
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)
            return x_grad, x_grad_clone

        # Accumulate in-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)

        # Accumulate out-of-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=True)
        self.assertEqual(x_grad, x_grad_clone)

    def test_hessian_vector(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x.data + y.data
        y_grad = x.data + 2 * y.data
        self.assertEqual(x.grad.data, x_grad)
        self.assertEqual(y.grad.data, y_grad)

        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x.grad.data, x_grad + x_hv)
        self.assertEqual(y.grad.data, y_grad + y_hv)

    def test_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x.data + y.data
        y_grad = x.data + 2 * y.data
        self.assertEqual(x.grad.data, x_grad)
        self.assertEqual(y.grad.data, y_grad)

        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(
            outputs=[grad_sum], grad_outputs=[torch.ones(2, 2)],
            inputs=[x], create_graph=True)
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4

        self.assertEqual(x_hv[0].data, expected_x_hv)
        self.assertEqual(x.grad.data, x_grad)
        self.assertEqual(y.grad.data, y_grad)

    def test_grad_nonleaf(self):
        x_init = torch.randn(2, 2, requires_grad=True)
        x = x_init
        y = torch.randn(2, 2, requires_grad=True)
        grad_output = torch.ones(2, 2)

        def fn(x):
            return x ** 2 + y * x + y ** 2

        for i in range(5):
            grad_x, = torch.autograd.grad(
                fn(x), x, grad_outputs=grad_output, create_graph=True)

            grad_x_expected = 2 * x.data + y.data
            self.assertIsNone(y.grad)
            self.assertIsNone(x.grad)
            self.assertEqual(grad_x.data, grad_x_expected)

            x = x + 0.05 * grad_x

        val_init = fn(x_init).data.sum()
        val_final = fn(x).data.sum()
        self.assertGreater(val_final, val_init)

        x.backward(grad_output)
        self.assertIsNotNone(y.grad)
        self.assertIsNotNone(x_init.grad)

    def test_grad_nonleaf_many_outputs(self):
        # This checks an edge case for function callbacks
        # We want to capture two grads of a function, but can only
        # register a single callback.
        x = torch.randn(4, 2, requires_grad=True)
        a, b = x.chunk(2)

        def hook(*grads):
            hook_called[0] = True
        hook_called = [False]
        x.register_hook(hook)

        go = torch.randn(2, 2)
        grad_a, grad_b = torch.autograd.grad(
            (a + 2 * b), [a, b], grad_outputs=go, create_graph=True)

        self.assertEqual(grad_a.data, go)
        self.assertEqual(grad_b.data, go * 2)
        self.assertFalse(hook_called[0])
        self.assertIsNone(x.grad)

    def test_sharded_grad(self):
        leaves = [torch.zeros(5, 5, requires_grad=True) for _ in range(10)]
        intermediates = [l * i + l * l for i, l in enumerate(leaves)]
        loss = sum(v * i for i, v in enumerate(intermediates)).sum()

        # define a helper for dividing intermediates into groups
        def group(l, group_size):
            return (l[i:i + group_size] for i in range(0, len(l), group_size))

        # Compute the d loss / d intermediates in chunks of shard_size
        shard_size = 2
        d_intermediates = [d_i for intermediates_batch in group(intermediates, shard_size)
                           for d_i in torch.autograd.grad(loss, intermediates_batch)]
        # Compute rest of backward pass
        torch.autograd.backward(intermediates, d_intermediates)

        for i, l in enumerate(leaves):
            self.assertEqual(l.grad.data, i * i * (1 + l.data))

    def test_backward_badcalls(self):
        x = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            x.backward()

    def test_grad_badcalls(self):
        x = torch.ones(1)
        y = x ** 2
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            torch.autograd.grad(x, y)
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            torch.autograd.grad(y, x)

        x = torch.ones(1, requires_grad=True)
        y = x ** 2
        torch.autograd.grad(y, x)  # this should succeed now

    def test_grad_unreachable(self):
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # Make sure x and y have grad accumulators allocated
        z = x * 2
        w = y * 2

        grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)

        # This is slightly different than the case above, because z doesn't even
        # have a grad accumulator allocated.
        z = torch.ones(1, requires_grad=True)
        grad_x, grad_z = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)

    def test_hooks(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        counter = [0]

        def bw_hook(inc, grad):
            self.assertIsInstance(grad, torch.Tensor)
            counter[0] += inc

        z = x ** 2 + x * 2 + x * y + y
        x.register_hook(lambda *args: bw_hook(0, *args))
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 1)

        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 4)

        test2.remove()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            return grad.mul(2)

        test.remove()
        z.register_hook(bw_hook_modify)
        y.grad.data.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad.data, (x.data + 1) * 2)

        y.register_hook(bw_hook_modify)
        y.grad.data.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad.data, (x.data + 1) * 4)

    def test_hooks_cpp(self):
        # Tests hooks for autograd function implemented in C++
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.eval()

        counter = [0]

        def bw_hook(grad):
            counter[0] += 1
            return grad * 2

        x = torch.ones(5, 5, requires_grad=True)
        z = bn(x)
        z.register_hook(bw_hook)
        z.sum().backward()

        self.assertEqual(counter[0], 1, 'bw_hook not called')
        self.assertEqual(x.grad.data, torch.ones(5, 5) * 2)

    def test_hook_none(self):
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        class NoneGradientFunction(Function):

            def forward(self, x, y):
                assert self.needs_input_grad[0]
                assert not self.needs_input_grad[1]
                return x, y

            def backward(self, grad_x, grad_y):
                return grad_x, None

        fn = NoneGradientFunction()
        was_called = [False]

        def hook(grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertIsNotNone(grad_input[0])
            self.assertIsNotNone(grad_input[1])
            self.assertIsNotNone(grad_output[0])
            self.assertIsNotNone(grad_output[1])
            was_called[0] = True
        fn.register_hook(hook)

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        sum(fn(x, y)).sum().backward()
        self.assertTrue(was_called[0])

    def test_retain_grad(self):
        input = torch.rand(1, 3, requires_grad=True)
        h1 = input * 3
        out = (h1 * h1).sum()

        # It should be possible to call retain_grad() multiple times
        h1.retain_grad()
        h1.retain_grad()

        # Gradient should be accumulated
        out.backward(retain_graph=True)
        self.assertEqual(h1.data * 2, h1.grad.data)
        out.backward(retain_graph=True)
        self.assertEqual(h1.data * 4, h1.grad.data)

        input.grad.data.zero_()
        # It should be a no-op for leaves
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input.data * 18, input.grad.data)

    def test_retain_grad_cycle(self):
        import gc
        import weakref
        counter = [0]
        refs = [None]

        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            y = x * 2
            y.retain_grad()

            def inc(*args):
                counter[0] += 1
            refs[0] = weakref.ref(y, inc)
            return y / 2

        z = run_test()
        gc.collect()
        self.assertIsNone(refs[0]())
        self.assertEqual(counter[0], 1)
        z.sum().backward()

    def test_backward(self):
        v_t = torch.randn(5, 5)
        x_t = torch.randn(5, 5)
        y_t = torch.rand(5, 5) + 0.1
        z_t = torch.randn(5, 5)
        grad_output = torch.randn(5, 5)
        v = Variable(v_t, requires_grad=True)
        x = Variable(x_t, requires_grad=True)
        y = Variable(y_t, requires_grad=True)
        z = Variable(z_t, requires_grad=True)

        v.backward(grad_output)
        self.assertEqual(v.grad.data, grad_output)

        a = x + (y * z) + 4 * z ** 2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z_t.pow(2) / y_t + 1
        y_grad = z_t - 4 * x_t * z_t.pow(2) / y_t.pow(2)
        z_grad = 8 * x_t * z_t / y_t + y_t
        self.assertEqual(x.grad.data, x_grad * grad_output)
        self.assertEqual(y.grad.data, y_grad * grad_output)
        self.assertEqual(z.grad.data, z_grad * grad_output)

    def test_sparse_backward(self):
        class FixedGradientFunction(Function):

            def __init__(self, grad):
                self.grad = grad

            def forward(self, x):
                return x

            def backward(self, grad_x):
                return self.grad

        size = torch.Size([6, 3, 2])
        i1 = torch.LongTensor([
            [0, 3, 4],
            [0, 2, 2],
        ])
        v1 = torch.DoubleTensor([[1, 2], [4, 5], [7, 8]])
        sparse_grad1 = Variable(torch.sparse.DoubleTensor(i1, v1, size))
        i2 = torch.LongTensor([
            [0, 1, 3, 4],
            [0, 1, 2, 2],
        ])
        v2 = torch.DoubleTensor([[1, 2], [4, 3], [4, 5], [7, 8]])
        sparse_grad2 = Variable(torch.sparse.DoubleTensor(i2, v2, size))
        dense_grad = Variable(torch.rand(size).double())
        sparse_fn1 = FixedGradientFunction(sparse_grad1)
        sparse_fn2 = FixedGradientFunction(sparse_grad2)
        dense_fn = FixedGradientFunction(dense_grad)

        # sparse first
        x = torch.randn(5, 5, requires_grad=True)
        (sparse_fn1(x) + dense_fn(x) + sparse_fn2(x)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # dense first
        x = torch.randn(5, 5, requires_grad=True)
        (dense_fn(x) + sparse_fn1(x) + sparse_fn2(x)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # sparse only
        x = torch.randn(5, 5, requires_grad=True)
        (sparse_fn1(x) + sparse_fn2(x)).sum().backward()
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    def test_multi_backward(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q = torch.randn(5, 5, requires_grad=True)

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        q2 = q * 2
        z = x + y + q2
        c = a * b + q2
        grad_z = torch.randn(5, 5)
        grad_c = torch.randn(5, 5)
        torch.autograd.backward([z, c], [grad_z, grad_c])

        self.assertEqual(x.grad.data, grad_z)
        self.assertEqual(y.grad.data, grad_z)
        self.assertEqual(a.grad.data, grad_c * b.data)
        self.assertEqual(b.grad.data, grad_c * a.data)
        self.assertEqual(q.grad.data, (grad_c + grad_z) * 2)

    def test_multi_backward_no_grad(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=False)

        z = x + y
        q = y * 2

        # NB: we currently raise an exception if any arguments to backwards
        # have requires_grad=False and don't have a grad_fn. We may want to
        # relax that check to a warning.
        def call_backwards():
            torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])
        self.assertRaises(RuntimeError, call_backwards)

    def test_dependent_backward(self):
        x = torch.randn(10, requires_grad=True)
        y = x ** 2
        z = y ** 3

        go_y = torch.randn(10)
        go_z = torch.randn(10)
        torch.autograd.backward([y, z], [go_y, go_z])

        xd = x.data
        self.assertEqual(x.grad.data, 2 * xd * go_y + 6 * xd.pow(5) * go_z)

    def test_save_output_nr(self):
        x = torch.randn(10, requires_grad=True)

        class MultiOutputFn(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:5], x[5:]

            @staticmethod
            def backward(ctx, *grad):
                return torch.cat(grad)

        a, b = MultiOutputFn.apply(x)
        self.assertEqual(b.output_nr, 1)

        class TestFn(Function):
            @staticmethod
            def forward(ctx, b):
                ctx.save_for_backward(b)
                return b * 2

            @staticmethod
            def backward(ctx, grad_b):
                b, = ctx.saved_tensors
                self.assertEqual(b.output_nr, 1)

        TestFn.apply(b).sum().backward()

    def test_free_deep_graph(self):
        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build a "chain" computation graph
            for i in range(depth):
                y = y + y * 0.000001

            # graph deletion occurs when the above locals go out of scope.
            # In this case `del y` will trigger it but it's easier to leave
            # it to Python to delete the locals.

        # Should not stack overflow
        scope()

    def test_free_deep_graph_complicated(self):
        def scope():
            depth = 100000
            randchoice = torch.randint(2, [depth, 2])
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # Hold the two previous values
            prev_values = [None, None]

            # Build a "chain with skip connections" graph
            for i in range(depth):
                prev_tensors = [tensor for tensor in prev_values[:-1]
                                if tensor is not None]
                prev_values.append(y)
                prev_values.pop(0)

                # Definitely pick one tensor to add
                y += y * 0.000001

                # Possibly add other tensors
                nprev = len(prev_tensors)
                if nprev == 2:
                    y += randchoice[depth].mul(torch.cat(prev_tensors)).sum()

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    def test_free_deep_graph_pyfunction(self):
        class MyOp(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        def scope():
            depth = 150000
            x = torch.randn(1, requires_grad=True)
            y = x.clone()

            # build deeply nested computation graph
            for i in range(depth):
                y = MyOp.apply(y, y)

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

    def test_no_grad(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4)
        with torch.no_grad():
            w = x + y
        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.grad_fn)

    def test_no_grad_python_function(self):
        """Python Functions should respect grad mode."""
        x = torch.ones(5, 5, requires_grad=True)

        class MyOp(Function):
            @staticmethod
            def forward(self, x):
                return x + 1

            @staticmethod
            def backward(self, dy):
                return dy

        with torch.no_grad():
            y = MyOp.apply(x)
        self.assertFalse(y.requires_grad)

    def test_indexing(self):
        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        def compare(x, y, idx, indexed_tensor, indexed_var):
            indexed_var_t = indexed_var.data
            if not isinstance(indexed_tensor, torch.Tensor):
                indexed_var_t = indexed_var_t[0]
            self.assertEqual(indexed_tensor, indexed_var_t)

            indexed_var.sum().backward()
            expected_grad = torch.Tensor(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad.data, expected_grad)

        def check_index(x, y, idx):
            if y.grad is not None:
                y.grad.data.zero_()
            indexed_tensor = x[idx]
            indexed_var = y[idx]
            compare(x, y, idx, indexed_tensor, indexed_var)

        check_index(x, y, 1)
        check_index(x, y, (1, 1))
        check_index(x, y, slice(1, None))
        check_index(x, y, slice(None, 2))
        check_index(x, y, (slice(None, 2), 2))
        check_index(x, y, (slice(1, 2), 2))
        check_index(x, y, (1, slice(2, None)))
        check_index(x, y, (slice(None, None), slice(2, None)))
        check_index(x, y, torch.LongTensor([0, 2]))
        check_index(x, y, torch.rand(4, 4).bernoulli().byte())
        check_index(x, y, (Ellipsis, slice(2, None)))
        check_index(x, y, ([0], [0]))
        check_index(x, y, ([1, 2, 3], [0]))
        check_index(x, y, ([1, 2], [2, 1]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([slice(None), [2, 3]]))
        check_index(x, y, ([[2, 3], slice(None)]))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0]))
        check_index(x, y, ([0], ))

        x = torch.arange(1., 49).view(4, 3, 4)
        y = Variable(x, requires_grad=True)

        check_index(x, y, (slice(None), [0], [0]))
        check_index(x, y, ([0], [0], slice(None)))
        check_index(x, y, (slice(None), [0, 1, 2], [0]))
        check_index(x, y, ([0, 1, 2], [0], slice(None)))
        check_index(x, y, (slice(None), [1, 2], [2, 1]))
        check_index(x, y, ([1, 2], [2, 1], slice(None)))
        check_index(x, y, (slice(None), [[1, 2], [2, 0]], [[0, 1], [2, 3]]))
        check_index(x, y, ([[1, 2], [3, 0]], [[0, 1], [2, 2]], slice(None)))
        check_index(x, y, (slice(None), slice(None), [2, 1]))
        check_index(x, y, (slice(None), [2, 1], slice(None)))
        check_index(x, y, ([2, 1], slice(None), slice(None)))

        # advanced indexing, with less dim, or ellipsis
        check_index(x, y, ([0], ))
        check_index(x, y, ([0], slice(None)))
        check_index(x, y, ([0], Ellipsis))
        check_index(x, y, ([1, 2], [0, 1]))
        check_index(x, y, ([1, 2], [0, 1], Ellipsis))
        check_index(x, y, (Ellipsis, [1, 2], [0, 1]))

        # advanced indexing, with a tensor wrapped in a variable
        z = torch.LongTensor([0, 1])
        zv = Variable(z, requires_grad=False)
        seq = [z, Ellipsis]
        seqv = [zv, Ellipsis]

        if y.grad is not None:
            y.grad.data.zero_()
        indexed_tensor = x[seq]
        indexed_var = y[seqv]
        compare(x, y, seq, indexed_tensor, indexed_var)

    def test_indexing_duplicates(self):
        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx:
            expected_grad[i] += 1
        self.assertEqual(y.grad.data, expected_grad)

        # with advanced indexing
        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = [[1, 1, 3, 2, 1, 2], [0]]
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        self.assertEqual(y.grad.data, expected_grad)

        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = [[[1, 2], [0, 0]], [[0, 1], [1, 1]]]
        y[idx].sum().backward()
        expected_grad = torch.Tensor([[0, 2, 0, 0],
                                      [1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 0, 0]])
        self.assertEqual(y.grad.data, expected_grad)

        x = torch.arange(1., 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)

        idx = [[1, 1, 1], slice(None), slice(None)]
        y[idx].sum().backward()
        expected_grad = torch.Tensor(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad.data, expected_grad)

    def test_volatile_deprecated(self):
        v = torch.autograd.torch.randn(3, 3)
        with warnings.catch_warnings(record=True) as w:
            self.assertFalse(v.volatile)
        self.assertIn('volatile', str(w[0].message))

    def test_saved_variables_deprecated(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_variables
                return (grad_output, grad_output)

        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            x = torch.randn((3, 3), requires_grad=True)
            y = torch.randn((3, 3), requires_grad=True)
            model = MyFunction()
            model.apply(x, y).sum().backward()

            has_deprecated = map(lambda warn:
                                 'deprecated' in str(warn) and
                                 'saved_variables' in str(warn),
                                 warns)
            has_deprecated = reduce(lambda x, y: x or y, has_deprecated)
            self.assertTrue(has_deprecated)

    def test_requires_grad(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.randn(5, 5, requires_grad=True)
        a = x + y
        self.assertFalse(a.requires_grad)
        b = a + z
        self.assertTrue(b.requires_grad)

        def error():
            raise RuntimeError
        # Make sure backward isn't called on these
        a._backward_hooks = OrderedDict()
        x._backward_hooks = OrderedDict()
        y._backward_hooks = OrderedDict()
        a._backward_hooks['test'] = error
        x._backward_hooks['test'] = error
        y._backward_hooks['test'] = error
        b.backward(torch.ones(5, 5))

    def test_requires_grad_(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        self.assertIs(x, x.requires_grad_())
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_())
        self.assertTrue(y.requires_grad)
        self.assertIs(x, x.requires_grad_(True))
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_(True))
        self.assertTrue(y.requires_grad)
        z = x * y
        self.assertRaises(RuntimeError, lambda: z.requires_grad_(False))
        self.assertIs(z, z.requires_grad_())
        self.assertTrue(z.requires_grad)
        self.assertIs(z, z.requires_grad_(True))
        self.assertTrue(z.requires_grad)

        self.assertIs(x, x.requires_grad_(False))
        self.assertFalse(x.requires_grad)
        self.assertIs(y, y.requires_grad_(False))
        self.assertFalse(y.requires_grad)

    def test_requires_grad_inplace(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

        # non-leaf Variable
        a = torch.randn(5, 5) + 0
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

    def test_no_requires_grad_inplace(self):
        # basic case, should be able to modify inplace while requires_grad is False
        a = torch.randn(2, 3)
        a.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad.data, torch.ones(2, 3))

        # same but with a view
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad.data, torch.ones(2, 3))

        # should fail if requires_grad = True when we modify inplace
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    def test_requires_grad_factory(self):
        x = torch.randn(2, 3)
        fns = [torch.ones_like, torch.testing.randn_like]
        dtypes = [torch.float32, torch.float64]
        for fn in fns:
            for requires_grad in [True, False]:
                for dtype in dtypes:
                    for use_cuda in [True, False]:
                        if not use_cuda:
                            output = fn(x, dtype=dtype, requires_grad=requires_grad)
                            self.assertEqual(requires_grad, output.requires_grad)
                            self.assertIs(dtype, output.dtype)
                        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
                            output = fn(x, dtype=dtype, device=1, requires_grad=requires_grad)
                            self.assertEqual(requires_grad, output.requires_grad)
                            self.assertIs(dtype, output.dtype)
                            self.assertEqual(1, output.get_device())

    def test_grad_assignment(self):
        x = torch.randn(5, 5)
        a = torch.randn(2, 2)  # size mismatch
        b = Variable(torch.randn(5, 5).long())  # type mismatch

        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2)
        with self.assertRaises(RuntimeError):
            x.grad = Variable(torch.randn(5, 5).long())
        with self.assertRaises(RuntimeError):
            x.grad = x

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        with self.assertRaises(RuntimeError):
            x.grad = Variable(torch.randn(5, 5).cuda())

        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("At least 2 CUDA devices needed")
        x = Variable(torch.randn(5, 5).cuda(0))
        with self.assertRaises(RuntimeError):
            x.grad = Variable(torch.randn(5, 5).cuda(1))

    def test_duplicate_backward_root(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        x = a * b
        grad_output = torch.randn_like(x)
        torch.autograd.backward([x, x], [grad_output, grad_output])

        self.assertEqual(a.grad.data, b.data * grad_output * 2)
        self.assertEqual(b.grad.data, a.data * grad_output * 2)

    def test_backward_no_grad(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = a + 2
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    def test_next_functions(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        self.assertIsNotNone(a.grad_fn)
        next_functions = a.grad_fn.next_functions
        self.assertEqual(len(next_functions), 2)
        self.assertIsInstance(next_functions[0][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[0][1], 0)
        self.assertIsInstance(next_functions[1][0], torch._C._functions.AccumulateGrad)
        self.assertEqual(next_functions[1][1], 0)

        b = a + 5
        next_functions = b.grad_fn.next_functions
        self.assertEqual(len(next_functions), 1)
        self.assertIs(next_functions[0][0], a.grad_fn)

    def test_inplace(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5), retain_graph=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5), retain_graph=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5), retain_graph=True)
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        x.grad.data.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad.data, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad.data, torch.Tensor(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = torch.ones(5, 5, requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x.data, torch.ones(5, 5) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad.data, torch.ones(5, 5))
        z = x * y
        x.add_(2)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))

    def test_mark_non_differentiable(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input > 0
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return (grad_output * 0).type(torch.DoubleTensor)

        x = torch.randn(5, 5, requires_grad=True)
        mask = MyFunction.apply(x)
        self.assertFalse(mask.requires_grad)
        y = x.masked_fill(mask, 0)
        y.sum().backward()

    def test_mark_non_differentiable_mixed(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                a = input + 1
                b = input + 2
                ctx.mark_non_differentiable(a)
                return a, b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                self.assertTrue((grad_a == 0).all())
                self.assertTrue((grad_b == 1).all())
                return grad_b

        x = torch.randn(5, 5, requires_grad=True)
        a, b = MyFunction.apply(x)
        self.assertFalse(a.requires_grad)
        self.assertTrue(b.requires_grad)
        b.sum().backward()
        self.assertEqual(x.grad.data, torch.ones(5, 5))

    def test_mark_non_differentiable_none(self):
        # This used to segfault because MyFunction would send back null
        # gradients to MulBackward, which is implemented in C++. C++
        # implemented functions expect incoming  grad_ouptuts to be non-null.
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, input):
                output = input.clone()
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                return None

        x = torch.randn(5, 5, requires_grad=True)
        r = MyFunction.apply(x * x)
        (r * x).sum().backward()

    def test_return_duplicate(self):
        class DoubleDuplicate(Function):
            @staticmethod
            def forward(ctx, x):
                output = x * 2
                return output, output

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def fn(x):
            a, b = DoubleDuplicate.apply(x)
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, requires_grad=True)
        gradcheck(fn, [x])
        gradgradcheck(fn, [x])

    def test_return_duplicate_inplace(self):
        class DoubleInplace(Function):
            @staticmethod
            def forward(ctx, x):
                x.mul_(2)
                ctx.mark_dirty(x)
                return x, x

            @staticmethod
            def backward(ctx, grad1, grad2):
                return grad1 * 2 + grad2 * 2

        def inplace_fn(x):
            a, b = DoubleInplace.apply(x.clone())
            self.assertIs(a, b)
            return a + b

        x = torch.randn(5, 5, requires_grad=True)
        gradcheck(inplace_fn, [x])
        gradgradcheck(inplace_fn, [x])

        # Can't modify leaf variables in-place
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x))
        # Functions which modify views in-place must return only one output
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x.clone()[0]))

    @suppress_warnings
    def test_resize(self):
        x = torch.ones(2, 3)
        self.assertTrue(x.resize(3, 2).size() == (3, 2))

    def _test_setitem(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        expected_grad[index] = 0
        self.assertEqual(x.grad, expected_grad)

    def _test_setitem_tensor(self, size, index):
        x = torch.ones(*size, requires_grad=True)
        y = x + 2
        y_version = y._version
        value = x.new(x[index].size()).fill_(7)
        value.requires_grad = True
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        expected_grad_input[index] = 0
        self.assertEqual(x.grad, expected_grad_input)
        self.assertEqual(value.grad, torch.ones_like(value))

        # case when x broadcasts to as y[1]
        x = torch.randn(4, requires_grad=True)
        y = torch.zeros(2, 3, 4)
        y[1] = x
        y.backward(torch.randn(2, 3, 4))
        self.assertEqual(x.size(), x.grad.size())

    def test_setitem(self):
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem((10,), [[0, 4, 2]])
        self._test_setitem((5, 5), [[0, 4], [2, 2]])
        self._test_setitem((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5, 5), [[0, 1], [1, 0]])
        self._test_setitem_tensor((5,), 3)
        self._test_setitem_tensor((5,), Variable(torch.LongTensor([3]), requires_grad=False).sum())
        self._test_setitem_tensor((5,), [[0, 1, 2, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor((5, 5, 5), [Variable(torch.LongTensor([1,
                                              3]), requires_grad=False), [2, 4], slice(None)])

    def test_setitem_mask(self):
        mask = torch.ByteTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        # both select and sum return Scalars in ATen; ensure they work together.
        x = torch.randn(10, requires_grad=True)

        def func(x):
            return x.select(0, 1).sum()

        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_stack(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        z = torch.randn(10, 10, requires_grad=True)
        stacked = torch.stack([x, y, z], 0)
        grad = torch.randn(3, 10, 10)
        stacked.backward(grad)
        self.assertEqual(x.grad.data, grad[0])
        self.assertEqual(y.grad.data, grad[1])
        self.assertEqual(z.grad.data, grad[2])

    def test_put(self):
        root = torch.randn(4, 5, requires_grad=True)
        values = torch.randn(6, requires_grad=True)
        idx = Variable(torch.LongTensor([1, 2, 3, -1, -2, -3]))

        def func(root, values):
            x = root.clone()
            x.put_(idx, values)
            return x

        gradcheck(func, [root, values])
        gradgradcheck(func, [root, values])

    def test_put_accumulate(self):
        root = torch.randn(4, 5, requires_grad=True)
        values = torch.randn(6, requires_grad=True)
        idx = Variable(torch.LongTensor([1, 2, 3, 1, 2, 3]))

        def func(root, values):
            x = root.clone()
            x.put_(idx, values, accumulate=True)
            return x

        gradcheck(func, [root, values])
        gradgradcheck(func, [root, values])

    def test_fill(self):
        root = torch.randn(4, 5, requires_grad=True)

        def func(root):
            x = root.clone()
            x.fill_(2)
            return x

        gradcheck(func, [root])
        gradgradcheck(func, [root])

    def test_unused_output(self):
        x = torch.randn(10, 10, requires_grad=True)
        outputs = x.chunk(5)
        o = outputs[2]
        o = o * 4 + 2
        o.sum().backward()
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        self.assertEqual(x.grad.data, expected_grad)

        x.grad.data.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad.data, expected_grad)

    def test_gc_in_destructor(self):
        """
        Previously, if a Function destructor triggered a garbage collection,
        the Variable's tp_dealloc handler would get called twice leading to a
        segfault.
        """
        class CollectOnDelete(Function):

            def __del__(self):
                gc.collect()

        for i in range(10):
            Variable(torch.randn(10, 10), _grad_fn=CollectOnDelete())

    @unittest.skipIf(torch.cuda.device_count() < 2, "no multi-GPU")
    def test_unused_output_gpu(self):
        from torch.nn.parallel._functions import Broadcast
        x = Variable(torch.randn(5, 5).float().cuda(), requires_grad=True)
        outputs = Broadcast.apply(list(range(torch.cuda.device_count())), x)
        y = outputs[-1] * 2
        y.sum().backward()
        self.assertEqual(x.grad.data, torch.ones(5, 5) * 2)

    @unittest.skipIf(torch.cuda.device_count() < 2, "no multi-GPU")
    def test_backward_device(self):
        # check that current device matches the variable's device
        device = [None]

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                device[0] = torch.cuda.current_device()
                return grad_output.clone()

        v = Variable(torch.randn(1).cuda(1), requires_grad=True)
        Identity.apply(v).backward()
        self.assertEqual(device[0], 1)

    @unittest.skipIf(torch.cuda.device_count() < 2, "no multi-GPU")
    def test_inputbuffer_add_multigpu(self):
        input = torch.randn(1).cuda(0).requires_grad_()
        output = input.cuda(1) + input.cuda(1)
        output.backward()

    def test_detach(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(x.grad.data, torch.ones(10, 10))

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertEqual(x.grad.data, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad.data, torch.ones(10, 10) * 2)

        # in-place deatch on a view raises an exception
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, 'view', lambda: view.detach_())

    def test_detach_base(self):
        "detaching base does not detach view"
        x = torch.randn(10, 10, requires_grad=True)
        view = x.narrow(0, 1, 4)
        x.detach_()
        self.assertFalse(x.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNotNone(view.grad_fn)
        self.assertIs(view._base, x)

    def _test_type_conversion_backward(self, t, ):
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad.data), type(fvar.data))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad.data), type(dvar.data))

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIsInstance(x.float().cuda(), torch.cuda.FloatTensor)
            self.assertIsInstance(x.int().cuda(), torch.cuda.IntTensor)
            self.assertIsInstance(x.int().cuda().cpu(), torch.IntTensor)
            if torch.cuda.device_count() >= 2:
                x2 = x.float().cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIsInstance(x2.data, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().data.get_device(), 1)

        for t in [torch.DoubleTensor, torch.FloatTensor, torch.IntTensor, torch.ByteTensor]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                # TODO: t.dtype should work
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                if torch.cuda.is_available():
                    for x_cuda in (True, False):
                        for y_cuda in (True, False):
                            x_c = x.cuda() if x_cuda else x
                            y_c = y.cuda() if y_cuda else y
                            _, y_type = y_c.type().rsplit('.', 1)
                            y_typestr = ('torch.cuda.' if y_cuda else 'torch.') + y_type
                            self.assertEqual(y_c.type(), x_c.type(y_typestr).type())
                            self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                            self.assertEqual(y_c.data_ptr(), y_c.cuda().data_ptr() if y_cuda else y_c.data_ptr())

        self._test_type_conversion_backward(lambda x: x)
        if torch.cuda.is_available():
            self._test_type_conversion_backward(lambda x: x.cuda())
            if torch.cuda.device_count() >= 2:
                # one of these has to be the non-default device
                self._test_type_conversion_backward(lambda x: x.cuda(0))
                self._test_type_conversion_backward(lambda x: x.cuda(1))

    def _test_pyscalar_conversions(self, t, integral_conv):
        # integral -> integral
        l = t(torch.zeros(1, 1, 1, dtype=torch.long))
        pyscalar = -12345
        l[0] = pyscalar
        self.assertEqual(integral_conv(l), pyscalar)

        # floating point -> floating point
        f = Variable(t(torch.randn(1, 1)))
        pyscalar = -12345.1
        f[0] = pyscalar
        self.assertEqual(float(f), pyscalar)
        f[0] = float('nan')
        self.assertTrue(math.isnan(float(f)))
        f[0] = float('inf')
        self.assertEqual(float(f), float('inf'), allow_inf=True)
        f[0] = float('-inf')
        self.assertEqual(float(f), float('-inf'), allow_inf=True)

        # integral -> floating point
        # check we can convert something that loses precision
        pyscalar = 1234567890123456789
        self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
        l[0] = pyscalar
        self.assertEqual(float(l), float(pyscalar))

        # floating point -> integral
        f[0] = float('nan')
        self.assertRaises(ValueError, lambda: integral_conv(f[0]))
        f[0] = float('inf')
        self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
        f[0] = float('-inf')
        self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
        f[0] = sys.float_info.max
        self.assertEqual(integral_conv(f), sys.float_info.max)

        # bool, nonzero
        def test_nonzero(tensor, value, expected):
            tensor[0] = value
            self.assertEqual(expected, bool(tensor))
            self.assertEqual(expected, True if tensor else False)

        test_nonzero(l, 0, False)
        test_nonzero(l, -2, True)
        test_nonzero(f, 0.0, False)
        test_nonzero(f, sys.float_info.min, True)
        test_nonzero(f, float('nan'), bool(float('nan')))
        test_nonzero(f, float('inf'), bool(float('inf')))
        test_nonzero(f, float('-inf'), bool(float('-inf')))

    def test_pyscalar_conversions(self):
        self._test_pyscalar_conversions(lambda x: x, lambda x: int(x))
        if sys.version_info[0] == 2:
            self._test_pyscalar_conversions(lambda x: x, lambda x: long(x))
        if torch.cuda.is_available():
            self._test_pyscalar_conversions(lambda x: x.cuda(), lambda x: int(x))
            if sys.version_info[0] == 2:
                self._test_pyscalar_conversions(lambda x: x.cuda(), lambda x: long(x))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_pin_memory(self):
        x = torch.randn(2, 2, requires_grad=True)
        self.assertEqual(x, x.pin_memory())
        self.assertIsNot(x, x.pin_memory())
        self.assertTrue(x.pin_memory().requires_grad)
        gradcheck(lambda x: x.pin_memory(), [x])
        gradgradcheck(lambda x: x.pin_memory(), [x])

    def test_isolated_node(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        a = x + y
        b = torch.max(a, 1, True)[1].repeat(1, 5).double()
        o = (b + a).sum()
        o.backward()

    def test_shape(self):
        x = torch.randn(3, 4)
        self.assertEqual(2, len(x.shape))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[1], 4)

    def test_numpy_requires_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        self.assertRaisesRegex(RuntimeError, 'requires grad', lambda: x.numpy())

    def test_return_leaf(self):
        class Identity(Function):

            def forward(self, a, b):
                return a, a + b

            def backward(self, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        hook_called = [False]
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Identity()(x, y)

        # Make sure hooks only receive grad from usage of q, not x.
        def hook(grad):
            hook_called[0] = True
            self.assertEqual(grad.data, torch.ones(5, 5))

        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad.data, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad.data, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        class Inplace(InplaceFunction):

            def forward(self, a, b):
                self.mark_dirty(a)
                return a.add_(b), b + 2

            def backward(self, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)

        fn = Inplace(True)
        q, p = fn(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn, fn)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad.data, torch.ones(5, 5))

    def test_leaf_assignment(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad.data, torch.ones(5))
        self.assertEqual(z.grad.data, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5)
        with torch.no_grad():
            x[0] = y

        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        z = (x * y).sum()
        with torch.no_grad():
            x *= 2
        self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation',
                               lambda: z.backward())

    def test_no_grad_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(self, x):
                return x

            @staticmethod
            def backward(self, grad_output):
                return grad_output

        x = torch.randn(5, requires_grad=True)
        with torch.no_grad():
            y = MyFunction.apply(x)

        self.assertTrue(x.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_copy(self):
        # This tests checks backward engine for a very subtle bug that appreared
        # in one of the initial versions of autograd. Gradients tensors were
        # simply stored in lists while the function waited for all its gradients
        # to be computed. However, sometimes an output was used multiple times,
        # so the gradients needed to be summed. Engine used to keep a need_copy
        # set of tensors that will need a clone upon next addition and removed
        # them from the set as soon as the clone was performed. However, this
        # could lead to incorrect results if the same gradient tensor was
        # buffered in three places in the graph:
        # 1. When accumulating gradients in one of these places it was cloned
        #    and removed from need_copy set.
        # 2. When accumulating in second place, it wasn't in the need_copy set,
        #    so the gradients were simply accumulated in-place (which already
        #    modified the grad in 3rd place)
        # 3. When accumulating in the third place, it wasn't in the need_copy set
        #    as well, so the incoming gradient was summed in-place, yielding
        #    incorrect results in all functions, except the first one.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        # Simulate that we're in the middle of the graph
        a = x + 2
        b = y + 2
        c = x + 2
        # This op will just return grad_output two times in backward
        add1 = a + b
        add2 = add1 + c
        # Simulate a long branch, so grad_output will get buffered.
        for i in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        # expected gradients are:
        # for x: 34 (16 from final a, 16 from final c, 2 from add2)
        # for y: 17 (16 from final b, 1 from add2)
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        test_case = self

        class MyFn(Function):

            def forward(self, input):
                self.save_for_backward(None, input, None)
                return input * input

            def backward(self, grad_output):
                n1, input, n2 = self.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn()(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        class MyFn(Function):

            def forward(self, input):
                return input

            def backward(self, grad_output):
                return grad_output, None, None

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn()(x)
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_pickle(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            self.assertEqual(var1.data, var2.data)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    def test_dep_nograd(self):
        class F1(Function):

            def forward(self, input):
                out = torch.randn(input.size())
                self.mark_non_differentiable(out)
                return input, out

            def backward(self, grad_output, ignored):
                return grad_output

        class F2(Function):

            def forward(self, input, ignored):
                return input

            def backward(self, grad_output):
                return grad_output, None

        x = torch.randn(5, requires_grad=True)
        a, b = F1()(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2()(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad.data, torch.ones(x.size()))

    def test_set_grad_enabled(self):
        x = torch.tensor([1], requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)

    def test_reentrant(self):
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.data, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad.data, y_data)

    def test_cat(self):
        f_args_variable = (torch.randn(1, S, S, requires_grad=True),
                           torch.randn(2, S, S, requires_grad=True),
                           torch.randn(3, S, S, requires_grad=True),
                           0)
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat", "cat",
                              lambda a, b, c, dim: torch.cat((a, b, c), dim),
                              True, f_args_variable, f_args_tensor)

    def test_cat_negdim_1(self):
        f_args_variable = (torch.randn(S, S, 1, requires_grad=True),
                           torch.randn(S, S, 2, requires_grad=True),
                           torch.randn(S, S, 3, requires_grad=True),
                           -1)
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_negdim_1", "cat",
                              lambda a, b, c, dim: torch.cat((a, b, c), dim),
                              True, f_args_variable, f_args_tensor)

    def test_cat_negdim_2(self):
        f_args_variable = (torch.randn(S, 1, S, requires_grad=True),
                           torch.randn(S, 2, S, requires_grad=True),
                           torch.randn(S, 3, S, requires_grad=True),
                           -2)
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_negdim_2", "cat",
                              lambda a, b, c, dim: torch.cat((a, b, c), dim),
                              True, f_args_variable, f_args_tensor)

    def test_cat_empty(self):
        f_args_variable = (torch.randn(0, requires_grad=True),
                           torch.randn(S, S, requires_grad=True))
        # gradgradcheck doesn't work (because gradcheck doesn't work for empty outputs?)
        # hence False passed below, but gradcheck checked explicitly.
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_empty", "cat",
                              lambda a, b: torch.cat((a, b)),
                              False, f_args_variable, f_args_tensor)
        self.assertTrue(gradcheck(lambda a, b: torch.cat((a, b)), f_args_variable, eps=1e-6, atol=PRECISION))

    @skipIfNoLapack
    def test_potrf(self):
        root = Variable(torch.tril(torch.rand(S, S)), requires_grad=True)

        def run_test(upper):
            def func(root):
                x = torch.mm(root, root.t())
                return torch.potrf(x, upper)

            gradcheck(func, [root])
            gradgradcheck(func, [root])

        run_test(upper=True)
        run_test(upper=False)

    @skipIfNoLapack
    def test_trtrs(self):
        def _test_with_size(N, C):
            A = torch.rand(N, N, requires_grad=True)
            b = torch.rand(N, C, requires_grad=True)

            for upper, transpose, unitriangular in product((True, False), repeat=3):
                def func(A, b):
                    return torch.trtrs(b, A, upper, transpose, unitriangular)

                gradcheck(func, [A, b])
                gradgradcheck(func, [A, b])

        _test_with_size(S, S + 1)
        _test_with_size(S, S - 1)

    @unittest.skipIf(not TEST_MKL, "PyTorch is built without MKL support")
    def test_fft_ifft_rfft_irfft(self):
        def _test_complex(sizes, signal_ndim):
            x = torch.randn(sizes, requires_grad=True, dtype=torch.double)

            for normalized in (True, False):
                def fft(x):
                    return x.fft(signal_ndim, normalized=normalized)

                gradcheck(fft, [x])
                gradgradcheck(fft, [x], gen_non_contig_grad_outputs=True)

                def ifft(fx):
                    return fx.ifft(signal_ndim, normalized=normalized)

                # Use output of fft(x) for inverse fft, due to symmetry requirements
                fx = fft(x).detach()
                fx.requires_grad = True
                gradcheck(ifft, [fx])
                gradgradcheck(ifft, [fx], gen_non_contig_grad_outputs=True)

        def _test_real(sizes, signal_ndim):
            x = torch.randn(sizes, requires_grad=True, dtype=torch.double)
            if x.dim() == signal_ndim:
                start_dim = 0
            else:
                start_dim = 1
            signal_sizes = x.size()[start_dim:start_dim + signal_ndim]

            for normalized, onesided in product((True, False), repeat=2):
                def rfft(x):
                    return x.rfft(signal_ndim, normalized=normalized, onesided=onesided)

                gradcheck(rfft, [x])
                gradgradcheck(rfft, [x], gen_non_contig_grad_outputs=True)

                # Generally speaking, irfft itself won't and can't pass the
                # current gradcheck as it assumes the input follows conjugate
                # symmetry, an requirement that is never true with our point
                # numerical Jacobian estimate. Without input symmtry, irfft's
                # behavior is undefined.
                #
                # Even onesided results can't remove all redundancy. For
                # example, consider the .select(last_signal_dim, 0) slice.
                # It is entirely represented in the onesided results (except
                # for 1D), and will be reflected onto itself!
                #
                # So only 1D onesided irfft should pass grad check as it is
                # guaranteed that the input has no symmetrical values.
                #
                # In other cases, we test a function that first uses rfft to
                # generate a tensor that follows the conjugate symmetry irfft
                # expects, and then feeds it into irfft. Since rfft is already
                # tested above, we thereby verify the correctness of irfft.
                if signal_ndim == 1 and onesided:
                    def irfft(fx):
                        return fx.irfft(signal_ndim, normalized=normalized,
                                        onesided=onesided, signal_sizes=signal_sizes)

                    # Use output of rfft(x) for inverse rfft, due to symmetry requirements
                    fx = rfft(x).detach()
                    fx.requires_grad = True
                    gradcheck(irfft, [fx])
                    gradgradcheck(irfft, [fx], gen_non_contig_grad_outputs=True)
                else:
                    # Test this function: f(x) = ifft(rfft(x) + rfft(z)), where
                    # z is some fixed tensor of same size as x. rfft(z) term is
                    # needed because otherwise f becomes identity.
                    z = torch.randn(sizes, dtype=torch.double)
                    fz = z.rfft(signal_ndim, normalized=normalized, onesided=onesided)

                    def rfft_irfft(x):
                        fx = x.rfft(signal_ndim, normalized=normalized, onesided=onesided)
                        y = fx + fz
                        return y.irfft(signal_ndim, normalized=normalized,
                                       onesided=onesided, signal_sizes=signal_sizes)

                    gradcheck(rfft_irfft, [x])
                    gradgradcheck(rfft_irfft, [x], gen_non_contig_grad_outputs=True)

        _test_real((2, 10), 1)
        _test_real((2, 3, 4), 2)
        _test_real((2, 3, 4, 3), 3)

        _test_complex((2, 2, 10, 2), 1)
        _test_complex((1, 2, 3, 4, 2), 2)
        _test_complex((2, 1, 3, 4, 3, 2), 3)

    def test_variable_traverse(self):
        def get_out_and_unrefed_cycle():
            inp = torch.randn(10, requires_grad=True)
            tmp = inp.view(10, 1)
            out = tmp.view(10)

            # Create a reference cycle that contains an
            # intermediary Variable in the graph
            my_list = []
            my_list.append(tmp)
            my_list.append(my_list)

            return out

        out = get_out_and_unrefed_cycle()
        gc.collect()
        # This will segfault if things have been erroneously released
        out.backward(torch.randn(out.size()))

    def test_norm_subgradient(self):
        def run_test(input_size, norm_deg):
            input = torch.zeros(*input_size, requires_grad=True)
            input.norm(norm_deg).backward()
            self.assertEqual(input.grad.data.abs().sum(), 0)

        run_test((10,), 2)
        run_test((10, 10), 2)
        run_test((10,), 3)
        run_test((10,), 1)
        run_test((10,), 1.5)

    def test_profiler(self):
        x = torch.randn(10, 10)

        with profile() as p:
            y = x * 2 + 4

        last_end = 0
        names = ['mul', 'add']
        self.assertEqual(len(p.function_events), len(names))
        for info, expected_name in zip(p.function_events, names):
            self.assertGreater(info.cpu_interval.start, last_end)
            self.assertEqual(info.name, expected_name)
            last_end = info.cpu_interval.end

    def test_dir(self):
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn('shape', keys)

        for key in keys:
            self.assertTrue(hasattr(x, key))

    def test_as_strided(self):
        x = Variable(torch.arange(0., 25).view(5, 5), requires_grad=True)

        def as_strided(x):
            return x.as_strided([3, 3], [6, 2], 2)

        gradcheck(as_strided, [x], raise_exception=True)
        gradgradcheck(as_strided, [x], [torch.randn(3, 3)])

    def _test_where_functional(self, t):
        x = Variable(t(torch.randn(5, 5)), requires_grad=True)
        y = Variable(t(torch.randn(5, 5)), requires_grad=True)
        cond = Variable(t(mask_not_all_zeros((5, 5))), requires_grad=False)

        def where(cond, x, y):
            return torch.where(cond, x, y)

        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [Variable(t(torch.randn(5, 5)))])

        x = Variable(t(torch.randn(5, 1, 5)), requires_grad=True)
        y = Variable(t(torch.randn(5, 5, 1)), requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [Variable(t(torch.randn(5, 5, 5)))])

    def test_where_functional(self):
        self._test_where_functional(lambda t: t)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_where_functional_cuda(self):
        self._test_where_functional(lambda t: t.cuda())

    def test_inplace_view_backprop_base(self):
        # modify view and back-prop through base
        root = torch.randn(2, 2, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v1.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.data.tolist(), [[2, 2], [1, 1]])

    def test_inplace_view_backprop_view_of_view(self):
        # modify view and backprop through view-of-view
        root = torch.randn(2, 2, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = x.narrow(0, 0, 1)
        v1.mul_(2)
        v2.sum().backward()
        self.assertEqual(root.grad.data.tolist(), [[2, 2], [0, 0]])

    def test_inplace_view_of_view(self):
        # modify view-of-view and backprop through base
        root = torch.randn(2, 2, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.data.tolist(), [[1, 2], [1, 1]])

    def test_inplace_view_gradcheck(self):
        # gradcheck modifications to views
        a = torch.randn(4, 4, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)

        def func(root, b):
            x = root.clone()
            x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
            x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_view_makes_base_require_grad(self):
        # in-place modification to view makes base require grad
        a = torch.randn(4, 4, requires_grad=False)
        b = torch.randn(4, 2, requires_grad=True)

        def func(root, b):
            x = root.clone()
            self.assertFalse(x.requires_grad)
            x.narrow(1, 2, 2).mul_(b)
            self.assertTrue(x.requires_grad)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_view_backprop_view(self):
        # modify view and backprop through view
        a = Variable(torch.Tensor([2, 5]), requires_grad=False)
        b = Variable(torch.Tensor([3]), requires_grad=True)
        res = a.narrow(0, 1, 1).mul_(b)
        res.sum().backward()
        self.assertEqual(b.grad.data.tolist(), [5])
        self.assertIsNone(a.grad)

    def test_inplace_view_modify_base(self):
        # Test that an in-place operation on a base that forced it to require
        # grad also forces any previous views to require grad and backprop
        # correctly
        r = torch.ones(1, requires_grad=True)

        def fn(r):
            x = torch.ones(5)
            v = x.select(0, 1)
            self.assertFalse(v.requires_grad)
            self.assertIsNone(v.grad_fn)
            x.add_(r)  # v is now dependent on r due to the in-place op on x
            self.assertTrue(v.requires_grad)
            return v

        gradcheck(fn, [r])
        gradgradcheck(fn, [r])

    def test_inplace_view_python(self):
        # in-place modifications of Python-autograd created view
        a = torch.randn(4, 4, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)

        class PyAdd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.mark_dirty(x)
                x.add_(y)
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        def func(root, b):
            x = root.clone()
            PyAdd.apply(x.narrow(1, 2, 2).narrow(0, 1, 2), b)
            PyAdd.apply(x.narrow(1, 0, 2).narrow(0, 1, 2), b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_view_non_contig(self):
        data = torch.ones(2, 3, 2).select(2, 1).t()
        root = Variable(data, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.data.tolist(), [[1, 2], [1, 1], [1, 1]])

    def test_inplace_view_saved_output(self):
        # Test an in-place operation on a view in which the in-place op saves
        # its output. Previously, this created a reference cycle.
        dealloc = [0]

        class IncrementOnDelete(object):
            def __del__(self):
                dealloc[0] += 1

        def test():
            root = torch.randn(3, 3, requires_grad=True)
            copy = root.clone()
            copy.grad_fn.register_hook(IncrementOnDelete())
            view = copy.view(9)
            torch.nn.functional.relu(view, inplace=True)

        test()
        self.assertEqual(dealloc[0], 1)

    def test_mul_out(self):
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        x = torch.zeros_like(a)

        # out=... functions don't support automatic differentiation currently
        self.assertRaisesRegex(RuntimeError, 'out=', lambda: torch.mul(a, b, out=x))

        # the inputs can require grad if we're in no_grad() mode
        with torch.no_grad():
            torch.mul(a, b, out=x)
            self.assertEqual(x, a * b)

    def test_mul_out_result_requires_grad(self):
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        x = torch.zeros(2, 2, requires_grad=True)
        # we should throw an exception if the output requires grad
        self.assertRaisesRegex(RuntimeError, 'out=', lambda: torch.mul(a, b, out=x))

    def test_diagonal_derivative_requires_grad(self):
        # test that the backward requires grad
        # we do this is because diagonal_backward uses inplace
        # operations and gradgradcheck does not catch whether
        # they works as expected (it will succeed even if
        # the gradient has requires_grad == False
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.diagonal(a)**2
        c = b.sum()
        d, = torch.autograd.grad(c, a, retain_graph=True, create_graph=True)
        self.assertTrue(d.requires_grad)

    @staticmethod
    def _test_set_requires_grad_only_for_floats(self, cuda):
        dtypes = [torch.int64, torch.int32, torch.int16, torch.int8,
                  torch.float, torch.double]
        if cuda:
            dtypes.append(torch.half)

        def f1(dt):
            a = torch.ones(1, dtype=dt, device='cuda' if cuda else 'cpu')
            a.requires_grad_()

        def f2(dt):
            a = torch.ones(1, dtype=dt, device='cuda' if cuda else 'cpu')
            a.requires_grad = True

        def f3(dt):
            torch.ones(1, dtype=dt, device='cuda' if cuda else 'cpu', requires_grad=True)

        for dt in dtypes:
            a = torch.ones(1, dtype=dt, device='cuda' if cuda else 'cpu')
            a.requires_grad = False  # should always work
            a.requires_grad_(False)

            for f in [f1, f2, f3]:
                if dt.is_floating_point:
                    f(dt)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'floating point',
                                                msg="dt: {} device: {}".format(a.dtype, a.device)):
                        f(dt)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_set_requires_grad_only_for_floats_cuda(self):
        self._test_set_requires_grad_only_for_floats(self, True)

    def test_set_requires_grad_only_for_floats(self):
        self._test_set_requires_grad_only_for_floats(self, False)


def index_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape).mul_(max_indices).floor_().long()
    return index


def index_perm_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index


def gather_variable(shape, index_dim, max_indices, duplicate=False):
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = torch.LongTensor(*shape)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(
            torch.randperm(max_indices)[:shape[batch_dim]])
    if duplicate:
        index.select(batch_dim, 0).copy_(index.select(batch_dim, 1))
    return index


def mask_not_all_zeros(shape):
    assert len(shape) > 0
    while True:
        result = torch.randn(shape).gt(0)
        if result.sum() > 0:
            return result


def prod_zeros(dim_size, dim_select):
    assert len(dim_select) == 2
    result = torch.randn(dim_size, dim_size, dim_size)
    result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
    result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
    result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()
    return result


def prod_single_zero(dim_size):
    result = torch.randn(dim_size, dim_size)
    result[0, 1] = 0
    return result


def random_square_matrix_of_rank(l, rank):
    assert rank <= l
    A = torch.randn(l, l)
    u, s, v = A.svd()
    for i in range(l):
        if i >= rank:
            s[i] = 0
        elif s[i] == 0:
            s[i] = 1
    return u.mm(torch.diag(s)).mm(v.transpose(0, 1))


def random_symmetric_matrix(l):
    A = torch.randn(l, l)
    for i in range(l):
        for j in range(i):
            A[i, j] = A[j, i]
    return A


def random_symmetric_psd_matrix(l):
    A = torch.randn(l, l)
    return A.mm(A.transpose(0, 1))


def random_symmetric_pd_matrix(l, eps=1e-5):
    A = torch.randn(l, l)
    return A.mm(A.transpose(0, 1)) + torch.eye(l) * eps


def make_nonzero_det(A, sign=None, min_singular_value=0.1):
    u, s, v = A.svd()
    s[s < min_singular_value] = min_singular_value
    A = u.mm(torch.diag(s)).mm(v.t())
    det = A.det().item()
    if sign is not None:
        if (det < 0) ^ (sign < 0):
            A[0, :].neg_()
    return A


def random_fullrank_matrix_distinct_singular_value(l):
    A = torch.randn(l, l)
    u, _, v = A.svd()
    s = torch.arange(1., l + 1).mul_(1.0 / (l + 1))
    return u.mm(torch.diag(s)).mm(v.t())


def uniform_scalar(offset=0, requires_grad=False):
    v = torch.rand(()) + offset
    v.requires_grad = requires_grad
    return v


def normal_scalar_clamp(amin, amax, requires_grad=False):
    v = torch.randn(()).clamp(amin, amax)
    v.requires_grad = requires_grad
    return v


def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()


class dont_convert(tuple):
    pass


L = 20
M = 10
S = 5

# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name (will be used at test name suffix),    // optional
#   indices for possible dim arg,                            // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
# )
method_tests = [
    ('add', (S, S, S), ((S, S, S),)),
    ('add', (S, S, S), ((S, S),), 'broadcast_rhs'),
    ('add', (S, S), ((S, S, S),), 'broadcast_lhs'),
    ('add', (S, 1, S), ((M, S),), 'broadcast_all'),
    ('add', (), ((),), 'scalar'),
    ('add', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('add', (), ((S, S, S),), 'scalar_broadcast_lhs'),
    ('add', (S, S, S), (3.14,), 'constant'),
    ('add', (), (3.14,), 'scalar_constant'),
    ('__radd__', (S, S, S), (3.14,), 'constant'),
    ('__radd__', (), (3.14,), 'scalar_constant'),
    ('sub', (S, S, S), ((S, S, S),)),
    ('sub', (S, S, S), ((S, S),), 'broadcast_rhs'),
    ('sub', (S, S), ((S, S, S),), 'broadcast_lhs'),
    ('sub', (S, 1, S), ((M, S),), 'broadcast_all'),
    ('sub', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('sub', (), ((S, S, S),), 'scalar_broadcast_lhs'),
    ('sub', (S, S, S), (3.14,), 'constant'),
    ('sub', (), (3.14,), 'scalar_constant'),
    ('__rsub__', (S, S, S), (3.14,), 'constant'),
    ('__rsub__', (), (3.14,), 'scalar_constant'),
    ('mul', (S, S, S), ((S, S, S),)),
    ('mul', (), ((),), 'scalar'),
    ('mul', (S, S, S), ((S, S),), 'broadcast_rhs'),
    ('mul', (S, S), ((S, S, S),), 'broadcast_lhs'),
    ('mul', (S, 1, S), ((M, S),), 'broadcast_all'),
    ('mul', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('mul', (), ((S, S, S),), 'scalar_broadcast_lhs'),
    ('mul', (S, S, S), (3.14,), 'constant'),
    ('mul', (), (3.14,), 'scalar_constant'),
    ('__rmul__', (S, S, S), (3.14,), 'constant'),
    ('__rmul__', (), (3.14,), 'scalar_constant'),
    ('div', (S, S, S), (torch.rand(S, S, S) + 0.1,)),
    ('div', (S, S, S), (torch.rand(S, S) + 0.1,), 'broadcast_rhs'),
    ('div', (S, S), (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs'),
    ('div', (S, 1, S), (torch.rand(M, S) + 0.1,), 'broadcast_all'),
    ('div', (), (uniform_scalar(0.1),), 'scalar'),
    ('div', (S, S, S), (uniform_scalar(0.1),), 'scalar_broadcast_rhs'),
    ('div', (), (uniform_scalar(0.1),), 'scalar_broadcast_lhs'),
    ('div', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant'),
    ('__rdiv__', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant'),
    ('div', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant'),
    ('__rdiv__', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant'),
    ('pow', torch.rand(S, S, S) + 1e-3, (torch.rand(S, S, S) + 0.1,)),
    ('pow', torch.rand(S, S, S) + 1e-3, (torch.rand(1,) + 0.1,), 'broadcast_rhs'),
    ('pow', torch.rand(1,) + 1e-3, (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs'),
    ('pow', torch.rand(S, 1, S) + 1e-3, (torch.rand(1, S, 1) + 0.1,), 'broadcast_all'),
    ('pow', uniform_scalar(1e-3, requires_grad=True), (uniform_scalar(0.1),), 'scalar'),
    ('pow', torch.rand(S, S, S) + 1e-3, (uniform_scalar(0.1),), 'scalar_broadcast_rhs'),
    ('pow', uniform_scalar(1e-3, requires_grad=True), (torch.rand(S, S, S) + 0.1,), 'scalar_broadcast_lhs'),
    ('pow', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant'),
    ('__rpow__', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant'),
    ('pow', uniform_scalar(1e-3, requires_grad=True), (3.14,), 'scalar_constant'),
    ('__rpow__', uniform_scalar(1e-3, requires_grad=True), (3.14,), 'scalar_constant'),
    ('transpose', (1, 2, 3), (1, 2), 'dim', [0, 1]),
    ('transpose', (), (0, 0), 'scalar'),
    ('transpose', (1,), (0, 0), '1d'),
    ('transpose', torch.rand(L, L), (0, 1), '2d'),
    ('transpose', torch.rand(S, S, S), (2, 0), '3d'),
    ('t', (1, 2), NO_ARGS),
    ('view', (S, S, S), (S * S, S),),
    ('view', (S, S, S), (torch.Size([S * S, S]),), 'size'),
    ('view', (S,), (S,), '1d'),
    ('view', (), (dont_convert(()),), 'scalar_to_scalar'),
    ('view', (), (1,), 'scalar_to_1d'),
    ('reshape', (S, S, S), (S * S, S),),
    ('reshape', (S, S, S), (torch.Size([S * S, S]),), 'size'),
    ('reshape', (S,), (S,), '1d'),
    ('reshape', (), (dont_convert(()),), 'scalar_to_scalar'),
    ('reshape', (), (1,), 'scalar_to_1d'),
    ('view_as', (S, S, S), (non_differentiable(torch.rand(S * S, S)),)),
    ('view_as', (), (non_differentiable(torch.tensor(5.5)),), 'scalar'),
    ('view_as', (), (non_differentiable(torch.rand(1, 1)),), 'scalar_to_dims'),
    ('expand', (S, 1, 1), (S, S, S)),
    ('expand', (torch.Size([S, 1, S]),), (S, S, S), 'size'),
    ('expand', (S, 1), (S, S, S), 'new_dim'),
    ('expand', (1,), (S, S, S), '1_element'),
    ('expand', (1, S), (1, 1, S), 'new_dim_front_old_front_1'),
    ('expand', (), (dont_convert(()),), 'scalar_to_scalar'),
    ('expand', (), (1, 3, 2), 'scalar_to_dims'),
    ('exp', (S, S, S), NO_ARGS),
    ('exp', (), NO_ARGS, 'scalar'),
    ('expm1', (S, S, S), NO_ARGS),
    ('expm1', (), NO_ARGS, 'scalar'),
    ('erf', torch.rand(S, S, S), NO_ARGS),
    ('erf', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar'),
    ('erfinv', torch.rand(S, S, S).clamp(-0.9, 0.9), NO_ARGS),
    ('erfinv', normal_scalar_clamp(-0.9, 0.9, requires_grad=True), NO_ARGS, 'scalar'),
    ('log', torch.rand(S, S, S) + 1e-2, NO_ARGS),
    ('log', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
    ('log10', torch.rand(S, S, S) + 1e-2, NO_ARGS),
    ('log10', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
    ('log1p', torch.rand(S, S, S), NO_ARGS),
    ('log1p', uniform_scalar(requires_grad=True), NO_ARGS, 'scalar'),
    ('log2', torch.rand(S, S, S) + 1e-2, NO_ARGS),
    ('log2', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
    ('tanh', (S, S, S), NO_ARGS),
    ('tanh', (), NO_ARGS, 'scalar'),
    ('sigmoid', (S, S, S), NO_ARGS),
    ('sigmoid', (), NO_ARGS, 'scalar'),
    ('sinh', (S, S, S), NO_ARGS),
    ('sinh', (), NO_ARGS, 'scalar'),
    ('cosh', (S, S, S), NO_ARGS),
    ('cosh', (), NO_ARGS, 'scalar'),
    ('abs', (S, S, S), NO_ARGS),
    ('abs', (), NO_ARGS, 'scalar'),
    ('clamp', (S, S, S), (0, 1)),
    ('clamp', (S, S, S), (None, 0.5), 'min'),
    ('clamp', (S, S, S), (0.5, None), 'max'),
    ('clamp', (), (0, 1), 'scalar'),
    ('clamp', (), (None, 0.5), 'min_scalar'),
    ('clamp', (), (0.5, None), 'max_scalar'),
    ('sqrt', torch.rand(S, S, S) + 5e-4, NO_ARGS),
    ('sqrt', uniform_scalar(5e-4, requires_grad=True), NO_ARGS, 'scalar'),
    ('sin', (S, S, S), NO_ARGS),
    ('sin', (), NO_ARGS, 'scalar'),
    ('cos', (S, S, S), NO_ARGS),
    ('cos', (), NO_ARGS, 'scalar'),
    ('tan', torch.randn(S, S, S).clamp(-1, 1), NO_ARGS),
    ('asin', torch.randn(S, S, S).clamp(-0.9, 0.9), NO_ARGS),
    ('acos', torch.randn(S, S, S).clamp(-0.9, 0.9), NO_ARGS),
    ('atan', (S, S, S), NO_ARGS),
    ('atan', (), NO_ARGS, 'scalar'),
    ('atan2', (S, S, S), ((S, S, S),)),
    ('atan2', (), ((),), 'scalar'),
    ('reciprocal', torch.rand(S, S, S) + 0.1, NO_ARGS),
    ('reciprocal', uniform_scalar(0.1, requires_grad=True), NO_ARGS, 'scalar'),
    ('round', (S, S, S), NO_ARGS),
    ('round', (), NO_ARGS, 'scalar'),
    ('sign', (S, S, S), NO_ARGS),
    ('sign', (), NO_ARGS, 'scalar'),
    ('trunc', (S, S, S), NO_ARGS),
    ('trunc', (), NO_ARGS, 'scalar'),
    ('floor', (S, S, S), NO_ARGS),
    ('floor', (), NO_ARGS, 'scalar'),
    ('ceil', (S, S, S), NO_ARGS),
    ('ceil', (), NO_ARGS, 'scalar'),
    ('rsqrt', torch.rand(S, S, S) + 1e-2, NO_ARGS),
    ('rsqrt', uniform_scalar(1e-2, requires_grad=True), NO_ARGS, 'scalar'),
    ('frac', (S, S, S), NO_ARGS),
    ('frac', (), NO_ARGS, 'scalar'),
    ('fmod', (S, S, S), (1.5,)),
    ('fmod', (), (1.5,), 'scalar'),
    ('fmod', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
    ('fmod', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
    ('fmod', (S, S, S), (non_differentiable(torch.rand(S) + 1.5),), 'tensor_broadcast_rhs'),
    ('fmod', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
    ('fmod', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
    ('fmod', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
    ('fmod', (S, S, S), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor_broadcast_rhs'),
    ('remainder', (S, S, S), (1.5,)),
    ('remainder', (), (1.5,), 'scalar'),
    ('remainder', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
    ('remainder', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
    ('remainder', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
    ('remainder', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
    ('remainder', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
    ('lerp', (S, S, S), ((S, S, S), 0.4)),
    ('lerp', (S, S, S), ((S,), 0.4), 'broadcast_rhs'),
    ('lerp', (S,), ((S, S, S), 0.4), 'broadcast_lhs'),
    ('lerp', (S, 1, S), ((S, S), 0.4), 'broadcast_all'),
    ('lerp', (), ((), 0.4), 'scalar'),
    ('lerp', (S, S, S), ((), 0.4), 'scalar_broadcast_rhs'),
    ('lerp', (), ((S, S, S), 0.4), 'scalar_broadcast_lhs'),
    ('max', (S, S, S), NO_ARGS),
    ('max', (S, S, S), (1,), 'dim', [0]),
    ('max', (S, S, S), (1, True,), 'keepdim_dim', [0]),
    ('max', (), NO_ARGS, 'scalar'),
    ('max', (), (0,), 'scalar_dim', [0]),
    ('max', (), (0, True,), 'scalar_keepdim_dim', [0]),
    ('max', (S, S, S), ((S, S, S),), 'elementwise'),
    ('max', (S, S, S), ((S,),), 'elementwise_broadcast_rhs'),
    ('max', (S,), ((S, S, S),), 'elementwise_broadcast_lhs'),
    ('max', (S, 1, S), ((S, S),), 'elementwise_broadcast_all'),
    ('max', (), ((),), 'scalar_elementwise'),
    ('max', (S, S, S), ((),), 'scalar_elementwise_broadcast_rhs'),
    ('max', (), ((S, S, S),), 'scalar_elementwise_broadcast_lhs'),
    ('min', (S, S, S), NO_ARGS),
    ('min', (S, S, S), (1,), 'dim', [0]),
    ('min', (S, S, S), (1, True,), 'keepdim_dim', [0]),
    ('min', (), NO_ARGS, 'scalar'),
    ('min', (), (0,), 'scalar_dim', [0]),
    ('min', (), (0, True,), 'scalar_keepdim_dim', [0]),
    ('min', (S, S, S), ((S, S, S),), 'elementwise'),
    ('min', (S, S, S), ((S,),), 'elementwise_broadcast_rhs'),
    ('min', (S,), ((S, S, S),), 'elementwise_broadcast_lhs'),
    ('min', (S, 1, S), ((S, S),), 'elementwise_broadcast_all'),
    ('min', (), ((),), 'scalar_elementwise'),
    ('min', (S, S, S), ((),), 'scalar_elementwise_broadcast_rhs'),
    ('min', (), ((S, S, S),), 'scalar_elementwise_broadcast_lhs'),
    ('mean', (S, S, S), NO_ARGS),
    ('mean', (S, S, S), (1,), 'dim', [0]),
    ('mean', (S, S, S), (1, True,), 'keepdim_dim', [0]),
    ('mean', (), NO_ARGS, 'scalar'),
    ('mean', (), (0,), 'scalar_dim', [0]),
    ('mean', (), (0, True,), 'scalar_keepdim_dim', [0]),
    ('kthvalue', (S, S, S), (2,)),
    ('kthvalue', (), (1,), 'scalar'),
    ('kthvalue', (S, S, S), (2, 1,), 'dim', [1]),
    ('kthvalue', (), (1, 0,), 'scalar_dim', [1]),
    ('kthvalue', (S, S, S), (2, 1, True,), 'keepdim_dim', [1]),
    ('kthvalue', (), (1, 0, True), 'scalar_keepdim_dim', [1]),
    ('kthvalue', (S,), (2, 0,), 'dim_1d', [1]),
    ('kthvalue', (S,), (2, 0, True,), 'keepdim_dim_1d', [1]),
    ('median', (S, S, S), NO_ARGS),
    ('median', (S, S, S), (1,), 'dim', [0]),
    ('median', (S, S, S), (1, True,), 'keepdim_dim', [0]),
    ('median', (), NO_ARGS, 'scalar'),
    ('median', (), (0,), 'scalar_dim', [0]),
    ('median', (), (0, True,), 'scalar_keepdim_dim', [0]),
    ('mode', (S, S, S), NO_ARGS),
    ('mode', (S, S, S), (1,), 'dim', [0]),
    ('mode', (S, S, S), (1, True,), 'keepdim_dim', [0]),
    ('mode', (), NO_ARGS, 'scalar'),
    ('mode', (), (0,), 'scalar_dim', [0]),
    ('mode', (), (0, True,), 'scalar_keepdim_dim', [0]),
    ('sum', (S, S, S), NO_ARGS),
    ('sum', (S, S, S), (1,), 'dim', [0]),
    ('sum', (S, S, S), (1, True,), 'keepdim_dim', [0]),
    ('sum', (), NO_ARGS, 'scalar'),
    ('sum', (), (0,), 'scalar_dim', [0]),
    ('sum', (), (0, True,), 'scalar_keepdim_dim', [0]),
    ('sum', (S, S, S), ([1, 2],), 'multi_dim'),
    ('sum', (S, S, S), ([1, 2], True,), 'multi_dim_keepdim'),
    ('prod', (S, S, S), NO_ARGS),
    ('prod', (S, S, S), (1,), 'dim', [0]),
    ('prod', (S, S, S), (1, True,), 'keepdim_dim', [0]),
    ('prod', (), NO_ARGS, 'scalar'),
    ('prod', (), (0,), 'scalar_dim', [0]),
    ('prod', (), (0, True,), 'scalar_keepdim_dim', [0]),
    ('prod', prod_zeros(S, [0, 1]), NO_ARGS, 'zerodims2'),
    ('prod', prod_zeros(S, [0, 2]), NO_ARGS, 'zerodims1'),
    ('prod', prod_zeros(S, [1, 2]), NO_ARGS, 'zerodims0'),
    ('prod', prod_zeros(S, [0, 1]), (1,), 'zeros_dims2', [0]),
    ('prod', prod_zeros(S, [0, 2]), (1,), 'zeros_dims1', [0]),
    ('prod', prod_zeros(S, [1, 2]), (1,), 'zeros_dims0', [0]),
    ('prod', prod_zeros(S, [0, 1]), (1, True), 'keepdim_zeros_dims2', [0]),
    ('prod', prod_zeros(S, [0, 2]), (1, True), 'keepdim_zeros_dims1', [0]),
    ('prod', prod_zeros(S, [1, 2]), (1, True), 'keepdim_zeros_dims0', [0]),
    ('prod', prod_single_zero(S), NO_ARGS, 'single_zero'),
    ('prod', (torch.tensor(0., requires_grad=True)), NO_ARGS, 'scalar_zero'),
    ('prod', (torch.tensor(0., requires_grad=True)), (0,), 'scalar_dim_zero', [0]),
    ('prod', (torch.tensor(0., requires_grad=True)), (0, True,), 'scalar_keepdim_dim_zero', [0]),
    ('var', (S, S, S), NO_ARGS),
    ('var', (S, S, S), (1,), 'dim', [0]),
    ('var', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
    ('var', (S,), (0,), 'dim_1d', [0]),
    ('var', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
    ('std', (S, S, S), NO_ARGS),
    ('std', (S, S, S), (1,), 'dim', [0]),
    ('std', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
    ('std', (S,), (0,), 'dim_1d', [0]),
    ('std', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
    ('renorm', (S, S, S), (2, 1, 0.5), 'dim', [1]),
    ('renorm', (S, S, S), (1, 2, 3), 'norm_1'),
    ('renorm', (S, S, S), (float('inf'), 2, 0.5), 'norm_inf'),
    ('repeat', (S,), (2,), 'single_number'),
    ('repeat', (), (2, 3), 'scalar'),
    ('repeat', (2, 2), (3, 2)),
    ('repeat', (2, 2), (1, 3, 1, 2), 'unsqueeze'),
    ('cumsum', (S, S, S), (0,), 'dim0', [0]),
    ('cumsum', (S, S, S), (1,), 'dim1', [0]),
    ('cumsum', (), (0,), 'dim0_scalar', [0]),
    ('cumprod', (S, S, S), (0,)),
    ('cumprod', (S, S, S), (1,), 'dim1', [0]),
    ('cumprod', (), (0,), 'scalar'),
    ('cumprod', (torch.tensor(0., requires_grad=True)), (0,), 'scalar_zeros'),
    ('cumprod', prod_zeros(S, [0, 1]), (1,), 'zeros_dim2', [0]),
    ('cumprod', prod_zeros(S, [0, 2]), (1,), 'zeros_dim1', [0]),
    ('cumprod', prod_zeros(S, [1, 2]), (1,), 'zeros_dim0', [0]),
    ('unfold', (), (0, 1, 1), 'scalar', [0]),
    ('unfold', (S, S, S, S), (1, 3, 1), '', [0]),
    ('unfold', (S, S, S), (2, 3, 2), 'lastdim', [0]),
    ('addmm', (S, M), ((S, S), (S, M)),),
    ('addmm', (1,), ((S, S), (S, M)), 'broadcast_lhs'),
    ('addmm', (S, M), (0.2, 0.6, (S, S), (S, M)), 'coef'),
    ('addmm', (1,), (0.2, 0.6, (S, S), (S, M)), 'broadcast_lhs_coef'),
    ('addmm', (), ((S, S), (S, M)), 'scalar_broadcast_lhs'),
    ('addmm', (), (0.2, 0.6, (S, S), (S, M)), 'scalar_broadcast_lhs_coef'),
    ('addbmm', (S, M), ((S, S, S), (S, S, M)),),
    ('addbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs'),
    ('addbmm', (S, M), (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'),
    ('addbmm', (1,), (0.2, 0.6, (S, S, S), (S, S, M)), 'broadcast_lhs_coef'),
    ('addbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs'),
    ('addbmm', (), (0.2, 0.6, (S, S, S), (S, S, M)), 'scalar_broadcast_lhs_coef'),
    ('baddbmm', (S, S, M), ((S, S, S), (S, S, M)),),
    ('baddbmm', (1,), ((S, S, S), (S, S, M)), 'broadcast_lhs'),
    ('baddbmm', (S, S, M), (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'),
    ('baddbmm', (1,), (0.2, 0.6, (S, S, S), (S, S, M)), 'broadcast_lhs_coef'),
    ('baddbmm', (), ((S, S, S), (S, S, M)), 'scalar_broadcast_lhs'),
    ('baddbmm', (), (0.2, 0.6, (S, S, S), (S, S, M)), 'scalar_broadcast_lhs_coef'),
    ('addmv', (S,), ((S, M), (M,)),),
    ('addmv', (1,), ((S, M), (M,)), 'broadcast_lhs'),
    ('addmv', (S,), (0.2, 0.6, (S, M), (M,)), 'coef'),
    ('addmv', (1,), (0.2, 0.6, (S, M), (M,)), 'broadcast_lhs_coef'),
    ('addmv', (), ((S, M), (M,)), 'scalar_broadcast_lhs'),
    ('addmv', (), (0.2, 0.6, (S, M), (M,)), 'scalar_broadcast_lhs_coef'),
    ('addr', (S, M), ((S,), (M,)),),
    ('addr', (), ((S,), (M,)), 'broadcast_lhs'),
    ('addr', (S, M), (0.2, 0.6, (S,), (M,)), 'coef'),
    ('addr', (), (0.2, 0.6, (S,), (M,)), 'broadcast_lhs_coef'),
    ('dot', (L,), ((L,),),),
    ('mm', (S, M), ((M, S),)),
    ('bmm', (M, S, M), ((M, M, S),)),
    ('mv', (S, M), ((M,),)),
    ('ger', (S,), ((M,),)),
    ('matmul', (L,), ((L,),),),
    ('matmul', (S, M), ((M,),), "2d_1d"),
    ('matmul', (M, ), ((M, S),), "1d_2d"),
    ('matmul', (S, M), ((M, S),), "2d_2d"),
    ('matmul', (S, S, M, M), ((S, S, M, S),), "4d_4d"),
    ('matmul', (S, S, M, M), ((M,),), "4d_1d"),
    ('matmul', (M,), ((S, S, M, S),), "1d_4d"),
    ('addcmul', (S, S), ((S, S), (S, S))),
    ('addcmul', (S, S), ((S, 1), (1, S)), 'broadcast_rhs'),
    ('addcmul', (1,), ((S, S, 1), (1, S)), 'broadcast_all'),
    ('addcmul', (S, S), (0.5, (S, S), (S, S)), 'scale'),
    ('addcmul', (S, S), (0.5, (S, 1), (1, S)), 'scale_broadcast_rhs'),
    ('addcmul', (1,), (0.5, (S, S, 1), (1, S)), 'scale_broadcast_all'),
    ('addcmul', (), ((), ()), 'scalar'),
    ('addcmul', (S, S), ((), ()), 'scalar_broadcast_rhs'),
    ('addcmul', (), ((S, S, 1), (1, S)), 'scalar_broadcast_lhs'),
    ('addcmul', (), (0.5, (), ()), 'scalar_scale'),
    ('addcmul', (S, S), (0.5, (), ()), 'scalar_scale_broadcast_rhs'),
    ('addcmul', (), (0.5, (S, S, 1), (1, S)), 'scalar_scale_broadcast_lhs'),
    ('addcdiv', (S, S), ((S, S), (S, S))),
    ('addcdiv', (S, S), ((S, 1), (1, S)), 'broadcast_rhs'),
    ('addcdiv', (1,), ((S, S, 1), (1, S)), 'broadcast_all'),
    ('addcdiv', (S, S), (0.5, (S, S), (S, S)), 'scale'),
    ('addcdiv', (S, S), (0.5, (S, 1), (1, S)), 'scale_broadcast_rhs'),
    ('addcdiv', (1,), (0.5, (S, S, 1), (1, S)), 'scale_broadcast_all'),
    ('addcdiv', (), ((), ()), 'scalar'),
    ('addcdiv', (S, S), ((), ()), 'scalar_broadcast_rhs'),
    ('addcdiv', (), ((S, S, 1), (1, S)), 'scalar_broadcast_lhs'),
    ('addcdiv', (), (0.5, (), ()), 'scalar_scale'),
    ('addcdiv', (S, S), (0.5, (), ()), 'scalar_scale_broadcast_rhs'),
    ('addcdiv', (), (0.5, (S, S, 1), (1, S)), 'scalar_scale_broadcast_lhs'),
    ('zero_', (S, S, S), NO_ARGS),
    ('zero_', (), NO_ARGS, 'scalar'),
    ('norm', (S, S), (2,)),
    ('norm', (S, S), (0,), '0'),
    ('norm', (S, S), (0.5,), '0_5'),
    ('norm', (S, S), (1,), '1'),
    ('norm', (S, S), (3,), '3'),
    ('norm', (S, S), (float('inf'),), 'inf'),
    ('norm', (S, S), (-1,), 'neg_1'),
    ('norm', (S, S), (-0.5,), 'neg_0_5'),
    ('norm', (S, S), (-1.5,), 'neg_1_5'),
    ('norm', torch.rand(S, S, S) + 5e-2, (1.5,), '1_5'),
    ('norm', (S, S, S), (2, 1), '2_dim', [1]),
    ('norm', (S, S, S), (3, 1), '3_dim', [1]),
    ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1), '1_5_dim', [1]),
    ('norm', (S, S, S), (2, 1, True), 'keepdim_2_dim', [1]),
    ('norm', (S, S, S), (3, 1, True), 'keepdim_3_dim', [1]),
    ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1, True), 'keepdim_1_5_dim', [1]),
    ('norm', (), (2, 0), '2_dim_scalar', [1]),
    ('norm', (), (3, 0), '3_dim_scalar', [1]),
    ('norm', (), (2, 0, True), 'keepdim_2_dim_scalar', [1]),
    ('norm', (), (3, 0, True), 'keepdim_3_dim_scalar', [1]),
    ('clone', (S, M, S), NO_ARGS),
    ('clone', (), NO_ARGS, 'scalar'),
    ('dist', (S, S, S), ((S, S, S),)),
    ('dist', (S, S, S), ((S,),), 'broadcast_rhs'),
    ('dist', (S,), ((S, S, S),), 'broadcast_lhs'),
    ('dist', (S, 1, S), ((S, S),), 'broadcast_all'),
    ('dist', (), ((),), 'scalar'),
    ('dist', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('dist', (), ((S, S, S),), 'scalar_broadcast_lhs'),
    ('dist', (S, S, S), ((S, S, S), 4), '4'),
    ('dist', (S, S, S), ((S,), 4), '4_broadcast_rhs'),
    ('dist', (S,), ((S, S, S), 4), '4_broadcast_lhs'),
    ('dist', (S, 1, S), ((S, S), 4), '4_broadcast_all'),
    ('dist', (), ((), 4), 'scalar_4'),
    ('dist', (S, S, S), ((), 4), 'scalar_4_broadcast_rhs'),
    ('dist', (), ((S, S, S), 4), 'scalar_4_broadcast_lhs'),
    ('diag', (M, M), NO_ARGS, '2d'),
    ('diag', (3, 5), NO_ARGS, '2d_wide'),
    ('diag', (3, 5), (2,), '2d_wide_pos'),
    ('diag', (3, 5), (-2,), '2d_wide_neg'),
    ('diag', (5, 3), NO_ARGS, '2d_tall'),
    ('diag', (5, 3), (2,), '2d_tall_pos'),
    ('diag', (5, 3), (-2,), '2d_tall_neg'),
    ('diag', (M,), NO_ARGS, '1d'),
    ('diag', (M, M), (1,), '2d_1'),
    ('diag', (M, M), (2,), '2d_2'),
    ('diagonal', (M, M), NO_ARGS, '2d'),
    ('diagonal', (3, 5), NO_ARGS, '2d_wide'),
    ('diagonal', (3, 5), (2,), '2d_wide_pos'),
    ('diagonal', (3, 5), (-2,), '2d_wide_neg'),
    ('diagonal', (5, 3), NO_ARGS, '2d_tall'),
    ('diagonal', (5, 3), (2,), '2d_tall_pos'),
    ('diagonal', (5, 3), (-2,), '2d_tall_neg'),
    ('diagonal', (M, M), (1,), '2d_1'),
    ('diagonal', (M, M), (2,), '2d_2'),
    ('diagonal', (M, M, M), (1, 1, 2), '3d_1'),
    ('diagonal', (M, M, M), (2, 0, 1), '3d_2'),
    ('diagonal', (M, M, M), (-2, 0, 1), '3d_3'),
    ('tril', (M, M), NO_ARGS),
    ('tril', (M, M), (2,), 'idx'),
    ('triu', (M, M), NO_ARGS),
    ('triu', (M, M), (2,), 'idx'),
    ('trace', (M, M), NO_ARGS),
    ('cross', (S, 3), ((S, 3),)),
    ('cross', (S, 3, S), ((S, 3, S), 1), 'dim'),
    ('index_select', (S, S, S), (0, index_variable(2, S)), 'dim', [0]),
    ('index_select', (), (0, torch.tensor([0], dtype=torch.int64)), 'scalar_mixed_dim', [0]),
    ('index_select', (), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_dim', [0]),
    ('index_add', (S, S), (0, index_variable(2, S), (2, S)), 'dim', [0]),
    ('index_add', (), (0, torch.tensor([0], dtype=torch.int64), torch.tensor([2.])), 'scalar_input_dim', [0]),
    ('index_add', (), (0, torch.tensor(0, dtype=torch.int64), torch.tensor(2.)), 'scalar_all_dim', [0]),
    ('index_copy', (S, S), (0, index_perm_variable(2, S), (2, S)), 'dim', [0]),
    ('index_copy', (), (0, torch.tensor([0], dtype=torch.int64), torch.tensor([2.])), 'scalar_input_dim', [0]),
    ('index_copy', (), (0, torch.tensor(0, dtype=torch.int64), torch.tensor(2.)), 'scalar_all_dim', [0]),
    ('index_fill', (S, S), (0, index_variable(2, S), 2), 'dim', [0]),
    # FIXME: we should compute the derivative w.r.t torch.tensor(2)
    ('index_fill', (S, S), (0, index_variable(2, S), non_differentiable(torch.tensor(2))),
     'variable_dim', [0]),
    ('index_fill', (S, S), (0, torch.tensor(0, dtype=torch.int64), 2), 'scalar_index_dim', [0]),
    ('index_fill', (), (0, torch.tensor([0], dtype=torch.int64), 2), 'scalar_input_dim', [0]),
    ('index_fill', (), (0, torch.tensor(0, dtype=torch.int64), 2), 'scalar_both_dim', [0]),
    ('inverse', (S, S), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
    ('det', (S, S), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
    ('det', (1, 1), NO_ARGS, '1x1', NO_ARGS, [skipIfNoLapack]),
    ('det', lambda: random_symmetric_matrix(S), NO_ARGS, 'symmetric', NO_ARGS, [skipIfNoLapack]),
    ('det', lambda: random_symmetric_psd_matrix(S), NO_ARGS, 'symmetric_psd', NO_ARGS, [skipIfNoLapack]),
    ('det', lambda: random_symmetric_pd_matrix(S), NO_ARGS, 'symmetric_pd', NO_ARGS, [skipIfNoLapack]),
    ('det', lambda: random_square_matrix_of_rank(S, S - 2), NO_ARGS, 'dim2_null', NO_ARGS, [skipIfNoLapack]),
    ('det', lambda: random_square_matrix_of_rank(S, 1), NO_ARGS, 'rank1', NO_ARGS, [skipIfNoLapack]),
    ('det', lambda: random_square_matrix_of_rank(S, 2), NO_ARGS, 'rank2', NO_ARGS, [skipIfNoLapack]),
    ('det', lambda: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS,
     'distinct_singular_values', NO_ARGS, [skipIfNoLapack]),
    # For `logdet` and `slogdet`, the function at det=0 is not smooth.
    # We need to exclude tests with det=0 (e.g. dim2_null, rank1, rank2) and use
    # `make_nonzero_det` to make the random matrices have nonzero det. For
    # `logdet`, we also set `make_nonzero_det(matrix, sign=1)` to make the
    # matrix have positive det.
    ('logdet', lambda: make_nonzero_det(torch.randn(S, S), 1), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
    ('logdet', lambda: make_nonzero_det(torch.randn(1, 1), 1), NO_ARGS, '1x1', NO_ARGS, [skipIfNoLapack]),
    ('logdet', lambda: make_nonzero_det(random_symmetric_matrix(S), 1), NO_ARGS,
     'symmetric', NO_ARGS, [skipIfNoLapack]),
    ('logdet', lambda: make_nonzero_det(random_symmetric_pd_matrix(S), 1), NO_ARGS,
     'symmetric_pd', NO_ARGS, [skipIfNoLapack]),
    ('logdet', lambda: make_nonzero_det(random_fullrank_matrix_distinct_singular_value(S), 1, 0), NO_ARGS,
     'distinct_singular_values', NO_ARGS, [skipIfNoLapack]),
    ('slogdet', lambda: make_nonzero_det(torch.randn(1, 1), 1), NO_ARGS,
     '1x1_pos_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
    ('slogdet', lambda: make_nonzero_det(torch.randn(1, 1), -1), NO_ARGS,
     '1x1_neg_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
    ('slogdet', lambda: make_nonzero_det(torch.randn(S, S), 1), NO_ARGS,
     'pos_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
    ('slogdet', lambda: make_nonzero_det(torch.randn(S, S), -1), NO_ARGS,
     'neg_det', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
    ('slogdet', lambda: make_nonzero_det(random_symmetric_matrix(S)), NO_ARGS,
     'symmetric', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
    ('slogdet', lambda: random_symmetric_pd_matrix(S), NO_ARGS,
     'symmetric_pd', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
    ('slogdet', lambda: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS,
     'distinct_singular_values', NO_ARGS, [skipIfNoLapack], itemgetter(1)),
    ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S), NO_ARGS, '', NO_ARGS, [skipIfNoLapack]),
    ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:(S - 2)], NO_ARGS,
     'wide', NO_ARGS, [skipIfNoLapack]),
    ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:, :(S - 2)], NO_ARGS,
     'tall', NO_ARGS, [skipIfNoLapack]),
    ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:(S - 2)], (False,),
     'wide_all', NO_ARGS, [skipIfNoLapack], lambda usv: (usv[0], usv[1], usv[2][:, :(S - 2)])),
    ('svd', lambda: random_fullrank_matrix_distinct_singular_value(S)[:, :(S - 2)], (False,),
     'tall_all', NO_ARGS, [skipIfNoLapack], lambda usv: (usv[0][:, :(S - 2)], usv[1], usv[2])),
    ('svd', lambda: random_fullrank_matrix_distinct_singular_value(M), NO_ARGS,
     'large', NO_ARGS, [skipIfNoLapack]),
    ('gesv', (S, S), ((S, S),), '', NO_ARGS, [skipIfNoLapack]),
    ('gesv', (S, S, S), ((S, S, S),), 'batched', NO_ARGS, [skipIfNoLapack]),
    ('gesv', (2, 3, S, S), ((2, 3, S, S),), 'batched_dims', NO_ARGS, [skipIfNoLapack]),
    ('gesv', (2, 2, S, S), ((1, S, S),), 'batched_broadcast_A', NO_ARGS, [skipIfNoLapack]),
    ('gesv', (1, S, S), ((2, 2, S, S),), 'batched_broadcast_b', NO_ARGS, [skipIfNoLapack]),
    ('fill_', (S, S, S), (1,), 'number'),
    ('fill_', (), (1,), 'number_scalar'),
    # FIXME: we should compute the derivative w.r.t torch.tensor(1)
    ('fill_', (S, S, S), (non_differentiable(torch.tensor(1)),), 'variable'),
    ('eq_', (S, S, S), ((S, S, S),)),
    ('eq_', (S, S, S), ((1,),), 'broadcast_rhs'),
    ('eq_', (), ((),), 'scalar'),
    ('eq_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('ne_', (S, S, S), ((S, S, S),)),
    ('ne_', (S, S, S), ((1,),), 'broadcast_rhs'),
    ('ne_', (), ((),), 'scalar'),
    ('ne_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('gt_', (S, S, S), ((S, S, S),)),
    ('gt_', (S, S, S), ((1,),), 'broadcast_rhs'),
    ('gt_', (), ((),), 'scalar'),
    ('gt_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('ge_', (S, S, S), ((S, S, S),)),
    ('ge_', (S, S, S), ((1,),), 'broadcast_rhs'),
    ('ge_', (), ((),), 'scalar'),
    ('ge_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('lt_', (S, S, S), ((S, S, S),)),
    ('lt_', (S, S, S), ((1,),), 'broadcast_rhs'),
    ('lt_', (), ((),), 'scalar'),
    ('lt_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('le_', (S, S, S), ((S, S, S),)),
    ('le_', (S, S, S), ((1,),), 'broadcast_rhs'),
    ('le_', (), ((),), 'scalar'),
    ('le_', (S, S, S), ((),), 'scalar_broadcast_rhs'),
    ('eq_', (S, S, S), (0,), 'pyscalar'),
    ('ne_', (S, S, S), (0,), 'pyscalar'),
    ('gt_', (S, S, S), (0,), 'pyscalar'),
    ('ge_', (S, S, S), (0,), 'pyscalar'),
    ('le_', (S, S, S), (0,), 'pyscalar'),
    ('lt_', (), (0,), 'pyscalar'),
    ('eq_', (), (0,), 'pyscalar_scalar'),
    ('ne_', (), (0,), 'pyscalar_scalar'),
    ('gt_', (), (0,), 'pyscalar_scalar'),
    ('ge_', (), (0,), 'pyscalar_scalar'),
    ('lt_', (), (0,), 'pyscalar_scalar'),
    ('le_', (), (0,), 'pyscalar_scalar'),
    ('permute', (1, 2, 3, 4), (0, 2, 3, 1)),
    ('permute', (1, 2, 3, 4), (0, -2, -1, 1), 'neg_dim'),
    ('permute', (), (dont_convert(()),), 'scalar'),
    ('select', (S, S, S), (1, 2), 'dim', [0]),
    ('select', (S,), (0, 2), '1d'),
    ('narrow', (S, S, S), (1, 2, 2), 'dim', [0]),
    ('slice', (S, S, S), (-2, 1, -1, 2)),
    ('squeeze', (S, 1, S, 1), NO_ARGS),
    ('squeeze', (1, 1, 1, 1), NO_ARGS, 'input_sizes_are_ones'),
    ('squeeze', (S, 1, S, 1), (1,), '1_dim', [0]),
    ('squeeze', (S, 1, S, 1), (2,), 'not_1_dim', [0]),
    ('squeeze', (), (0,), 'scalar', [0]),
    ('unsqueeze', (S, S, S), (0,), 'first', [0]),
    ('unsqueeze', (S, S, S), (1,), 'middle', [0]),
    ('unsqueeze', (S, S, S), (3,), 'last', [0]),
    ('unsqueeze', (), (0,), 'scalar', [0]),
    ('chunk', (S, S, S), (2,)),
    ('chunk', (S, S, S), (S, 1), 'dim', [1]),
    ('split', (S, S, S), (2,)),
    ('split', (S, S, S), (S, 1), 'dim', [1]),
    ('split', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), 'size_list'),
    ('split', (S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], 2), 'size_list_dim', [1]),
    ('gather', (M, S), (0, gather_variable((S, S), 1, M, True)), 'dim0', [0]),
    ('gather', (M, S), (1, gather_variable((M, S // 2), 0, S, True)), 'dim1', [0]),
    ('gather', (), (0, torch.tensor([0], dtype=torch.int64)), 'scalar_input', [0]),
    ('gather', (S,), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_index', [0]),
    ('gather', (), (0, torch.tensor(0, dtype=torch.int64)), 'scalar_both', [0]),
    ('scatter', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', [0]),
    ('scatter', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', [0]),
    ('scatter', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim0', [0]),
    ('scatter_add', (M, S), (0, gather_variable((S, S), 1, M), (S, S)), 'dim0', [0]),
    ('scatter_add', (M, S), (1, gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1', [0]),
    ('scatter_add', (), (0, torch.tensor(0, dtype=torch.int64), ()), 'scalar_all_dim0', [0]),
    ('masked_select', (M, M), (mask_not_all_zeros((M, M)),)),
    ('masked_select', (M, M), (mask_not_all_zeros((M,)),), 'broadcast_rhs'),
    ('masked_select', (M,), (mask_not_all_zeros((M, M)),), 'broadcast_lhs'),
    ('masked_select', (M, 1, M), (mask_not_all_zeros((M, M)),),
     'broadcast_all'),
    ('masked_select', (), (torch.tensor(1, dtype=torch.uint8),), 'scalar'),
    ('masked_select', (M, M), (torch.tensor(1, dtype=torch.uint8),), 'scalar_broadcast_rhs'),
    ('masked_select', (), (mask_not_all_zeros((M, M)),), 'scalar_broadcast_lhs'),
    ('masked_fill', (M, M), (torch.ByteTensor(M, M).bernoulli_(), 10)),
    ('masked_fill', (M, M), (torch.ByteTensor(M, M).bernoulli_(), torch.tensor(10)), 'tensor'),
    # no lhs or all broadcast on masked_fill or masked_scatter because it's always inplace
    ('masked_fill', (M, M), (torch.ByteTensor(M,).bernoulli_(), 10), 'broadcast_rhs'),
    ('masked_fill', (), (torch.tensor(0, dtype=torch.uint8, requires_grad=False).bernoulli_(), 10), 'scalar'),
    ('masked_fill', (), (torch.tensor(0, dtype=torch.uint8, requires_grad=False).bernoulli_(), torch.tensor(10)),
     'scalar_variable'),
    ('masked_fill', (M, M), (torch.tensor(0, dtype=torch.uint8, requires_grad=False).bernoulli_(), 10),
     'scalar_broadcast_rhs'),
    ('masked_scatter', (M, M), (torch.ByteTensor(M, M).bernoulli_(), (M, M))),
    ('masked_scatter', (M, M), (torch.ByteTensor(M,).bernoulli_(), (M, M)),
     'broadcast_rhs'),
    ('masked_scatter', (M, M), (bernoulli_scalar(), (M, M)), 'scalar'),
    ('masked_scatter', (M, M), (bernoulli_scalar(), (M, M)),
     'scalar_broadcast_rhs'),
    ('resize_', (S, S, S), (torch.Size([S * S, S])), 'fewer_dims'),
    ('resize_', (), (dont_convert(()),), 'scalar'),
    ('resize_', (), (torch.Size([1, 1, 1])), 'scalar_to_dims'),
    ('resize_as_', (), (non_differentiable(torch.tensor(5.)),), 'scalar'),
    ('resize_as_', (), (non_differentiable(torch.randn((1, 1, 1))),), 'scalar_to_dims'),
    ('resize_as_', (S, S, S), (non_differentiable(torch.randn(S * S, S)),)),
    ('sort', (S, M, S), NO_ARGS),
    ('sort', (S, M, S), (1,), 'dim'),
    ('sort', (S, M, S), (1, True), 'dim_desc'),
    ('sort', (), NO_ARGS, 'scalar'),
    ('sort', (), (0,), 'dim_scalar'),
    ('sort', (), (0, True), 'dim_desc_scalar'),
    ('topk', (S, M, S), (3,)),
    ('topk', (S, M, S), (3, 1), 'dim', [1]),
    ('topk', (S, M, S), (3, 1, True), 'dim_desc', [1]),
    ('topk', (S, M, S), (3, 1, True, True), 'dim_desc_sort', [1]),
    ('topk', (), (1,), 'scalar'),
    ('topk', (), (1, 0), 'dim_sclar', [1]),
    ('topk', (), (1, 0, True), 'dim_desc_scalar', [1]),
    ('topk', (), (1, 0, True, True), 'dim_desc_sort_scalar', [1]),
    ('take', (S, S, S), (torch.LongTensor([[-3, 2], [20, 2]]),)),
    ('take', (S, S, S), (torch.tensor(0, dtype=torch.int64),), 'scalar_index'),
    ('take', (), (torch.LongTensor([0]),), 'scalar_data'),
    ('take', (), (torch.tensor(0, dtype=torch.int64),), 'scalar_both'),
    ('where', (M, M), (mask_not_all_zeros((M, M)), (M, M))),
    ('where', (M, 1, M), (mask_not_all_zeros((M, M)), (M, M, 1)), 'broadcast_all'),
    ('where', (), (bernoulli_scalar(), ()), 'scalar'),
    ('where', (M, 1, M), (bernoulli_scalar(), (M, M, 1)), 'scalar_broadcast_mask'),
    ('where', (), (mask_not_all_zeros((M, M)), ()), 'scalar_broadcast_non_mask'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([1, 2]),)),
    ('__getitem__', torch.randn(S, S, S), (slice(0, 3),), 'slice'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(0, 3), 1]),), 'slice_index'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 2, 3], [1, 3, 3], [0, 0, 2]]),), 'adv_index'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 0, 3], [1, 1, 3], [0, 0, 2]]),), 'adv_index_dup'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(None), slice(None), [0, 3]]),), 'adv_index_end'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([slice(None), [0, 3], slice(None)]),), 'adv_index_mid'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], slice(None), slice(None)]),), 'adv_index_beg'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], [1, 2], slice(None)]),), 'adv_index_comb'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], ]),), 'adv_index_sub'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], slice(None)]),), 'adv_index_sub_2'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 3], Ellipsis]),), 'adv_index_sub_3'),
    ('__getitem__', torch.randn(S, S, S), (dont_convert([[0, 2, 3], [1, 3, 3],
     torch.LongTensor([0, 0, 2])]),), 'adv_index_var'),
]
# TODO: clamp with min/max


def create_input(call_args, requires_grad=True, non_contiguous=False):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        def maybe_non_contig(tensor):
            return tensor if not non_contiguous else make_non_contiguous(tensor)

        if isinstance(arg, torch.Size) or isinstance(arg, dont_convert):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = torch.randn((), dtype=torch.double)
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            return Variable(maybe_non_contig(torch.randn(*arg, dtype=torch.double)), requires_grad=requires_grad)
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                return maybe_non_contig(arg.tensor)
            return maybe_non_contig(arg.tensor)
        elif isinstance(arg, torch.Tensor):
            if arg.dtype == torch.float:
                arg = arg.double()
            v = maybe_non_contig(arg).detach()
            v.requires_grad = requires_grad and v.is_floating_point()
            return v
        elif callable(arg):
            return map_arg(arg())
        else:
            return arg
    return tuple(map_arg(arg) for arg in call_args)


def unpack_variables(args):
    if isinstance(args, tuple):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args


EXCLUDE_FUNCTIONAL = {
    'addmm',
    'addmm_',
    'addbmm',
    'baddbmm',
    'addmv',
    'addmv_',
    'addr',
    'addr_',
    'reshape',
    'where'  # argument order
}
EXCLUDE_GRADCHECK = {
}
EXCLUDE_GRADGRADCHECK = {
}
EXCLUDE_GRADGRADCHECK_BY_TEST_NAME = {
    # *det methods uses svd in backward when matrix is not invertible. However,
    # svd backward is unstable unless the matrix has positive distinct singular
    # values. Generated random matrices satisfy this with high probability, but
    # we can't rely on it. So only test gradgrad on invertible test cases and
    # _distinct_singular_values.
    'test_det',
    'test_det_1x1',
    'test_det_symmetric',
    'test_det_symmetric_psd',
    'test_det_dim2_null',
    'test_det_rank1',
    'test_det_rank2',
    'test_logdet',
    'test_logdet_1x1',
    'test_logdet_symmetric',
    'test_slogdet_1x1_neg_det',
    'test_slogdet_neg_det',
    'test_slogdet_symmetric',
}


def exclude_tensor_method(name, test_name):
    # there are no tensor equivalents for these (inplace or out)
    exclude_all_tensor_method_by_test_name = {
        'test_clamp_min',
        'test_clamp_max',
        'test_clamp_min_scalar',
        'test_clamp_max_scalar',
        'test_slice',
        'test_where',
        'test_where_broadcast_all',
        'test_where_scalar',
        'test_where_scalar_broadcast_mask',
        'test_where_scalar_broadcast_non_mask',
    }
    # there are no out-of-place tensor equivalents for these
    exclude_outplace_tensor_method = {
        'index_add',
        'index_copy',
        'index_fill',
        'masked_fill',
        'masked_scatter',
        'scatter',
        'scatter_add',
        'det',
    }
    if test_name in exclude_all_tensor_method_by_test_name:
        return True
    is_magic_method = name[:2] == '__' and name[-2:] == '__'
    is_inplace = name[-1] == "_" and not is_magic_method
    if not is_inplace and name in exclude_outplace_tensor_method:
        return True
    return False


def gradgradcheck_method_precision_override(test_name):
    # these are just empirical observations, we should improve
    gradgradcheck_precision_override = {
        'test_norm': {'atol': 2e-2, 'rtol': 1e-2},
        'test_norm_1_5': {'atol': 1.5e-2, 'rtol': 1e-2},
        'test_norm_3': {'atol': 5e-2, 'rtol': 1e-2},
        'test_dist': {'atol': 5e-2, 'rtol': 1e-2},
        'test_dist_4': {'atol': 8e-2, 'rtol': 1e-2},
    }
    non_broadcasted_test_name = test_name.split("_broadcast")[0]
    override = gradgradcheck_precision_override.get(non_broadcasted_test_name)
    if override:
        if 'broadcast_lhs' in test_name or 'broadcast_rhs' in test_name:
            # errors accumulated across 1 dimension
            override = {'atol': override['atol'] * S, 'rtol': override['atol'] * S}
        elif 'broadcast_all' in test_name:
            # errors accumulated across multiple dimensions
            override = {'atol': override['atol'] * S * S, 'rtol': override['atol'] * S * S}
    return override


def run_grad_and_gradgrad_checks(test_case, name, test_name, apply_method, output_variable,
                                 input_variables, run_gradgradcheck=True):
    test_case.assertTrue(gradcheck(apply_method, input_variables, eps=1e-6, atol=PRECISION))
    if name in EXCLUDE_GRADGRADCHECK or test_name in EXCLUDE_GRADGRADCHECK_BY_TEST_NAME:
        return
    gradgradcheck_precision_override = gradgradcheck_method_precision_override(test_name)
    if gradgradcheck_precision_override is not None:
        atol = gradgradcheck_precision_override['atol']
        rtol = gradgradcheck_precision_override['rtol']
        test_case.assertTrue(gradgradcheck(apply_method, input_variables, None, atol=atol, rtol=rtol,
                                           gen_non_contig_grad_outputs=True))
    else:
        test_case.assertTrue(gradgradcheck(apply_method, input_variables, gen_non_contig_grad_outputs=True))


def run_functional_checks(test_case, test_name, name, apply_fn, run_grad_checks,
                          f_args_variable, f_args_tensor):
    output_variable = apply_fn(*f_args_variable)

    if run_grad_checks:
        run_grad_and_gradgrad_checks(test_case, name, test_name, apply_fn,
                                     output_variable, f_args_variable)

    self_variable = f_args_variable[0]
    if isinstance(output_variable, torch.Tensor) and output_variable.requires_grad and self_variable is not None:
        output_variable.backward(randn_like(output_variable))
        test_case.assertEqual(self_variable.type(), self_variable.grad.type())
        test_case.assertEqual(self_variable.size(), self_variable.grad.size())

for test in method_tests:
    name, self_size, args = test[:3]
    basic_test_name = 'test_' + name
    if len(test) >= 4 and test[3] != '':
        basic_test_name += '_' + test[3]

    dim_args_idx = test[4] if len(test) >= 5 else []

    skipTestIf = test[5] if len(test) >= 6 else []

    output_process_fn = test[6] if len(test) >= 7 else lambda x: x

    for dim_perm in product([-1, 1], repeat=len(dim_args_idx)):
        test_name = basic_test_name
        new_args = [arg * dim_perm[dim_args_idx.index(i)] if i in dim_args_idx else arg for i, arg in enumerate(args)]
        test_name = basic_test_name + ''.join('_neg' + str(i) for i, idx in enumerate(dim_perm) if idx < 0)
        new_args = tuple(new_args)

        # for-loop bodies don't define scopes, so we have to save the variables
        # we want to close over in some way
        def do_test(self, name=name, self_size=self_size, args=new_args, test_name=test_name,
                    output_process_fn=output_process_fn):
            def check(name):
                is_magic_method = name[:2] == '__' and name[-2:] == '__'
                is_inplace = name[-1] == "_" and not is_magic_method
                self_variable = create_input((self_size,))[0]
                # FixMe: run grad checks on inplace self
                if is_inplace:
                    self_variable.requires_grad = False
                # need to record this because methods can change the szie (e.g. unsqueeze)
                args_variable = create_input(args, requires_grad=not is_inplace)
                self_tensor = deepcopy(self_variable.data)
                args_tensor = deepcopy(unpack_variables(args_variable))
                output_variable = getattr(self_variable, name)(*args_variable)
                if not exclude_tensor_method(name, test_name):
                    output_tensor = getattr(self_tensor, name)(*args_tensor)
                    if not isinstance(output_tensor, torch.Tensor) and not isinstance(output_tensor, tuple):
                        output_tensor = torch.DoubleTensor((output_tensor,))
                    self.assertEqual(unpack_variables(output_variable), output_tensor)
                    # TODO: check that both have changed after adding all inplace ops

                def fn(*inputs):
                    output = getattr(inputs[0], name)(*inputs[1:])
                    return output_process_fn(output)

                if not is_inplace and name not in EXCLUDE_GRADCHECK:
                    run_grad_and_gradgrad_checks(self, name, test_name, fn,
                                                 output_variable, (self_variable,) + args_variable)

                # functional interface tests
                if hasattr(torch, name) and name not in EXCLUDE_FUNCTIONAL:
                    def fn(*inputs):
                        output = getattr(torch, name)(*inputs)
                        return output_process_fn(output)

                    f_args_variable = (self_variable,) + args_variable
                    f_args_tensor = (self_tensor,) + args_tensor
                    # could run the gradchecks again, but skip since we did it for the methods above.
                    run_functional_checks(self, test_name, name, fn,
                                          False, f_args_variable, f_args_tensor)

                # check for correct type of input.data and input.grad.data
                if not is_inplace:
                    self_variable = create_input((self_size,), requires_grad=True)[0]
                    args_variable = create_input(args, requires_grad=False)
                    output_variable = getattr(self_variable, name)(*args_variable)
                    if isinstance(output_variable, torch.autograd.Variable):
                        output_variable.backward(randn_like(output_variable))
                        self.assertTrue(type(self_variable.data) == type(self_variable.grad.data))
                        self.assertTrue(self_variable.size() == self_variable.grad.size())

                    # compare grads to inplace grads
                    inplace_name = name + '_'
                    # can't broadcast inplace to left hand side
                    skip_inplace = ('broadcast_lhs' in test_name or
                                    'broadcast_all' in test_name)
                    if hasattr(torch.ones(1), inplace_name) and not skip_inplace:
                        output_variable = getattr(self_variable, name)(*args_variable)
                        if not isinstance(output_variable, tuple):
                            output_variable = (output_variable,)
                        inplace_self_variable = deepcopy(self_variable)
                        inplace_self_variable_copy = tuple(i + 0 if i is not None else None
                                                           for i in (inplace_self_variable,))
                        inplace_args_variable = deepcopy(args_variable)
                        inplace_args_variable_copy = tuple(i + 0 if i is not None else None
                                                           for i in inplace_args_variable)

                        inplace_output_variable = (
                            getattr(inplace_self_variable_copy[0], inplace_name)(*inplace_args_variable_copy))
                        if not isinstance(inplace_output_variable, tuple):
                            inplace_output_variable = (inplace_output_variable,)
                        self.assertEqual(inplace_output_variable, output_variable)
                        # Check that gradient is the same
                        for inp_i, i in zip((inplace_self_variable,) + inplace_args_variable,
                                            (self_variable,) + args_variable):
                            if not isinstance(inp_i, torch.Tensor):
                                assert not isinstance(i, torch.Tensor)
                                continue
                            if inp_i.grad is not None:
                                inp_i.grad.data.zero_()
                            if i.grad is not None:
                                i.grad.data.zero_()
                        for io, o in zip(inplace_output_variable, output_variable):
                            grad = randn_like(io).double()
                            io.backward(grad)
                            o.backward(grad)
                        for inp_i, i in zip((inplace_self_variable,) + inplace_args_variable,
                                            (self_variable,) + args_variable):
                            if not isinstance(inp_i, torch.Tensor):
                                continue
                            self.assertEqual(inp_i.grad, i.grad)

            check(name)
            inplace_name = name + '_'
            # can't broadcast inplace to left hand side
            broadcast_skip_inplace = 'broadcast_lhs' in test_name or 'broadcast_all' in test_name
            if hasattr(torch.ones(1), inplace_name) and not broadcast_skip_inplace:
                check(inplace_name)

        assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name

        for skip in skipTestIf:
            do_test = skip(do_test)

        setattr(TestAutograd, test_name, do_test)


if __name__ == '__main__':
    run_tests()
