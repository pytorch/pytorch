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
from torch._six import inf, nan, istuple
from torch.autograd.gradcheck import gradgradcheck, gradcheck
from torch.autograd.function import once_differentiable
from torch.autograd.profiler import profile
from torch.utils.checkpoint import checkpoint
from common_utils import (TEST_MKL, TestCase, run_tests, skipIfNoLapack,
                          suppress_warnings, skipIfRocm,
                          prod_single_zero, random_square_matrix_of_rank,
                          random_symmetric_matrix, random_symmetric_psd_matrix,
                          random_symmetric_pd_matrix, make_nonzero_det,
                          random_fullrank_matrix_distinct_singular_value, load_tests)
from common_cuda import TEST_CUDA
from torch.autograd import Variable, Function, detect_anomaly
from torch.autograd.function import InplaceFunction
from torch.testing import make_non_contiguous, randn_like
from common_methods_invocations import (method_tests,
                                        create_input, unpack_variables,
                                        EXCLUDE_FUNCTIONAL, EXCLUDE_GRADCHECK,
                                        EXCLUDE_GRADGRADCHECK,
                                        EXCLUDE_GRADGRADCHECK_BY_TEST_NAME,
                                        exclude_tensor_method,
                                        mask_not_all_zeros,
                                        L, S)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

PRECISION = 1e-4


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
        go = torch.ones((), requires_grad=True)
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
        self.assertExpected(x_grad_desc, "x_grad_desc")
        self.assertExpected(y_grad_desc, "y_grad_desc")

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

        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            MyFunction.apply(v).backward()
            self.assertEqual(v.grad, torch.full(shape, 2))

            v.grad.data.zero_()
            MyFunction.apply(v.clone()).backward()
            self.assertEqual(v.grad, torch.full(shape, 2))

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

    def test_invalid_gradients(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return torch.randn(10, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, 'expected shape'):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            MyFunction.apply(input).sum().backward()
        with self.assertRaisesRegex(RuntimeError, 'expected type'):
            input = torch.randn(10, dtype=torch.double, requires_grad=True)
            MyFunction.apply(input).sum().backward()

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

    def test_slogdet_sign(self):
        a = torch.randn(3, 3, requires_grad=True)
        s, logdet = a.slogdet()

        # test that sign should not require grad
        self.assertFalse(s.requires_grad)

        # test that backward through computation involving sign works
        def sign_mul_logdet(mat):
            s, logdet = mat.slogdet()
            return s * logdet

        u, s, v = a.detach().svd()
        s.abs_().clamp_(0.0001)
        for sign in (-1, 1):
            s[-1] = sign
            mat = torch.chain_matmul(u, s.diag(), v.t()).requires_grad_()
            gradcheck(sign_mul_logdet, mat)
            gradgradcheck(sign_mul_logdet, mat)

    def test_sum_to_with_empty_dim_grad(self):
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        c = a + b
        assert c.shape == (4, 0)
        c.sum().backward()

        self.assertEqual(b.grad, torch.zeros(4, 1))
        self.assertEqual(a.grad, torch.zeros(4, 0))

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

    def test_grad_nonleaf_register_hook(self):
        # This checks an edge case for register_hook.
        # We want to capture grad of a nonleaf tensor,
        # but avoid segfault during backward of other nonleaf tensors
        x = torch.randn(5, requires_grad=True)
        x_list = x.unbind()

        x0 = x_list[0]
        hook_results = [None]

        def hook(grad):
            hook_results[0] = grad
        x0.register_hook(hook)

        x_list[0].backward()
        self.assertEqual(hook_results[0], torch.tensor(1.))
        expected_grad = torch.tensor([1., 0, 0, 0, 0])
        self.assertEqual(x.grad, expected_grad)
        self.assertIsNone(x_list[0].grad)

        for i in range(1, 5, 1):
            x_list[i].backward()
            self.assertEqual(hook_results[0], None)
            expected_grad[i] = 1.0
            self.assertEqual(x.grad, expected_grad)
            self.assertIsNone(x_list[i].grad)

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

    def test_grad_fn_badcalls(self):
        error_regex = 'expected .* arguments, got .* instead'
        x = torch.ones(1, requires_grad=True)
        y = x ** 2
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn(x.detach(), x.detach())  # too many
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn()  # too few

        y.grad_fn(x.detach())  # this should succeed

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
        sparse_grad1 = torch.sparse.DoubleTensor(i1, v1, size)
        i2 = torch.LongTensor([
            [0, 1, 3, 4],
            [0, 1, 2, 2],
        ])
        v2 = torch.DoubleTensor([[1, 2], [4, 3], [4, 5], [7, 8]])
        sparse_grad2 = torch.sparse.DoubleTensor(i2, v2, size)
        dense_grad = torch.rand(size).double()
        sparse_fn1 = FixedGradientFunction(sparse_grad1)
        sparse_fn2 = FixedGradientFunction(sparse_grad2)
        dense_fn = FixedGradientFunction(dense_grad)

        # sparse first
        x = torch.randn(size, requires_grad=True)
        (sparse_fn1(x) + dense_fn(x) + sparse_fn2(x)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # dense first
        x = torch.randn(size, requires_grad=True)
        (dense_fn(x) + sparse_fn1(x) + sparse_fn2(x)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # sparse only
        x = torch.randn(size, requires_grad=True)
        (sparse_fn1(x) + sparse_fn2(x)).sum().backward()
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    def test_sparse_mm_backward(self):
        size = (3, 3)
        sparse = torch.sparse_coo_tensor(size, requires_grad=True)
        dense = torch.randn(size, requires_grad=True)

        z = sparse.mm(dense)
        with self.assertRaisesRegex(RuntimeError,
                                    "calculating the gradient of a sparse Tensor argument to mm is not supported."):
            z.sum().backward()

        z = dense.addmm(sparse, dense)
        with self.assertRaisesRegex(RuntimeError,
                                    "calculating the gradient of a sparse Tensor argument to mm is not supported."):
            z.sum().backward()

    @skipIfRocm
    def test_sparse_ctor_getter_backward(self):
        # See NOTE [ Sparse: autograd and API ] on the expected behavior of this test
        def test(size, sparse_dim, nnz, device):
            v_size = [nnz] + list(size[sparse_dim:])
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)

            inp = torch.randn(v_size, requires_grad=True)
            other = self.genSparseTensor(size, sparse_dim, nnz, is_uncoalesced=True)[0]
            other = other.to(device)

            def fn(v):
                x = torch.sparse_coo_tensor(i, v, size, device=device)
                y = (x + other).coalesce()
                yv = y.values()
                new_v = yv.tanh()
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                return z.coalesce().values()

            gradcheck(fn, (inp,))
            # FIXME: make gradgradcheck work.
            # gradgradcheck(fn, (inp,))

            # assert that _values is non-differentiable
            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.detach().requires_grad_()._values().backward(torch.ones_like(other._values()))

        devices = ['cpu']

        if torch.cuda.is_available():
            devices.append('cuda')

        for empty_i, empty_v, empty_nnz in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            for device in devices:
                test(sparse_size + dense_size, len(sparse_size), nnz, device)

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

    @unittest.skipIf(not TEST_CUDA, "need CUDA memory stats")
    def test_free_unneeded_tensor(self):
        x = torch.randn(2, 3, 10, 10, device='cuda', requires_grad=True)
        m = torch.randn(1, 3, 1, 1, device='cuda')

        z = x.sum()
        base_mem = torch.cuda.memory_allocated()
        z = ((x + 2) * m).sum()
        end_mem = torch.cuda.memory_allocated()

        # In the end the memory usage should remain equal, because neither of
        # (x + 2) and ((x + 2) * m) should be kept alive for backward, while the
        # previous allocation of z had the same size as the current one.
        self.assertEqual(base_mem, end_mem)

    def test_no_unnecessary_save(self):
        # If we kept x in the derivative Function of x * 2 we would
        # get an error in the backward that would complain that we've
        # modified x, which was needed for gradient computation.
        # Since we should elide unnecessary saves, this test should pass.
        mu = torch.ones(1, requires_grad=True)
        x = torch.empty(1)
        loss = 0
        for i in range(3):
            x.detach_()
            x.copy_(mu + i)
            loss += (x * torch.tensor([float(i)])).sum()
        loss.backward()

    def test_no_grad(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4)
        with torch.no_grad():
            w = x + y

        @torch.no_grad()
        def adder(x, y):
            return x + y

        z = adder(x, y)

        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.grad_fn)
        self.assertFalse(z.requires_grad)
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))
        self.assertIsNone(z.grad_fn)

        # test nested decorator and with-statement on no_grad
        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            w = adder(x, y)
            self.assertFalse(torch.is_grad_enabled())

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
        x = x.cuda().half()
        x.grad = torch.zeros_like(x)  # would raise an error unless sparse type is properly handled

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
        self.assertEqual(len(next_functions), 2)
        self.assertIs(next_functions[0][0], a.grad_fn)
        self.assertIs(next_functions[1][0], None)

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

    def test_unbind(self):
        stacked = torch.randn(3, 10, 10, requires_grad=True)
        x, y, z = stacked.unbind()
        grad = torch.randn(3, 10, 10)
        torch.autograd.backward([x, y, z], grad.unbind())
        self.assertEqual(stacked.grad.data, grad)
        # check that it works with only one gradient provided (#9977)
        for i in range(3):
            stacked = torch.randn(3, 10, 10, requires_grad=True)
            outs = stacked.unbind()
            gi = grad.unbind()[i]
            g, = torch.autograd.grad(outs[i], stacked, gi)
            g_expected = torch.stack([gi if j == i else torch.zeros_like(gi)
                                      for j in range(3)], dim=0)
            self.assertEqual(g, g_expected)

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

    @skipIfRocm
    def test_ctc_loss(self):
        batch_size = 64
        num_labels = 101
        target_length = 15
        gradcheck_input_size = 10

        # device, input_length
        tests = [('cpu', 150, False),
                 ('cpu', 150, True)]
        if torch.cuda.is_available():
            tests += [('cuda', 50, False),
                      ('cuda', 150, False),
                      ('cuda', 50, True),
                      ('cuda', 150, True)]

        for device, input_length, vary_lengths in tests:
            targets = torch.randint(1, num_labels, (batch_size, target_length),
                                    device=device, dtype=torch.long)
            x = torch.randn(gradcheck_input_size, device=device, requires_grad=True)
            tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                       device=device)
            input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                              if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
            target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                               if vary_lengths or i == 0 else target_length) for i in range(batch_size)]

            def ctc_after_softmax(x):
                x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                          .view(input_length, batch_size, num_labels))
                log_probs = torch.log_softmax(x_full, 2)
                return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            gradcheck(ctc_after_softmax, [x])

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
    @skipIfRocm
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
    @skipIfRocm
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
        f[0] = nan
        self.assertTrue(math.isnan(float(f)))
        f[0] = inf
        self.assertEqual(float(f), inf, allow_inf=True)
        f[0] = -inf
        self.assertEqual(float(f), -inf, allow_inf=True)

        # integral -> floating point
        # check we can convert something that loses precision
        pyscalar = 1234567890123456789
        self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
        l[0] = pyscalar
        self.assertEqual(float(l), float(pyscalar))

        # floating point -> integral
        f[0] = nan
        self.assertRaises(ValueError, lambda: integral_conv(f[0]))
        f[0] = inf
        self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
        f[0] = -inf
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
        test_nonzero(f, nan, bool(nan))
        test_nonzero(f, inf, bool(inf))
        test_nonzero(f, -inf, bool(-inf))

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
        x = torch.tensor([1.], requires_grad=True)
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

    def test_broadcast_tensors(self):
        f_args_variable = (torch.randn(3, requires_grad=True),
                           torch.randn(1, 2, 1, requires_grad=True),
                           torch.randn(1, 1, requires_grad=True),
                           torch.randn(5, 1, 1, requires_grad=True))
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_broadcast_tensors", "broadcast",
                              lambda a, b, c, d: torch.broadcast_tensors(a, b, c, d),
                              True, f_args_variable, f_args_tensor)

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

    def test_cat_empty_legacy(self):
        f_args_variable = (torch.randn(0, requires_grad=True),
                           torch.randn(S, S, requires_grad=True))
        # gradgradcheck doesn't work, probably because legacy size tracking is wrong somewhere,
        # hence False passed below, but gradcheck checked explicitly.
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_empty_legacy", "cat",
                              lambda a, b: torch.cat((a, b)),
                              False, f_args_variable, f_args_tensor)
        self.assertTrue(gradcheck(lambda a, b: torch.cat((a, b)), f_args_variable, eps=1e-6, atol=PRECISION))

    def test_cat_empty(self):
        f_args_variable = (torch.randn(0, S, requires_grad=True),
                           torch.randn(S, S, requires_grad=True))
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_empty", "cat",
                              lambda a, b: torch.cat((a, b)),
                              True, f_args_variable, f_args_tensor)

    @skipIfNoLapack
    def test_cholesky(self):
        def func(root):
            x = torch.matmul(root, root.transpose(-1, -2)) + 1e-05
            return torch.cholesky(x, upper)

        def run_test(upper, dims):
            root = torch.rand(*dims)
            indices = torch.ones(dims[-1], dims[-1], dtype=torch.uint8).tril()
            indices = indices.expand_as(root)
            root[indices] = 0
            root.requires_grad_()

            gradcheck(func, [root])
            gradgradcheck(func, [root])

        for upper, dims in product([True, False], [(3, 3), (4, 3, 2, 2)]):
            run_test(upper, dims)
            run_test(upper, dims)

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

    def test_pow_zero_tensor_gradient(self):
        def run_test(input_size, exponent):
            input = torch.zeros(*input_size, requires_grad=True)
            input.pow(exponent).sum().backward()
            self.assertEqual(input.grad.data.abs().sum(), 0)

        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    def test_pow_scalar_base(self):
        a = torch.arange(1, 13, dtype=torch.double).view(3, 4).requires_grad_()
        gradcheck(lambda a: torch.pow(2, a), (a,))

    # test for backward in https://github.com/pytorch/pytorch/issues/15511
    def test_pdist_large(self):
        def func(x):
            return torch.pdist(x, p=2)

        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            # shape[0] should be able to be (roughly) arbitrarily large, but the kernel
            # is currently limited to smaller sizes (see issue above); this is just testing
            # a floor.
            shape = (1000, 1)
            x = torch.randn(shape, device=device).requires_grad_()
            output = torch.pdist(x, p=2)
            # just run a single backward, as gradcheck/gradgradcheck is expensive here
            output.sum().backward()

    @skipIfNoLapack
    def test_pinverse(self):
        # Why is pinverse tested this way, and not ordinarily as other linear algebra methods?
        # 1. Pseudo-inverses are not generally continuous, which means that they are not differentiable
        # 2. Derivatives for pseudo-inverses exist typically for constant rank (Golub et al, 1973)
        # 3. This method creates two orthogonal matrices, and a constructs a test case with large
        #    singular values (given by x to the function).
        # 4. This will ensure that small perturbations don't affect the rank of matrix, in which case
        #    a derivative exists.
        # 5. This test exists since pinverse is implemented using SVD, and is hence a backpropable method
        m, n = 5, 10
        U = torch.randn(n, m).qr()[0].t()  # Orthogonal with dimensions m x n
        V = torch.randn(n, m).qr()[0].t()  # Orthogonal with dimensions m x n

        def func(x):
            S = torch.cat([x, torch.zeros(n - m)], 0)
            M = U.mm(torch.diag(S)).mm(V.t())
            return M.pinverse()

        gradcheck(func, [torch.rand(m).add_(1).requires_grad_()])
        gradcheck(func, [torch.rand(m).add_(10).requires_grad_()])
        gradgradcheck(func, [torch.rand(m).add_(1).requires_grad_()])
        gradgradcheck(func, [torch.rand(m).add_(10).requires_grad_()])

    def test_chain_matmul(self):
        def gen_matrices(p):
            matrices = []
            for (pi, pi_1) in zip(p[:-1], p[1:]):
                matrices.append(torch.randn(pi, pi_1).requires_grad_())
            return matrices

        gradcheck(torch.chain_matmul, gen_matrices([5, 10, 15, 5]))
        gradcheck(torch.chain_matmul, gen_matrices([3, 5, 2, 6]))
        gradcheck(torch.chain_matmul, gen_matrices([6, 2, 4, 8, 10]))
        gradgradcheck(torch.chain_matmul, gen_matrices([5, 10, 15, 5]))
        gradgradcheck(torch.chain_matmul, gen_matrices([3, 5, 2, 6]))
        gradgradcheck(torch.chain_matmul, gen_matrices([6, 2, 4, 8, 10]))

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

        def test(x, prepro_fn, size, strides, offset=None):
            x = x.to(torch.double).detach().requires_grad_()

            # Check that forward will **not** resize storage because it may
            # cause NaN in output and fail numerical Jacobian check consequently
            with torch.no_grad():
                y = prepro_fn(x) if prepro_fn is not None else x
                max_offset = sum((si - 1) * st for si, st in zip(size, strides))
                max_offset += offset if offset is not None else y.storage_offset()
                assert max_offset < len(y.storage()), "test case resizes storage"

            def closure(x):
                if prepro_fn is not None:
                    x = prepro_fn(x)
                return x.as_strided(size, strides, offset)

            gradcheck(closure, [x])
            gradgradcheck(closure, [x])

        # test
        test(torch.arange(0, 25), lambda x: x.view(5, 5), [3, 3], [6, 2], 2)

        # test crazy stride at dim with size 1 case
        test(torch.randn(12), None, [1, 2, 1, 5], [0, 5, 100, 1], 2)

        # test expand case
        test(torch.randn(5), None, [3, 3, 3], [0, 1, 0], 2)
        test(torch.randn(5), None, [3, 3, 3], [0, 0, 0], 4)
        test(torch.randn(5), lambda x: x.expand(5, 5), [5, 5], [0, 1], 0)

        # test non-expand overlapping case
        test(torch.randn(35), None, [6, 6], [5, 1], 2)
        test(torch.randn(15), None, [3, 2], [3, 6], 2)

        # test transpose case
        test(torch.randn(3, 4), None, [4, 3], [1, 4])

        # test "getting things outside the input" case
        x = torch.randn(6, 2)
        test(x[3:], None, [3, 2], [2, 1], 0)  # should be all zeros
        self.assertEqual(x[3:].as_strided([3, 2], [2, 1], 0), x[:3])

        # test select on expanded input case
        test(torch.randn(2, 3), lambda x: x.expand(10, 2, 3), [2, 3], [3, 1], 0)

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

    def test_reduce_dtype(self):
        def test_reduction(op, has_no_dim):
            x = torch.randn(3, 3, dtype=torch.float, requires_grad=True)

            if has_no_dim:
                grad1, = torch.autograd.grad([op(x)], [x])
                grad2, = torch.autograd.grad([op(x, dtype=torch.double)], [x])
                self.assertEqual(grad1, grad2)
                self.assertEqual(grad2.dtype, torch.float)

            gi = torch.randn(op(x, dim=0).shape, dtype=torch.float)
            grad1, = torch.autograd.grad([op(x, dim=0)], [x], gi)
            grad2, = torch.autograd.grad([op(x, dim=0, dtype=torch.double)], [x], gi.double())
            self.assertEqual(grad1, grad2)
            self.assertEqual(grad2.dtype, torch.float)

        test_reduction(torch.sum, True)
        test_reduction(torch.prod, True)
        test_reduction(torch.cumsum, False)
        test_reduction(torch.cumprod, False)

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

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_rnn_backward_to_input_but_not_parameters_cuda(self):
        # this checks whether it is possible to not require
        # weight parameters, but require inputs, see #7722
        dev = torch.device('cuda')
        l = torch.nn.LSTM(2, 3).to(dev)
        for p in l.parameters():
            p.requires_grad = False
        s = torch.randn(1, 1, 2, requires_grad=True, device=dev)
        out, _ = l(s)
        out.sum().backward()
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_lstmcell_backward_only_one_output_grad(self):
        # checks that undefined gradients doen't hamper the backward
        # see #11872
        dev = torch.device('cuda')
        l = torch.nn.LSTMCell(2, 3).to(dev).double()
        s = torch.randn(1, 2, device=dev, dtype=torch.double, requires_grad=True)
        for i in range(2):
            out = l(s)[i]
            out.sum().backward()
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    def test_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                gI = gO.clone().expand(size)
                gI[0] = 0
                gI[0] /= 0  # Generate a nan
                if ctx.fail_0th:
                    return gI, None, None
                else:
                    return None, gI, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        out.backward()  # Should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 0th output."):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            self.assertIn('No forward pass information', str(w[0].message))

        inp = torch.rand(size, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 1th output."):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out = MyFunc.apply(inp, inp, False)
                    out.backward()
            self.assertIn('MyFunc.apply', str(w[0].message))

    @skipIfNoLapack
    def test_symeig_no_eigenvectors(self):
        A = torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float32, requires_grad=True)
        w, v = torch.symeig(A, eigenvectors=False)
        with self.assertRaisesRegex(RuntimeError, 'cannot compute backward'):
            torch.autograd.backward([w, v], [torch.ones_like(w), torch.ones_like(v)])

    @skipIfNoLapack
    def test_svd_no_singularvectors(self):
        A = torch.randn(2, 2, dtype=torch.float32, requires_grad=True)
        u, s, v = torch.svd(A, compute_uv=False)
        with self.assertRaisesRegex(RuntimeError, 'cannot compute backward'):
            torch.autograd.backward([u, s, v], [torch.ones_like(u), torch.ones_like(s), torch.ones_like(v)])

    def test_no_grad_copy(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            @staticmethod
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.])

            @staticmethod
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)

        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        # non-contiguous grad should be copied
        NonContGradFunc.apply(MyFunc.apply(a, b)).backward()
        self.assertFalse(a.grad.data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(b.grad.data_ptr() == MyFunc.static_grad_ptr)
        # test case that should trigger no copy for one of a,b
        a.grad = b.grad = None
        MyFunc.apply(a, b)[1][0].backward()
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad.data_ptr()
        p_b = b.grad.data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

    def test_gradcheck_single_input(self):
        def f(inp):
            return inp.mul(5)

        gradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True))
        gradgradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True))

    def test_gradcheck_sparse_input(self):
        def fn(sparse):
            return torch.sparse.sum(sparse)

        gradcheck(fn, torch.rand(10).to_sparse().requires_grad_(True), check_sparse_nnz=True)
        with self.assertRaisesRegex(RuntimeError, 'gradcheck expects all tensor inputs are dense'):
            gradcheck(fn, torch.rand(10).to_sparse().requires_grad_(True), check_sparse_nnz=False)

    @unittest.skipIf(not TEST_CUDA, "Requires cuda for multi device")
    def test_multi_device_reentrant_autograd(self):
        # Output on gpu so that this task will be associated with the gpu thread
        def fn_on_gpu(inp):
            # Artificially increase the priority of the next op to make sure it runs
            # as soon as we reach it before the ops of branch1.
            dummy = inp * 2 * 2 * 2 * 2
            return inp.cuda()

        def parent_on_cpu(inp):
            # Slow branch of ops on gpu so that the work queue for the gpu thread
            # won't empty too quickly. They also have smaller priorities than the
            # ones created by fn_on_gpu
            branch1 = inp.cuda()
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            # Perform checkpoint on cpu tensors. So the last op performed in the reentrant
            # autograd is an AccumulateGrad that runs on the cpu thread for the gpu thread.
            # So the cpu thread will notify the gpu thread with an empty FunctionTask.
            branch2 = checkpoint(fn_on_gpu, inp)
            out = branch2 + branch1
            return out

        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)
        # This will segfault if the empty FunctionTask is not handled properly in the
        # gpu thread ReadyQueue
        out.sum().backward()


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


def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()


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


def add_test(
        name,
        self_size,
        args,
        variant_name='',
        dim_args_idx=(),
        skipTestIf=(),
        output_process_fn=lambda x: x,
        kwargs=None):
    kwargs = kwargs if kwargs else {}
    basic_test_name = 'test_' + name
    if variant_name != '':
        basic_test_name += '_' + variant_name

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
                self_variable = create_input((self_size,))[0][0]
                # FixMe: run grad checks on inplace self
                if is_inplace:
                    self_variable.requires_grad = False
                # need to record this because methods can change the size (e.g. unsqueeze)
                args_variable, kwargs_variable = create_input(args, requires_grad=not is_inplace, call_kwargs=kwargs)
                self_tensor = deepcopy(self_variable.data)
                args_tensor = deepcopy(unpack_variables(args_variable))
                output_variable = getattr(self_variable, name)(*args_variable, **kwargs_variable)
                if not exclude_tensor_method(name, test_name):
                    output_tensor = getattr(self_tensor, name)(*args_tensor, **kwargs_variable)
                    if not isinstance(output_tensor, torch.Tensor) and not istuple(output_tensor):
                        output_tensor = torch.DoubleTensor((output_tensor,))
                    self.assertEqual(unpack_variables(output_variable), output_tensor)
                    # TODO: check that both have changed after adding all inplace ops

                def fn(*inputs):
                    output = getattr(inputs[0], name)(*inputs[1:], **kwargs)
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
                    self_variable = create_input((self_size,), requires_grad=True)[0][0]
                    args_variable, kwargs_variable = create_input(args, requires_grad=False, call_kwargs=kwargs)
                    output_variable = getattr(self_variable, name)(*args_variable, **kwargs_variable)
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
                        output_variable = getattr(self_variable, name)(*args_variable, **kwargs_variable)
                        if not isinstance(output_variable, tuple):
                            output_variable = (output_variable,)
                        inplace_self_variable = deepcopy(self_variable)
                        inplace_self_variable_copy = tuple(i.clone() if isinstance(i, torch.Tensor) else i
                                                           for i in (inplace_self_variable,))
                        inplace_args_variable = deepcopy(args_variable)
                        inplace_args_variable_copy = tuple(i.clone() if isinstance(i, torch.Tensor) else i
                                                           for i in inplace_args_variable)

                        inplace_output_variable = (
                            getattr(inplace_self_variable_copy[0], inplace_name)(*inplace_args_variable_copy,
                                                                                 **kwargs_variable))
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

for test in method_tests():
    add_test(*test)

if __name__ == '__main__':
    run_tests()
