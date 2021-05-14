import contextlib
import gc
import sys
import io
import math
import random
import tempfile
import time
import threading
import unittest
import warnings
from copy import deepcopy
from collections import OrderedDict
from itertools import product, permutations
from operator import mul
from functools import reduce, partial
import torch

from torch import nn
from torch._six import inf, nan
from torch.autograd.function import once_differentiable
from torch.autograd.profiler import (profile, format_time, EventList,
                                     FunctionEvent, FunctionEventAvg,
                                     record_function, emit_nvtx)
import torch.autograd.functional as autogradF
from torch.utils.checkpoint import checkpoint
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (TestCase, run_tests, skipIfNoLapack,
                                                  suppress_warnings, slowTest,
                                                  load_tests,
                                                  IS_WINDOWS, IS_MACOS, CudaMemoryLeakCheck,
                                                  TEST_WITH_ROCM,
                                                  gradcheck, gradgradcheck, make_tensor)
from torch.autograd import Variable, Function, detect_anomaly, kineto_available
from torch.autograd.function import InplaceFunction
import torch.autograd.forward_ad as fwAD
from torch.testing import randn_like
from torch.testing._internal.common_methods_invocations import (method_tests,
                                                                create_input, unpack_variables,
                                                                EXCLUDE_FUNCTIONAL, EXCLUDE_GRADCHECK,
                                                                EXCLUDE_GRADGRADCHECK,
                                                                EXCLUDE_GRADGRADCHECK_BY_TEST_NAME,
                                                                exclude_tensor_method,
                                                                mask_not_all_zeros,
                                                                S)
from torch.testing._internal.common_device_type import (instantiate_device_type_tests, skipCUDAIfRocm,
                                                        onlyCPU, onlyCUDA, onlyOnCPUAndCUDA, dtypes, dtypesIfCUDA,
                                                        deviceCountAtLeast, skipCUDAIfCudnnVersionLessThan,
                                                        skipCUDAIf, skipMeta)

_END_SENTINEL = object()

def getattr_qualified(obj, qname, default=None):
    """ Like getattr but works with qualified names

    e.g. getattr(torch, 'fft.rfft')
    """
    path = qname.split('.')
    for name in path:
        obj = getattr(obj, name, _END_SENTINEL)
        if obj is _END_SENTINEL:
            return default
    return obj

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

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

    def test_tensor_grad_warnings(self):
        dummy = torch.empty(1)

        with warnings.catch_warnings(record=True) as w:
            # Accessing .grad on leaf
            dummy.requires_grad_()
            foo = dummy.grad
            self.assertEqual(len(w), 0)

            # Accessing .grad on non-leaf
            dummy = dummy.clone()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

            # Accessing .grad on non-leaf that retains gradients
            dummy.retain_grad()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

    def _function_test(self, cls):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones((), requires_grad=True)
        result.sum().backward(go, create_graph=True)

        self.assertEqual(x.grad, y + torch.ones(5, 5))
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)
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
                         'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')
        self.assertEqual(graph_desc(y.grad.grad_fn),
                         'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')

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
            self.assertEqual(v.grad, torch.full(shape, 2.))

            with torch.no_grad():
                v.grad.zero_()
            MyFunction.apply(v.clone()).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.))

    def test_function_returns_undefined_tensor(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                return None

        # Test that undefined tensors returned from custom backward function
        # are propagated as undefined and not tensor full of zeroes
        x = torch.ones(1, requires_grad=True)

        MyFunction.apply(x).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x ** 2).backward()
        self.assertIsNone(x.grad)

        MyFunction.apply(x).sum().backward()
        self.assertIsNone(x.grad)

        self.assertIsNone(torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0])

    def test_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(grad, torch.zeros(1))
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_dont_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.set_materialize_grads(False)
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertIsNone(grad)
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_legacy_function_deprecation_exception(self):
        # Trigger exception
        class MyFunction(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        # Check exception occurs
        with self.assertRaisesRegex(
                RuntimeError,
                'Legacy autograd function with non-static forward method is deprecated'):
            MyFunction()(torch.randn(3, 4))

    class SimulateBackwardError(Function):

        @staticmethod
        def forward(ctx, input):
            return input.clone()

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            raise Exception("Simulate error on backward pass")

    def test_custom_function_exception(self):

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        tmp = (t1 + t2) * (t1 + t2)
        t3 = TestAutograd.SimulateBackwardError.apply(tmp)
        with self.assertRaisesRegex(Exception, "Simulate error on backward pass"):
            t3.sum().backward()

    def test_custom_function_non_tensor_inputs_outputs(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale

                # Save scale
                ctx.scale = scale
                ctx.save_for_backward(t1, t2, t3)
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *grads):
                # Verify grads
                self.assertEqual(7, len(grads))
                self.assertIsNone(grads[0])
                self.assertIsNone(grads[2])
                self.assertIsNone(grads[3])
                self.assertIsNone(grads[5])

                scale = ctx.scale
                var1, var2, var3 = ctx.saved_tensors
                return (
                    grads[1] * scale + grads[4] * var2 * scale + grads[6],
                    grads[1] * var3 * scale + grads[4] * var1 * scale,
                    None,
                    grads[1] * var2 * scale + grads[4] * scale,
                )

        t1 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t2 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t3 = torch.rand(10, dtype=torch.double)
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        # Test gradcheck
        def foo(t1, t2, t3):
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

        gradcheck(foo, (t1, t2, t3))

    def test_custom_function_no_tensors(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *args):
                return (args[0], args[1], None, args[2])

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

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

    def test_unrelated_inputs(self):
        # test to ensure grad(grad)check runs successfully even if there is an
        # unrelated (but differentiable) inputs

        def my_function(x, y):
            return x * x

        x = torch.rand(10, dtype=torch.double, requires_grad=True)
        y = torch.rand(10, dtype=torch.double, requires_grad=True)

        gradcheck(my_function, (x, y))
        gradgradcheck(my_function, (x, y))

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

    def test_accumulate_grad_tensor_reference(self):
        def _test_grad_tensor(params_grad_tensor, backward_grad_tensor, should_preserve_reference, create_graph):
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            params.grad = params_grad_tensor
            grad_saved = params.grad
            params.backward(backward_grad_tensor, create_graph=create_graph)
            self.assertEqual(id(grad_saved) == id(params.grad), should_preserve_reference)

        for create_graph in (False, True):
            # Accumulate dense gradient to sparse gradient will change the `params.grad` reference
            _test_grad_tensor(
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                torch.tensor([1.5, 1.5]),
                False,  # never accumulates in-place
                create_graph)

            # Accumulate dense gradient to dense gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.tensor([1.5, 1.5]),
                torch.tensor([1.5, 1.5]),
                not create_graph,
                create_graph)

            # Accumulate sparse gradient to sparse gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                not create_graph,
                create_graph)

    @skipIfNoLapack
    def test_slogdet_sign(self):
        a = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
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
            mat = torch.linalg.multi_dot([u, s.diag(), v.t()]).requires_grad_()
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

        with torch.no_grad():
            x_grad = 2 * x + y
            y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        self.assertEqual(x.grad, x_grad + x_hv)
        self.assertEqual(y.grad, y_grad + y_hv)

    def test_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x + y
        y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(
            outputs=[grad_sum], grad_outputs=[torch.ones(2, 2)],
            inputs=[x], create_graph=True)
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4

        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        # Test that grad_outputs and outputs have the same shape
        grad_out = torch.ones(2)
        try:
            torch.autograd.grad(
                outputs=[grad_sum], grad_outputs=[grad_out],
                inputs=[x], create_graph=True)
            self.assertFail()
        except RuntimeError as error:
            self.assertEqual(str(error), "Mismatch in shape: grad_output[0] has a shape of "
                             + str(grad_out.shape) + " and output[0] has a shape of "
                             + str(grad_sum.shape) + ".")

    def test_grad_nonleaf(self):
        x_init = torch.randn(2, 2, requires_grad=True)
        x = x_init
        y = torch.randn(2, 2, requires_grad=True)
        grad_output = torch.ones(2, 2)

        def fn(x):
            return x ** 2 + y * x + y ** 2

        for _ in range(5):
            grad_x, = torch.autograd.grad(
                fn(x), x, grad_outputs=grad_output, create_graph=True)

            grad_x_expected = 2 * x + y
            self.assertIsNone(y.grad)
            self.assertIsNone(x.grad)
            self.assertEqual(grad_x, grad_x_expected)

            x = x + 0.05 * grad_x

        val_init = fn(x_init).sum()
        val_final = fn(x).sum()
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

        self.assertEqual(grad_a, go)
        self.assertEqual(grad_b, go * 2)
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

    def test_hook_with_no_name(self):
        # Create a hook that do not have a __name__ attribute
        class MyHookClass:
            def __call__(self, grad):
                return grad.clone()

        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()
        # Should run fine

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
            self.assertEqual(l.grad, i * i * (1 + l))

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

        # allow_unused=False, but grads contains None inside, should throw
        with self.assertRaisesRegex(RuntimeError,
                                    "Set allow_unused=True"):
            grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=False)

    def test_grad_unreachable_discovery(self):
        # Test that certain nodes are not erroneously executed when an input
        # is unreachable. See #39784
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                self.fail("This node should not be executed!")

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        (gY,) = torch.autograd.grad(x, (y, ), allow_unused=True)
        self.assertIsNone(gY)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        z = torch.randn(1, requires_grad=True)
        (gY, gZ) = torch.autograd.grad(x + z, (y, z), allow_unused=True)
        self.assertIsNone(gY)
        self.assertIsNotNone(gZ)

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        torch.autograd.backward(x, inputs=(y, ))  # allow_unused is implicitly True!
        self.assertIsNone(y.grad)

    def test_hooks(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)

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
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad, (x + 1) * 2)

        y.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad, (x + 1) * 4)

    def test_hooks_cpp(self):
        # Tests hooks for autograd function implemented in C++
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()

        counter = [0]

        def bw_hook(grad):
            counter[0] += 1
            return grad * 2

        x = torch.ones(5, 5, dtype=torch.double, requires_grad=True)
        z = bn(x)
        z.register_hook(bw_hook)
        z.sum().backward()

        self.assertEqual(counter[0], 1, msg='bw_hook not called')
        self.assertEqual(x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-5, rtol=0)

    def test_hook_none(self):
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        class NoneGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                assert ctx.needs_input_grad[0]
                assert not ctx.needs_input_grad[1]
                return x, y

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                return grad_x, None

        was_called = [False]

        def hook(grad):
            self.assertIsNotNone(grad)
            was_called[0] = True

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        rx, ry = NoneGradientFunction.apply(x, y)
        rx.register_hook(hook)
        ry.register_hook(hook)
        sum(rx, ry).sum().backward()
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
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)

        with torch.no_grad():
            input.grad.zero_()
        # It should be a no-op for leaves
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    def test_retain_grad_cycle(self):
        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            y = x * 2
            y.retain_grad()

            return y / 2, torch._C._WeakTensorRef(y)

        z, ref = run_test()
        self.assertTrue(ref.expired())
        z.sum().backward()

    def test_backward(self):
        v = torch.randn(5, 5, requires_grad=True)
        x = torch.randn(5, 5, requires_grad=True)
        y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
        z = torch.randn(5, 5, requires_grad=True)
        grad_output = torch.randn(5, 5)

        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)

        a = x + (y * z) + 4 * z ** 2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z.pow(2) / y + 1
        y_grad = z - 4 * x * z.pow(2) / y.pow(2)
        z_grad = 8 * x * z / y + y
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_sparse_mm_backward(self):
        size = (3, 3)
        sparse = torch.sparse_coo_tensor(size, requires_grad=True)
        dense = torch.randn(size, requires_grad=True)

        with self.assertRaisesRegex(
                RuntimeError,
                "The backward pass for this operation requires the 'mat1' tensor to be strided,"):
            z = dense.addmm(sparse, dense)

        mm_test_cases = [
            # a requires grad, a is sparse, b requires grad, b is sparse, error message
            (False, True, True, False, None),
            (False, False, True, True, "The backward pass for this operation requires the 'mat2'"),
            (False, True, True, True, "The backward pass for this operation requires the 'mat2'"),
            (True, False, True, True, "The backward pass for this operation requires the 'mat2'"),
            (True, True, False, False, "The backward pass for this operation requires the 'self'"),
            (True, True, True, False, "The backward pass for this operation requires the 'self'"),
            (True, True, True, True, "The backward pass for this operation requires the 'mat2'"),
        ]
        for a_req_grad, a_is_sparse, b_req_grad, b_is_sparse, err_msg in mm_test_cases:
            # We should only be testing cases with sparse inputs, and at least one
            # input needs to require grad so we can call a backward pass
            assert a_is_sparse or b_is_sparse
            assert a_req_grad or b_req_grad

            a = torch.randn(size, requires_grad=a_req_grad)
            if a_is_sparse:
                a = a.to_sparse()
            b = torch.randn(size, requires_grad=b_req_grad)
            if b_is_sparse:
                b = b.to_sparse()

            # If no error expected, check that sparse and dense cases match
            if err_msg is None:
                r = a.mm(b)
                r.sum().backward()
                a_grad = None if a.grad is None else a.grad.clone().detach()
                b_grad = None if b.grad is None else b.grad.clone().detach()

                # Redo with only dense tensors
                a = (a.to_dense() if a.is_sparse else a).clone().detach()
                a.requires_grad = a_req_grad
                b = (b.to_dense() if b.is_sparse else b).clone().detach()
                b.requires_grad = b_req_grad
                r = a.mm(b)
                r.sum().backward()

                self.assertEqual(a_grad, a.grad)
                self.assertEqual(b_grad, b.grad)

            else:
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mm(b)

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

        self.assertEqual(x.grad, grad_z)
        self.assertEqual(y.grad, grad_z)
        self.assertEqual(a.grad, grad_c * b)
        self.assertEqual(b.grad, grad_c * a)
        self.assertEqual(q.grad, (grad_c + grad_z) * 2)

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

    def test_backward_with_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        def fn():
            return x ** 2 + y * x + y ** 2

        gradient = torch.ones(2, 2)
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        @torch.no_grad()
        def reset_grad():
            x.grad.zero_()
            y.grad.zero_()

        torch.autograd.backward(fn(), gradient, inputs=[x, y])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, y_grad_expected)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[x])
        self.assertEqual(x.grad, x_grad_expected)
        self.assertEqual(y.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=[y])
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        torch.autograd.backward(fn(), gradient, inputs=y)
        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        reset_grad()
        self.assertRaisesRegex(RuntimeError, 'cannot be empty',
                               lambda: torch.autograd.backward(fn(), gradient, inputs=[]))

    def test_backward_with_nonleaf_inputs(self):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x_nonleaf = x * 1
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        out = x_nonleaf ** 2 + y * x_nonleaf + y ** 2

        out.backward(torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[x, y])
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        self.assertEqual(y.grad, y_grad_expected)
        self.assertEqual(x.grad, x_grad_expected)

        self.assertRaisesRegex(RuntimeError, 'not a leaf Tensor',
                               lambda: out.backward(torch.ones(2, 2, dtype=torch.double),
                                                    create_graph=True, inputs=[x, y, x_nonleaf]))

        # backward doesn't have an allow_unused flag, so the behavior of backward
        # when variable is not part of the graph is as if allow_used were true
        # x.grad will simply be None.
        out.backward(torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[z])
        self.assertIsNone(z.grad)

    def test_dependent_backward(self):
        x = torch.randn(10, requires_grad=True)
        y = x ** 2
        z = y ** 3

        go_y = torch.randn(10)
        go_z = torch.randn(10)
        torch.autograd.backward([y, z], [go_y, go_z])

        xd = x
        self.assertEqual(x.grad, 2 * xd * go_y + 6 * xd.pow(5) * go_z)

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
            for _ in range(depth):
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
            for _ in range(depth):
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
            for _ in range(depth):
                y = MyOp.apply(y, y)

            # graph deletion occurs when the above locals go out of scope.

        # Should not stack overflow
        scope()

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
            ft = torch.tensor([float(i)])
            multiplied = x * ft
            s = multiplied.sum()
            loss += s
        loss.backward()

    def test_no_grad(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
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

    def test_set_grad_generator_functions(self):
        @torch.no_grad()
        def gen_no_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), False)
                yield i

        with torch.enable_grad():
            for _ in gen_no_grad():
                self.assertEqual(torch.is_grad_enabled(), True)

        @torch.enable_grad()
        def gen_enable_grad():
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), True)
                yield i

        with torch.no_grad():
            for _ in gen_enable_grad():
                self.assertEqual(torch.is_grad_enabled(), False)

    def test_set_grad_generator_functions_recursive(self):
        # enable_grad_decorator_recursive and no_grad_decorator_recursive call each other
        # recursively, to ensure that the decorators preserve the caller's setting
        @torch.enable_grad()
        def enable_grad_decorator_recursive(depth):
            self.assertTrue(torch.is_grad_enabled())
            if depth > 0:
                no_grad_decorator_recursive(depth - 1)
                self.assertTrue(torch.is_grad_enabled())

        @torch.no_grad()
        def no_grad_decorator_recursive(depth):
            self.assertFalse(torch.is_grad_enabled())
            if depth > 0:
                enable_grad_decorator_recursive(depth - 1)
                self.assertFalse(torch.is_grad_enabled())

        # enable_grad_context_manager_recursive and no_grad_context_manager_recursive call
        # each other recursively, to ensure that the decorators preserve the caller's setting
        def enable_grad_context_manager_recursive(depth):
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if depth > 0:
                    no_grad_context_manager_recursive(depth - 1)
                    self.assertTrue(torch.is_grad_enabled())

        def no_grad_context_manager_recursive(depth):
            with torch.no_grad():
                self.assertFalse(torch.is_grad_enabled())
                if depth > 0:
                    enable_grad_context_manager_recursive(depth - 1)
                    self.assertFalse(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertFalse(torch.is_grad_enabled())

    def test_set_grad_coroutines(self):
        @torch.no_grad()
        def coro_no_grad(n=10):
            self.assertFalse(torch.is_grad_enabled())
            for i in range(n):
                self.assertFalse(torch.is_grad_enabled())
                r = yield i
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertFalse(torch.is_grad_enabled())

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            self.assertTrue(torch.is_grad_enabled())
            for i in range(n):
                self.assertTrue(torch.is_grad_enabled())
                r = yield i
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(i, r)
            self.assertTrue(torch.is_grad_enabled())

        with torch.enable_grad():
            self.assertTrue(torch.is_grad_enabled())
            coro, r = coro_no_grad(), None
            try:
                while True:
                    self.assertTrue(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertTrue(torch.is_grad_enabled())

            except StopIteration:
                pass

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            coro, r = coro_enable_grad(), None
            try:
                while True:
                    self.assertFalse(torch.is_grad_enabled())
                    r = coro.send(r)
                    self.assertFalse(torch.is_grad_enabled())

            except StopIteration:
                pass

    def test_set_grad_coroutines_benign_exceptions(self):
        class RecoverableException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    has_raised = True

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except RecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    has_raised = True

        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            try:
                while True:
                    r = coro.throw(RecoverableException)
                    self.assertLess(r, 0)

            except StopIteration:
                pass

    def test_set_grad_coroutines_critical_exceptions(self):
        class UnrecoverableException(Exception):
            pass

        class SecondaryException(Exception):
            pass

        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertFalse(torch.is_grad_enabled())
                    raise SecondaryException

        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    self.assertTrue(torch.is_grad_enabled())
                    raise SecondaryException

        with torch.enable_grad():
            coro = coro_no_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

        with torch.no_grad():
            coro = coro_enable_grad()
            assert 0 == next(coro)
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

    def test_set_grad_coroutines_exit(self):
        @torch.no_grad()
        def coro_no_grad(state):
            for i in range(10):
                try:
                    self.assertFalse(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertFalse(torch.is_grad_enabled())
                    state.add('GeneratorExit')
                    raise

        @torch.enable_grad()
        def coro_enable_grad(state):
            for i in range(10):
                try:
                    self.assertTrue(torch.is_grad_enabled())
                    yield i

                except GeneratorExit:
                    self.assertTrue(torch.is_grad_enabled())
                    state.add('GeneratorExit')
                    raise

        state = set()
        with torch.enable_grad():
            coro = coro_no_grad(state)
            for i in range(5):
                next(coro)

            coro.close()
        self.assertTrue('GeneratorExit' in state)

        state = set()
        with torch.no_grad():
            coro = coro_enable_grad(state)
            for i in range(5):
                next(coro)

            coro.close()
        self.assertTrue('GeneratorExit' in state)

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
            expected_grad = torch.empty(x.size()).fill_(0)
            expected_grad[idx] = 1
            self.assertEqual(y.grad, expected_grad)

        def check_index(x, y, idx):
            if y.grad is not None:
                with torch.no_grad():
                    y.grad.zero_()
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
        check_index(x, y, torch.rand(4, 4).bernoulli().bool())
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
            with torch.no_grad():
                y.grad.zero_()
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
        self.assertEqual(y.grad, expected_grad)

        # with advanced indexing
        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        idx = [[1, 1, 3, 2, 1, 2], [0]]
        y[idx].sum().backward()
        expected_grad = torch.zeros(4, 4)
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1., 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        idx = [[[1, 2], [0, 0]], [[0, 1], [1, 1]]]
        y[idx].sum().backward()
        expected_grad = torch.tensor([[0., 2., 0., 0.],
                                      [1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 0., 0.]])
        self.assertEqual(y.grad, expected_grad)

        x = torch.arange(1., 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)

        idx = [[1, 1, 1], slice(None), slice(None)]
        y[idx].sum().backward()
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        self.assertEqual(y.grad, expected_grad)

    def test_index_backward_does_not_save_tensor(self):
        # Example from https://github.com/pytorch/pytorch/issues/24853.
        # if `index(tensor, indices)` saves `tensor` for backwards, then it will
        # trigger a version check on `tensor` during the backward pass, which
        # will cause the following code to error because `tensor` gets modified
        # by the indexing line.
        a = torch.tensor([1., 0, 0])
        b = torch.zeros(3, requires_grad=True)
        tensor = b + 0
        tensor[a != 0] = tensor[a != 0]
        tensor.backward(torch.zeros_like(tensor))

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
            MyFunction.apply(x, y).sum().backward()

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

        # non-leaf
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
        self.assertEqual(a.grad, torch.ones(2, 3))

        # same but with a view
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # should fail if requires_grad = True when we modify inplace
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    def test_attribute_deletion(self):
        x = torch.randn((5, 5), requires_grad=True)
        del x.grad
        self.assertIsNone(x.grad)
        with self.assertRaises(RuntimeError):
            del x.data
        with self.assertRaises(TypeError):
            x.data = None
        with self.assertRaises(RuntimeError):
            del x.requires_grad
        with self.assertRaises(RuntimeError):
            del x._grad_fn
        with self.assertRaises(RuntimeError):
            del x._backward_hooks

    def test_duplicate_backward_root(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        x = a * b
        grad_output = torch.randn_like(x)
        torch.autograd.backward([x, x], [grad_output, grad_output])

        self.assertEqual(a.grad, b * grad_output * 2)
        self.assertEqual(b.grad, a * grad_output * 2)

    def test_backward_no_grad(self):
        a = torch.randn(5, 5, requires_grad=True)
        b = a + 2
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    def test_backward_twice_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True',
                               lambda: c.backward(torch.tensor([1, 1, 1], dtype=torch.double)))

    def test_backward_twice_retained_graph_with_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = b + 1
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    def test_backward_twice_retained_graph_without_saved_values(self):
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        c = torch.zeros(3, dtype=torch.double)
        c[[1, 2]] = b[[1, 1]]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

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

        with torch.no_grad():
            x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        prev_version = z._version
        w = z.exp_()
        self.assertNotEqual(z._version, prev_version)
        r.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(x.grad, torch.empty(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = torch.ones(5, 5, requires_grad=True)
        x = leaf.clone()
        x.add_(10)
        self.assertEqual(x, torch.ones(5, 5) * 11)
        # x should be still usable
        y = x + 2
        y.backward(torch.ones(5, 5))
        self.assertEqual(leaf.grad, torch.ones(5, 5))
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
                return (grad_output * 0).to(torch.double)

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
        self.assertEqual(x.grad, torch.ones(5, 5))

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

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
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

        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
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
        mask = torch.BoolTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        # both select and sum return Scalars in ATen; ensure they work together.
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        def func(x):
            return x.select(0, 1).sum()

        gradcheck(func, [x])
        gradgradcheck(func, [x])

    def test_diagonal_expanded_v(self):
        value = torch.rand([])
        v_expanded = torch.tensor(value).expand(10)
        a = torch.rand(10, 10, dtype=torch.double, requires_grad=True)
        result, = torch.autograd.grad(a.diagonal(), a, v_expanded)
        self.assertEqual(result, torch.eye(10, dtype=torch.double) * value)

    def test_select_expanded_v(self):
        v_expanded = torch.rand(10).expand(10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        result, = torch.autograd.grad(a[0], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[0] = v_expanded
        self.assertEqual(result, expected)

    def test_slice_expanded_v(self):
        v_expanded = torch.rand(10, 1).expand(2, 10, 10)
        a = torch.rand(10, 10, 10, requires_grad=True)
        result, = torch.autograd.grad(a[3:5], a, v_expanded)
        expected = torch.zeros(10, 10, 10)
        expected[3:5] = v_expanded
        self.assertEqual(result, expected)

    def test_unbind(self):
        stacked = torch.randn(3, 10, 10, requires_grad=True)
        x, y, z = stacked.unbind()
        grad = torch.randn(3, 10, 10)
        torch.autograd.backward([x, y, z], grad.unbind())
        self.assertEqual(stacked.grad, grad)
        # check that it works with only one gradient provided (#9977)
        for i in range(3):
            stacked = torch.randn(3, 10, 10, requires_grad=True)
            outs = stacked.unbind()
            gi = grad.unbind()[i]
            g, = torch.autograd.grad(outs[i], stacked, gi)
            g_expected = torch.stack([gi if j == i else torch.zeros_like(gi)
                                      for j in range(3)], dim=0)
            self.assertEqual(g, g_expected)

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
        self.assertEqual(x.grad, expected_grad)

        with torch.no_grad():
            x.grad.zero_()
        grad_output = torch.randn(2, 10)
        outputs = x.chunk(5)
        outputs[0].backward(grad_output)
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        self.assertEqual(x.grad, expected_grad)

    def _test_sparse_gather(self, size_x, size_ind, dim):
        x = torch.randn(size_x, requires_grad=True)
        if len(size_ind) > 0 and len(size_x) > 0:
            ind = torch.randint(x.size(dim), size_ind)
        else:
            ind = torch.zeros(size_ind, dtype=torch.int64)
        out = torch.gather(x, dim, ind, sparse_grad=False)
        grad = torch.rand_like(out)
        out.backward(grad)
        grad_dense = x.grad.clone()
        x.grad = None
        out = torch.gather(x, dim, ind, sparse_grad=True)
        out.backward(grad)
        self.assertEqual(grad_dense, x.grad.to_dense())

    def test_sparse_gather_dim0(self):
        self._test_sparse_gather((10, 10), (5, 10), 0)

    def test_sparse_gather_dim1(self):
        self._test_sparse_gather((10, 10, 5), (10, 5, 5), 1)

    def test_sparse_gather_dim_neg(self):
        self._test_sparse_gather((10, 10, 5), (10, 10, 2), -1)

    def test_sparse_gather_ind_scalar(self):
        self._test_sparse_gather((10,), (), 0)

    def test_sparse_gather_x_scalar(self):
        self._test_sparse_gather((), (2,), 0)

    def test_sparse_gather_both_scalar(self):
        self._test_sparse_gather((), (), 0)

    def test_gc_in_destructor(self):
        """
        Previously, if a Function destructor triggered a garbage collection,
        the Variable's tp_dealloc handler would get called twice leading to a
        segfault.
        """
        class CollectOnDelete(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

            def __del__(self):
                gc.collect()

        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    def test_naughty_autograd_function_attribute_access(self):
        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_x):
                return grad_x

        with self.assertWarnsRegex(DeprecationWarning, "should not be instantiated"):
            f = Id()

        # # After raising warning, should still return an instance
        self.assertIsInstance(f, Id)
        x = torch.zeros(1, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "non-static forward method is deprecated"):
            f(x)
        t = Id.apply(x)
        self.assertEqual(t.grad_fn.name(), "IdBackward")

        # THPFunction is the base class of both grad_fn and autograd functions,
        # which means that a lot of accessors on them may segfault. Test that we
        # properly error in this case.
        t = torch.ones(1, requires_grad=True)
        t._backward_hooks = dict()
        with self.assertRaisesRegex(RuntimeError, "Attribute '_register_hook_dict' is invalid"):
            f._register_hook_dict(t)
        with self.assertRaisesRegex(RuntimeError, "Attribute 'register_hook' is invalid"):
            f.register_hook(lambda x, y: None)
        with self.assertRaisesRegex(RuntimeError, "Attribute 'next_functions' is invalid"):
            f.next_functions
        with self.assertRaisesRegex(RuntimeError, "Attribute 'name' is invalid"):
            f.name()
        with self.assertRaisesRegex(RuntimeError, "underlying PyNode has already been deallocated"):
            f.metadata

    @unittest.expectedFailure
    def test_naughty_anomaly_access(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, g):
                return g

        x = torch.zeros(1, requires_grad=True)
        y = MyFunction.apply(x)
        y.backward()
        y.grad_fn.metadata
        g = y.grad_fn
        del y
        g.metadata  # this currently fails, but shouldn't

    def test_naughty_autograd_function_stashing_ctx(self):
        saved_ctx = []

        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                saved_ctx.append(ctx)
                return ctx.saved_tensors

        p = torch.zeros(1, requires_grad=True)
        loss = Id.apply(p)
        loss.backward(retain_graph=True)
        del loss
        # At this point in time, it complains that the graph has been freed
        # (which indeed true, although a somewhat indirect way of stating the
        # problem).
        self.assertRaises(RuntimeError, lambda: saved_ctx[0].saved_tensors)

    def test_custom_autograd_repeated_grad_grad(self):
        # This test failed the equality check in PR #22983; it's an interesting
        # and different test case worth enshrining.  mult1 is not testing
        # anything that interesting, but mult2 is the interesting case.

        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        class Mult(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        def check_gradgrad_repeated(x, y):
            gy, = torch.autograd.grad(y[0], x, create_graph=True)
            ggy_1, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            gy, = torch.autograd.grad(y[0], x, create_graph=True)
            ggy_2, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])

        x = torch.ones(2, 4, 4).requires_grad_()
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))

    def test_custom_autograd_no_early_free(self):
        # This test failed complaining that buffers had already been freed
        # prior to #22983.  Also pretty interesting test case.
        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x ** 2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        x = torch.tensor(2).double().requires_grad_()

        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))

        y = double(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)

        y = double2(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)  # should not error!

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
        self.assertEqual(x.grad, torch.ones(10, 10))

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)

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
        self.assertEqual(type(fvar.grad), type(fvar))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

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
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().get_device(), 1)

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
        err_msg_outputs = r"Can't call numpy\(\) on Tensor that requires grad. Use tensor.detach\(\).numpy\(\) instead."
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()

        with torch.no_grad():
            x.numpy()

        x = torch.randn(2, 2)
        x.numpy()

        with torch.no_grad():
            x.numpy()

    def test_return_leaf(self):
        class Identity(Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        hook_called = [False]
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Identity.apply(x, y)

        # Make sure hooks only receive grad from usage of q, not x.
        def hook(grad):
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))

        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        class Inplace(InplaceFunction):
            @staticmethod
            def forward(ctx, a, b):
                ctx.mark_dirty(a)
                return a.add_(b), b + 2

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)

        q, p = Inplace.apply(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn.__class__, Inplace._backward_cls)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad, torch.ones(5))
        self.assertEqual(z.grad, torch.ones(5) * 2)

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
        for _ in range(4):
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
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(None, input, None)
                return input * input

            @staticmethod
            def backward(ctx, grad_output):
                n1, input, n2 = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None, None

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))

    def test_pickle(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    def test_dep_nograd(self):
        class F1(Function):
            @staticmethod
            def forward(ctx, input):
                out = torch.randn(input.size())
                ctx.mark_non_differentiable(out)
                return input, out

            @staticmethod
            def backward(ctx, grad_output, ignored):
                return grad_output

        class F2(Function):
            @staticmethod
            def forward(ctx, input, ignored):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None

        x = torch.randn(5, requires_grad=True)
        a, b = F1.apply(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2.apply(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad, torch.ones(x.size()))

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

    def test_simple_reentrant(self):
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        # Reentrant starts on CPU thread, finishs on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        # Parent graph.
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a

        # Reentrant child graph.
        b = torch.rand(3, 3, requires_grad=True)
        e = b * b
        f = TestAutograd.SimulateBackwardError.apply(e)
        reentrant_root = f.sum()

        class ReentrantFunc(Function):

            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will throw an error.
                reentrant_root.backward()
                return grad

        d = ReentrantFunc.apply(c)
        with self.assertRaisesRegex(Exception, 'Simulate error'):
            d.sum().backward()

    def test_broadcast_tensors(self):
        f_args_variable = (torch.randn(3, dtype=torch.double, requires_grad=True),
                           torch.randn(1, 2, 1, dtype=torch.double, requires_grad=True),
                           torch.randn(1, 1, dtype=torch.double, requires_grad=True),
                           torch.randn(5, 1, 1, dtype=torch.double, requires_grad=True))
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_broadcast_tensors", "broadcast",
                              lambda a, b, c, d: torch.broadcast_tensors(a, b, c, d),
                              True, f_args_variable, f_args_tensor)

    def test_block_diag(self):
        f_args_variable = (torch.randn(1, S, dtype=torch.double, requires_grad=True),
                           torch.randn(2, S, dtype=torch.double, requires_grad=True),
                           torch.randn(3, S, dtype=torch.double, requires_grad=True))
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_block_diag", "block_diag",
                              lambda a, b, c: torch.block_diag(a, b, c),
                              True, f_args_variable, f_args_tensor)

    def test_cat(self):
        f_args_variable = (torch.randn(1, S, S, dtype=torch.double, requires_grad=True),
                           torch.randn(2, S, S, dtype=torch.double, requires_grad=True),
                           torch.randn(3, S, S, dtype=torch.double, requires_grad=True),
                           0)
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat", "cat",
                              lambda a, b, c, dim: torch.cat((a, b, c), dim),
                              True, f_args_variable, f_args_tensor)

    def test_cat_negdim_1(self):
        f_args_variable = (torch.randn(S, S, 1, dtype=torch.double, requires_grad=True),
                           torch.randn(S, S, 2, dtype=torch.double, requires_grad=True),
                           torch.randn(S, S, 3, dtype=torch.double, requires_grad=True),
                           -1)
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_negdim_1", "cat",
                              lambda a, b, c, dim: torch.cat((a, b, c), dim),
                              True, f_args_variable, f_args_tensor)

    def test_cat_negdim_2(self):
        f_args_variable = (torch.randn(S, 1, S, dtype=torch.double, requires_grad=True),
                           torch.randn(S, 2, S, dtype=torch.double, requires_grad=True),
                           torch.randn(S, 3, S, dtype=torch.double, requires_grad=True),
                           -2)
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_negdim_2", "cat",
                              lambda a, b, c, dim: torch.cat((a, b, c), dim),
                              True, f_args_variable, f_args_tensor)

    def test_cat_empty_legacy(self):
        f_args_variable = (torch.randn(0, dtype=torch.double, requires_grad=True),
                           torch.randn(S, S, dtype=torch.double, requires_grad=True))
        # gradgradcheck doesn't work, probably because legacy size tracking is wrong somewhere,
        # hence False passed below, but gradcheck checked explicitly.
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_empty_legacy", "cat",
                              lambda a, b: torch.cat((a, b)),
                              False, f_args_variable, f_args_tensor)
        self.assertTrue(gradcheck(lambda a, b: torch.cat((a, b)), f_args_variable, eps=1e-6, atol=PRECISION))

    def test_cat_empty(self):
        f_args_variable = (torch.randn(0, S, dtype=torch.double, requires_grad=True),
                           torch.randn(S, S, dtype=torch.double, requires_grad=True))
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat_empty", "cat",
                              lambda a, b: torch.cat((a, b)),
                              True, f_args_variable, f_args_tensor)

    def test_trapz(self):
        f_args_variable = (torch.randn(2, 3, dtype=torch.double, requires_grad=True),
                           torch.tensor([[1.0, 2.0, 5.5], [2.3, 0.5, 6.2]], dtype=torch.double, requires_grad=True))
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_trapz", "trapz",
                              lambda y, x: torch.trapz(y, x),
                              True, f_args_variable, f_args_tensor)


    def test_var_mean_differentiable(self):
        dim = [2, 4]
        keepdim = False
        input1 = torch.randn(3, 4, 5, 6, 2, 3, requires_grad=True)
        input2 = deepcopy(input1)
        var1, mean1 = torch.var_mean(input1, dim=dim, keepdim=keepdim)
        var2 = input2.var(dim=dim, keepdim=keepdim)
        mean2 = input2.mean(dim=dim, keepdim=keepdim)
        grad = torch.randn(3, 4, 6, 3, requires_grad=True)

        r1 = var1 * var1 * mean1 * mean1
        r2 = var2 * var2 * mean2 * mean2
        self.assertTrue(torch.allclose(r1, r2, rtol=0.01, atol=0.0))

        torch.autograd.backward(r1, grad)
        torch.autograd.backward(r2, grad)
        self.assertTrue(torch.allclose(input1.grad, input2.grad, rtol=0.01, atol=0.0))

    @slowTest
    @skipIfNoLapack
    def test_lobpcg(self):

        def func(k, A, largest=True, B=None):
            X_shape = list(A.shape)
            X_shape[-1] = k
            X = torch.eye(A.size(-2), k, dtype=A.dtype, device=A.device)
            if A.dim() > 2:
                X = X.expand(X_shape)

            D, U = torch.lobpcg(A=A, k=k, B=B, X=X)

            # LOBPCG uses a random initial eigenspace approximation
            # if parameter `X` is not provided.
            # This may cause a non-deterministic behavior
            # when it comes to the sign of an eigenvector
            # (note if v is an eigenvector, so is -v),
            # hence we eliminate this non-determinism
            # by making sure that each column of U
            # gets multiplied by the sign of its max (in absolute value) element.
            # Also, gradcheck changes the content of the input by +/- eps (default to 1e-06)
            # to compute the numerical gradient which can also cause the signs to flip.
            _, idx = U.abs().max(-2, keepdim=True)
            sign = U.gather(-2, idx).sign()
            U = U * sign
            return D, U

        def run_symeig_test(k, sizes, largest=True):
            A = torch.rand(*sizes).double()
            A = A.matmul(A.transpose(-1, -2)) / 10
            A.requires_grad_(True)

            gradcheck(lambda A: func(k, A, largest), A, check_batched_grad=False)

            # Custom gradient vectors for better stability due to some
            # non-determinism in the lobpcg's forward.
            # Note it is not required if symeig is in forward instead (tested).
            D_grad = torch.rand(*A.shape[:-2], k) / 100
            U_grad = torch.rand(*A.shape[:-1], k) / 100
            gradgradcheck(lambda A: func(k, A, largest), A, [D_grad, U_grad], atol=1e-4, check_batched_grad=False)

            # check whether A.grad is symmetric
            A = A.detach().requires_grad_(True)
            D, U = func(k, A, largest)
            (D.sum() + U.sum()).backward()
            self.assertEqual(A.grad, A.grad.transpose(-1, -2))

        # the tests below take about 1-2 minutes to finish,
        # but we want to be extra sure that the backward is correct.
        for largest in [True, False]:
            run_symeig_test(1, (6, 6), largest=largest)
            run_symeig_test(1, (2, 6, 6), largest=largest)
            run_symeig_test(1, (2, 2, 6, 6), largest=largest)
            run_symeig_test(2, (6, 6), largest=largest)
            run_symeig_test(2, (2, 6, 6), largest=largest)
            run_symeig_test(2, (2, 2, 6, 6), largest=largest)
            run_symeig_test(3, (9, 9), largest=largest)
            run_symeig_test(3, (2, 9, 9), largest=largest)
            run_symeig_test(3, (2, 2, 9, 9), largest=largest)

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
            self.assertEqual(input.grad.abs().sum(), 0)

        run_test((10,), 2)
        run_test((10, 10), 2)
        run_test((10,), 3)
        run_test((10,), 1)
        run_test((10,), 1.5)
        run_test((10,), inf)

    def test_norm_inf_subgradient(self):
        def run_test(input, expected, dim=None):
            x = torch.tensor(input, requires_grad=True)
            out = x.norm(inf, dim=dim, keepdim=True)
            out.backward(torch.ones(out.size()))
            self.assertEqual(x.grad, expected)

        run_test([0., 0., 0.], [0., 0., 0.])
        run_test([1., 0., 1.], [0.5, 0., 0.5])
        run_test([[1., 0., 1.], [0., 1., 1.]], [[0.25, 0., 0.25], [0., 0.25, 0.25]])
        run_test([[1., 0., 1.], [0., 1., 0.]], [[0.5, 0., 0.5], [0., 1., 0.]], (1,))
        run_test(torch.ones((2, 2, 2)), torch.full((2, 2, 2), 0.25), (0, 2))

    def test_pow_zero_tensor_gradient(self):
        def run_test(input_size, exponent):
            input = torch.zeros(*input_size, requires_grad=True)
            input.pow(exponent).sum().backward()
            self.assertEqual(input.grad.abs().sum(), 0)

        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    def test_pow_scalar_base(self):
        a = torch.arange(1, 13, dtype=torch.double).view(3, 4).requires_grad_()
        gradcheck(lambda a: torch.pow(2, a), (a,))

    def test_sinc(self):
        # The derivative of sinc(x) at x=0 has to be special cased.
        # A naive computation will result in 0/0 -> NaN.
        # We also need to be careful when we are very close to 0, as the
        # derivative's denominator is squared, and there are some floats
        # that are positive and whose squares are zero.
        a = torch.tensor([0.0, torch.finfo(torch.double).tiny, 1.0],
                         dtype=torch.double,
                         requires_grad=True)
        gradcheck(torch.sinc, a)

    def test_igamma(self):
        # 1e-3 offset to avoid zeros
        # NOTE: derivative for s is not implemented
        s = (torch.rand(100, dtype=torch.double) + 1e-3)
        x = (torch.rand(100, dtype=torch.double) + 1e-3).requires_grad_()
        gradcheck(torch.igamma, (s, x))
        gradgradcheck(torch.igamma, (s, x))

    def test_igammac(self):
        # 1e-3 offset to avoid zeros in s
        # NOTE: derivative for s is not implemented
        s = (torch.rand(100, dtype=torch.double) + 1e-3)
        x = (torch.rand(100, dtype=torch.double)).requires_grad_()
        gradcheck(torch.igamma, (s, x))
        gradgradcheck(torch.igamma, (s, x))

    def test_profiler(self):
        x = torch.randn(10, 10)

        with profile(use_kineto=kineto_available()) as p:
            self.assertTrue(torch.autograd._profiler_enabled())
            y = x * 2 + 4

        self.assertFalse(torch.autograd._profiler_enabled())

        names = ['aten::mul', 'aten::add']
        found_indices = set()
        for evt in p.function_events:
            if evt.name in names:
                found_indices.add(names.index(evt.name))
        self.assertEquals(len(found_indices), len(names))

    def test_profiler_seq_nr(self):
        with profile(use_kineto=kineto_available()) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = z.sum()
            s.backward()
        print(p.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=-1))
        # expecting aten::add, aten::sum to have the sequence numbers,
        # expecting the corresponding backward nodes to have the same numbers
        # as the forward ops
        add_seq_nr = -1
        sum_seq_nr = -1
        found_add = found_sum = False
        found_bwd_add = found_bwd_sum = False
        found_empty = False
        for e in p.function_events:
            if e.name == "aten::add":
                add_seq_nr = e.sequence_nr
                self.assertFalse(found_add)
                found_add = True
            elif e.name == "aten::sum":
                sum_seq_nr = e.sequence_nr
                self.assertFalse(found_sum)
                found_sum = True
            elif "Add" in e.name and "Backward" in e.name:
                self.assertEqual(e.sequence_nr, add_seq_nr)
                self.assertFalse(found_bwd_add)
                found_bwd_add = True
            elif "Sum" in e.name and "Backward" in e.name:
                self.assertEqual(e.sequence_nr, sum_seq_nr)
                self.assertFalse(found_bwd_sum)
                found_bwd_sum = True
            # check that nested ops (e.g. empty) don't have
            # sequence number
            if e.name == "aten::empty":
                self.assertEqual(e.sequence_nr, -1)
                found_empty = True
        self.assertGreaterEqual(add_seq_nr, 0)
        self.assertGreaterEqual(sum_seq_nr, 0)
        self.assertNotEqual(add_seq_nr, sum_seq_nr)
        self.assertTrue(found_add)
        self.assertTrue(found_sum)
        self.assertTrue(found_bwd_add)
        self.assertTrue(found_bwd_sum)
        self.assertTrue(found_empty)

    def test_profiler_unboxed_only(self):
        x = torch.rand(3, 4)

        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            x.resize_([3, 2])

    def test_profiler_propagation(self):
        def foo(x):
            with record_function("in_foo") as rf:
                return x * 2

        x = torch.rand(3, 4)
        traced_foo = torch.jit.trace(foo, x)

        def bar(x):
            with record_function("in_bar") as rf:
                # we expect that profiler will be able
                # propagate across fork
                fut = torch.jit._fork(traced_foo, x)
                y = torch.jit._wait(fut)
                # note: continuation (and rf's end) can
                # be executed in a different thread
                with record_function("in_bar_after_wait") as rf2:
                    y = y * 2
                return y

        traced_bar = torch.jit.trace(bar, x)

        with profile(use_kineto=kineto_available()) as p:
            traced_bar(x)

        found_foo = False
        found_bar = False
        found_bar_after_wait = False
        for info in p.function_events:
            if info.name == "in_foo":
                self.assertFalse(found_foo)
                found_foo = True
            elif info.name == "in_bar":
                self.assertFalse(found_bar)
                found_bar = True
            elif info.name == "in_bar_after_wait":
                self.assertFalse(found_bar_after_wait)
                found_bar_after_wait = True
        self.assertTrue(found_foo)
        self.assertTrue(found_bar)
        self.assertTrue(found_bar_after_wait)

    def test_record_function_callbacks(self):
        x = torch.randn(10, 10)
        with profile(use_kineto=kineto_available()) as p:
            with record_function("foo"):
                y = x * 2 + 4

        function_events = p.function_events
        foo_event = [event for event in function_events if "foo" in event.name][0]
        self.assertEqual(foo_event.count, 1)

    def test_profiler_aggregation_fake(self):
        events = EventList()
        id = [0]

        def get_id():
            id[0] = id[0] + 1
            return id[0]

        # [[thread_id, [(start, end, id), ....]], ...]
        # Using list instead of a dict so order is guaranteed for any Python
        # version
        threads = [
            [1, [(0, 1, get_id()), (1, 2, get_id())]],
            [0, [(0, 2, get_id()), (1, 2, get_id()), (1, 3, get_id())]],
        ]
        for thread, ranges in threads:
            for range in ranges:
                assert(len(range) == 3)
                events.append(
                    FunctionEvent(
                        id=range[2],
                        node_id=0,
                        name="",
                        thread=thread,
                        start_us=range[0],
                        end_us=range[1],
                    )
                )

        events._populate_cpu_children()

        # Note that [1, 3] pushes out [0, 2] first. Then we record [1, 2]
        # as a child of [1, 3]
        res = [[], [], [], [], [4]]

        def get_children_ids(event):
            return [child.id for child in event.cpu_children]

        assert([get_children_ids(event) for event in events] == res)

    def test_profiler_aggregation_table(self):
        """
        Test if the profiling result is aggregated for `str(prof)`

        See: https://github.com/pytorch/pytorch/issues/37500
        """

        x = torch.randn(1024)
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            torch.einsum("i->", x)

        prof_str = str(prof)
        prof_table = prof.table()

        self.assertEqual(prof_table, prof_str)

    def test_profiler_function_event_avg(self):
        avg = FunctionEventAvg()
        avg.add(FunctionEvent(id=0, node_id=0, name="foo", thread=0, start_us=10, end_us=15))
        avg.add(FunctionEvent(id=1, node_id=0, name="foo", thread=0, start_us=20, end_us=30))
        avg.add(avg)
        self.assertEqual(avg.key, "foo")

        # aggregate stats
        self.assertEqual(avg.count, 4)
        self.assertEqual(avg.cpu_time_total, 30)
        self.assertEqual(avg.self_cpu_time_total, 30)
        self.assertEqual(avg.cuda_time_total, 0)

        # average stats
        self.assertEqual(avg.cpu_time, 7.5)
        self.assertEqual(avg.cuda_time_total, 0)

    def test_profiler_shapes(self):
        print("")
        layer1 = torch.nn.Linear(20, 30)
        layer2 = torch.nn.Linear(30, 40)
        input = torch.randn(128, 20)
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            layer2(layer1(input))

        print(prof.function_events)

        linear_expected_shapes = [
            [[128, 20], [30, 20], [30]],
            [[128, 30], [40, 30], [40]],
        ]

        found_indices = set()
        for event in prof.function_events:
            if event.name == "aten::linear":
                self.assertTrue(event.input_shapes in linear_expected_shapes)
                found_indices.add(linear_expected_shapes.index(event.input_shapes))
        self.assertEqual(len(found_indices), len(linear_expected_shapes))

    def test_profiler_aggregation_lstm(self):
        print("")
        rnn = torch.nn.LSTM(10, 20, 2)
        total_time_s = 0
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            for i in range(20):
                input = torch.randn(5, 3, 10)
                h = torch.randn(2, 3, 20)
                c = torch.randn(2, 3, 20)
                start = time.time()
                rnn(input, (h, c))
                end = time.time()
                total_time_s += end - start

        print(prof.table(
            sort_by="self_cpu_time_total", row_limit=10, header="TEST"))
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=10))
        print(prof.table(
            sort_by="self_cpu_time_total", row_limit=10, max_src_column_width=300, header="TEST", top_level_events_only=True))
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total", row_limit=10, top_level_events_only=True))

        total_time_us = total_time_s * 1000.0 * 1000.0  # make it us which is profiler default
        print(
            "Total time based on python measurements: ",
            format_time(total_time_us)
        )
        print(
            "CPU time measurement python side overhead: {:.2f}%".format(
                (total_time_us / prof.self_cpu_time_total - 1.0) * 100.0
            )
        )

        if sys.platform != "win32":
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)

    def test_record_function(self):
        x = torch.randn(10, 10)

        def forward(x):
            with record_function("outer"):
                y = x * 2 + 4
                with record_function("inner"):
                    y = y - 1
            y = y / 1

        forward(x)

        with profile(use_kineto=kineto_available()) as p:
            forward(x)

        events = p.function_events
        important_events = [
            'outer',
            'aten::mul',
            'aten::add',
            'inner',
            'aten::sub',
            'aten::div'
        ]
        idx = 0
        for info in events:
            if info.name == important_events[idx]:
                idx = idx + 1
            if idx == len(important_events):
                break
        self.assertEqual(idx, len(important_events))

        # We can also use record_function to decorate arbitrary function
        @record_function('my_func')
        def f(x, y):
            return x + y

        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)

        self.assertTrue('my_func' in str(p))

    def test_record_function_multithreaded(self):
        rf = record_function("outer")
        rf.__enter__()
        with record_function("inner"):
            # test that exiting the record function after starting another one
            # doesn't throw.
            rf.__exit__(None, None, None)

        with record_function("inner"):
            rf.__enter__()
        # test that exiting the record function after ending another one
        # doesn't throw.
        rf.__exit__(None, None, None)


    def test_dir(self):
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn('shape', keys)

        # real and imag are only implemented for complex tensors.
        y = torch.randn(10, 10, dtype=torch.cfloat)
        for key in ['real', 'imag']:
            self.assertRaises(RuntimeError, lambda: hasattr(x, key))
            self.assertTrue(hasattr(y, key))
            keys.remove(key)

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

    def _test_lerp_tensor_weights(self, cast):
        def construct_inputs(*shapes):
            start = cast(torch.randn(shapes[0], dtype=torch.double)).requires_grad_()
            end = cast(torch.randn(shapes[1], dtype=torch.double)).requires_grad_()
            weight = cast(torch.randn(shapes[2], dtype=torch.double)).requires_grad_()
            return [start, end, weight]

        all_test_shapes = [((3, 3, 3), (3, 3, 3), (3, 3, 3)),  # no broadcasting
                           ((3,), (3, 3, 3), (3, 3, 3)),  # start broadcasting - 1
                           ((3, 3, 3), (3,), (3, 3, 3)),  # end broadcasting - 1
                           ((3, 3, 3), (3, 3, 3), (3,)),  # weight broadcasting - 1
                           ((), (3, 3, 3), (3, 3, 3)),  # start broadcasting - 2
                           ((3, 3, 3), (), (3, 3, 3)),  # end broadcasting - 2
                           ((3, 3, 3), (3, 3, 3), ()),  # weight broadcasting - 2
                           ((3, 3), (3, 3, 3), (3,))]  # all broadcasting

        for shapes in all_test_shapes:
            cur_inputs = construct_inputs(*shapes)
            gradcheck(torch.lerp, cur_inputs)
            gradgradcheck(torch.lerp, cur_inputs)

    def test_lerp_tensor_weights(self):
        self._test_lerp_tensor_weights(lambda t: t)

    def test_reduce_dtype(self):
        def test_reduction(op, has_no_dim, takes_dtype=True):
            x = torch.randn(3, 3, dtype=torch.float, requires_grad=True)

            if has_no_dim:
                grad1, = torch.autograd.grad([op(x)], [x])
                grad2, = torch.autograd.grad([op(x, dtype=torch.double)], [x])
                self.assertEqual(grad1, grad2)
                self.assertEqual(grad2.dtype, torch.float)

            gi = torch.randn(op(x, dim=0).shape, dtype=torch.float)
            grad1, = torch.autograd.grad([op(x, dim=0)], [x], gi)
            if takes_dtype:
                grad2, = torch.autograd.grad([op(x, dim=0, dtype=torch.double)], [x], gi.double())
            else:
                grad2, = torch.autograd.grad([op(x.double(), dim=0)], [x], gi.double())
            self.assertEqual(grad1, grad2)
            self.assertEqual(grad2.dtype, torch.float)

        test_reduction(torch.sum, True)
        test_reduction(torch.prod, True)
        test_reduction(torch.cumsum, False)
        test_reduction(torch.cumprod, False)
        test_reduction(torch.logcumsumexp, False, takes_dtype=False)

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

    def test_inplace_view_leaf_errors(self):
        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        x = torch.zeros(1, requires_grad=True)
        y = x.view_as(x)
        with self.assertRaisesRegex(RuntimeError,
                                    "a view of a leaf Variable that "
                                    "requires grad is being used in "
                                    "an in-place operation."):
            y.add_(1)

    def test_inplace_view_backward(self):
        # Issue #10532: Make sure that this does not raise RuntimeError.
        net = nn.Sequential(
            nn.InstanceNorm2d(2),
            nn.ReLU(True)
        )

        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        g, = torch.autograd.grad(net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape) , create_graph=True)
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))

        # https://discuss.pytorch.org/t/freeing-buffer-strange-behavior/31955/8
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)

        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0., 0., True)
        prob_interpolated = torch.sigmoid(tmp2)

        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=inputs,
                                        grad_outputs=torch.ones(prob_interpolated.size()),
                                        create_graph=True, retain_graph=True)[0]

        gradient_penalty = gradients.sum()
        gradient_penalty.backward()

        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), "ThresholdBackwardBackward")

    def test_inplace_view_weak_grad_fn(self):
        # Issue 23502: Test that b's grad_fn is preserved.
        a = torch.arange(10.0, requires_grad=True)

        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()

        c = b.clone()
        del b
        gc.collect()

        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))

        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()

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

    def test_nested_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, fail_0th):
                ctx.fail_0th = fail_0th
                ctx.save_for_backward(inp1)
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                inp, = ctx.saved_tensors
                fail_0th = ctx.fail_0th
                g = gO.clone().expand(size)
                gI = MyFunc2.apply(g * inp, g + inp, fail_0th)
                return gI, None

        class MyFunc2(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1 * 2.0 + inp2

            @staticmethod
            def backward(ctx, gO):
                fail_0th = ctx.fail_0th
                g1 = gO.clone()
                g2 = gO.clone()
                g1[0] = 0
                g2[0] = 0
                # generate a nan
                if fail_0th:
                    g1[0] /= 0
                else:
                    g2[0] /= 0
                return g1, g2, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        ginp, = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        gsum.backward()  # should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, True)
        ginp, = torch.autograd.grad(out, (inp,), create_graph=True)
        gsum = ginp.sum()
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 0th output."):
                with detect_anomaly():
                    gsum.backward()
        self.assertIn('No forward pass information', str(w[1].message))

        inp = torch.rand(size, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 1th output."):
                with detect_anomaly():
                    out = MyFunc.apply(inp, False)
                    ginp, = torch.autograd.grad(out, (inp,), create_graph=True)
                    gsum = ginp.sum()
                    gsum.backward()
        self.assertIn('MyFunc2.apply', str(w[1].message))
        self.assertIn('MyFunc.apply', str(w[2].message))

    def test_anomaly_grad_warnings(self):
        # PyTorch won't throw warnings if there is an error
        # but we'd want to at least see them in stderr

        class StdErrDiverter:
            def __enter__(self):
                self.stderr_orig = sys.stderr
                self.stderr_new = io.StringIO()
                sys.stderr = self.stderr_new
                return self

            def __exit__(self, *args):
                self.captured = self.stderr_new.getvalue()
                sys.stderr = self.stderr_orig


        # if the warnings don't throw, they will be handled as regular warnings
        with self.assertRaisesRegex(RuntimeError,
                                    "one of the variables needed for gradient computation has been "
                                    "modified by an inplace operation"):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    a = torch.randn(5, requires_grad=True)
                    d1 = a + 1
                    d2 = d1 ** 2
                    d1 += 1
                    torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 2)
        self.assertIn('Anomaly Detection has been enabled', str(w[0].message))
        self.assertIn('Error detected in PowBackward0', str(w[1].message))

        # if the warning throws, it will be printed to sys.stderr
        with self.assertRaisesRegex(RuntimeError,
                                    "one of the variables needed for gradient computation has been "
                                    "modified by an inplace operation"):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    warnings.simplefilter("error")
                    with StdErrDiverter() as s:
                        a = torch.randn(5, requires_grad=True)
                        d1 = a + 1
                        d2 = d1 ** 2
                        d1 += 1
                        torch.autograd.grad(d2.sum(), a)

        self.assertEqual(len(w), 1)
        self.assertIn('Anomaly Detection has been enabled', str(w[0].message))
        self.assertIn('Error detected in PowBackward0', s.captured)

    def test_anomaly_assign_parent_cleanup(self):
        # Test that python objects created are properly cleaned up when assign_parent is called
        import weakref

        def get_ref():
            # we use torch.exp here but any function that will construct a new node in its
            # backward call in grad mode will work
            x = torch.randn(2, 2, requires_grad=True)
            t = x.exp()

            # ExpBackward calls mul, creating the MulBackward node when create_graph=True.
            # In anomaly mode, a PyObject referencing MulBackward's "parent" ExpBackward is added to
            # MulBackward's anomaly metadata dict, creating the following reference chain:
            #
            # grad -> MulBackward -> PyObject -> ExpBackward
            #
            with detect_anomaly():
                grad = torch.autograd.grad(t, x, torch.ones_like(t), create_graph=True)

            # We add a weak reference to a new Foo object, which we insert into ExpBackward's metadata dict
            #
            # (PyObject) -> ExpBackward -> dict -> *Foo*
            #            t ----^        WeakRef ---^
            #
            # We want to test that when grad goes out of scope at the end of this function that PyObject is destroyed
            # We can test this by seeing whether Foo is not kept alive once t is destroyed
            class Foo(object):
                pass
            my_obj = Foo()
            meta_dict = t.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return t, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    def test_nested_anomaly_printstack_cleanup(self):
        # Test if metadata dict PyObject is properly destroyed
        import weakref

        def get_ref():
            # This is similar to the construction in test_anomaly_assign_parent_cleanup:
            #
            # MyFuncBackward2 -> PyObject -> MyFuncBackward -> dict -> Foo
            #                               out ---^         WeakRef ---^
            #
            # We want to check that Foo is still properly destroyed even when MyFunc2Backward's
            # AnomalyMetadata calls printstack, which does some python object manipulation.
            #
            # You might be wondering why we still have to test_anomaly_assign_parent_cleanup,
            # since if PyObject is not destroyed here, wouldn't this test would detect that also?
            # The answer is that custom function's PyObject (THPFunction) actually only hold
            # a weak reference to the c++ node!
            class MyFunc(Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    x, = ctx.saved_tensors
                    return MyFunc2.apply(x)

            class MyFunc2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO + float("NaN")

            inp = torch.rand(1, requires_grad=True)
            out = MyFunc.apply(inp)
            ginp, = torch.autograd.grad(out, (inp,), create_graph=True)

            with warnings.catch_warnings(record=True) as w:
                with self.assertRaisesRegex(RuntimeError, "Function 'MyFunc2Backward' returned nan values in its 0th output."):
                    with detect_anomaly():
                        ginp.backward()

            class Foo(object):
                pass
            my_obj = Foo()
            meta_dict = out.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return out, ref

        t, ref = get_ref()
        self.assertIsNotNone(ref())
        del t
        self.assertIsNone(ref())

    @skipIfNoLapack
    def test_eig_no_eigenvectors(self):
        A = torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float32, requires_grad=True)
        w, v = torch.eig(A, eigenvectors=False)
        with self.assertRaisesRegex(RuntimeError, 'is not differentiable'):
            torch.autograd.backward([w, v], [torch.ones_like(w), torch.ones_like(v)])

    @skipIfNoLapack
    def test_eig_complex_eigenvalues(self):
        A = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, requires_grad=True)
        w, v = torch.eig(A, eigenvectors=True)
        with self.assertRaisesRegex(RuntimeError, 'does not support complex eigenvalues'):
            torch.autograd.backward([w, v], [torch.ones_like(w), torch.ones_like(v)])

    @skipIfNoLapack
    def test_symeig_no_eigenvectors(self):
        A = torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float32, requires_grad=True)
        w, v = torch.symeig(A, eigenvectors=False)
        with self.assertRaisesRegex(RuntimeError, 'is not differentiable'):
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

    def test_no_grad_copy_sparse(self):
        # create autograd function that saves grad pointer as class static
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad, grad

        class NonContGradFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            def backward(ctx, grad):
                # Create a sparse tensor with non-contigous indices and values
                # and return as grad.
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse.FloatTensor(ni, nv, torch.Size([10, 3]))
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return ngrad, ngrad

        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        # test case that should trigger no copy for one of a,b
        emb_matrix = MyFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # check one of them is using the computed buffer
        self.assertTrue(p_a == p_g or p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for i in range(10):
            loss.backward(retain_graph=True)

        # non-contiguous indices and value, we should trigger a copy.
        a.grad = b.grad = None
        emb_matrix = NonContGradFunc.apply(a, b)
        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        loss.backward(retain_graph=True)
        p_g = NonContGradFunc.static_grad_ptr
        p_a = a.grad._values().data_ptr()
        p_b = b.grad._values().data_ptr()
        # check a,b uses different grad buffer
        self.assertFalse(p_a == p_b)
        # Verify we cloned both grads.
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        for i in range(10):
            loss.backward(retain_graph=True)

    def test_gradcheck_single_input(self):
        def check(fast_mode):
            def f(inp):
                return inp.mul(5)

            gradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True), fast_mode=fast_mode)
            gradgradcheck(f, torch.rand(10, dtype=torch.float64, requires_grad=True), fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_sparse_input(self):
        def check(fast_mode):
            def fn(sparse):
                return torch.sparse.sum(sparse)

            gradcheck(fn, torch.rand(10, dtype=torch.double).to_sparse().requires_grad_(True), check_sparse_nnz=True,
                      check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'gradcheck expects all tensor inputs are dense'):
                gradcheck(fn, torch.rand(10, dtype=torch.double).to_sparse().requires_grad_(True), check_sparse_nnz=False,
                          check_batched_grad=False, fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_nondeterministic(self):
        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return NonDetFunc.apply(grad_out, ctx._jitter) * (1 + torch.rand_like(grad_out) * ctx._jitter), None

        def check(fast_mode):
            inp = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
            gradcheck(lambda x: NonDetFunc.apply(x, 0.0), inp, check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Backward is not reentrant'):
                gradcheck(lambda x: NonDetFunc.apply(x, 1e-6), inp, check_batched_grad=False, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Backward is not reentrant'):
                gradgradcheck(lambda x: NonDetFunc.apply(x, 1e-12), inp, check_batched_grad=False, fast_mode=fast_mode)
            gradcheck(lambda x: NonDetFunc.apply(x, 0.0), inp, nondet_tol=1e-5, check_batched_grad=False,
                      fast_mode=fast_mode)
            gradcheck(lambda x: NonDetFunc.apply(x, 1e-6), inp, nondet_tol=1e-5, check_batched_grad=False,
                      fast_mode=fast_mode)
            gradgradcheck(lambda x: NonDetFunc.apply(x, 1e-12), inp, nondet_tol=1e-5, check_batched_grad=False,
                          fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_validates_inputs(self):
        def check(fast_mode):
            # when inputs are not dense, but check_sparse_nnz is false
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'dense when check_sparse_nnz is set to False.'):
                gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=False, check_batched_grad=False,
                          fast_mode=fast_mode)
            self.assertFalse(gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=False,
                                       check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))

            # when none of the inputs require grad (always raises even if raise_exception=False)
            x = torch.rand(10, requires_grad=False)
            with self.assertRaisesRegex(ValueError, 'at least one input tensor to require gradient'):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

            # (warning) when inputs are not double precision
            x = torch.ones(1, dtype=torch.float32, requires_grad=True)
            with self.assertWarnsRegex(UserWarning, "Input #0 requires gradient and is not a double precision"):
                self.assertTrue(gradcheck(lambda x: x, (x,), atol=1e-1, fast_mode=fast_mode))

            # when layout is not mkldnn(aka has strides) and input has a dimension with stride 0. (always raises
            # even if raise_exception=False)
            x = torch.ones(1, dtype=torch.float64, requires_grad=True)
            x = x.expand((2, 2))
            with self.assertRaisesRegex(RuntimeError, 'The 0th input has a dimension with stride 0'):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_gradcheck_validates_input_mkldnn(self):
        # when mkldnn inputs, forward mode testing is not allowed
        # Update tolerances below to make sure the gradient match even in single precision floats
        # Use the warning assert to hide the float32 warning
        x = torch.ones(1).to_mkldnn().requires_grad_()
        with self.assertWarnsRegex(UserWarning, "Input #0 requires gradient and is not a double precision"):
            with self.assertRaisesRegex(ValueError, 'MKLDNN inputs are not support for forward AD gradcheck.'):
                gradcheck(lambda x: x.to_dense(), (x,), raise_exception=False, fast_mode=False, check_forward_ad=True,
                          atol=1e-1, rtol=1e-1)

        with self.assertWarnsRegex(UserWarning, "Input #0 requires gradient and is not a double precision"):
            with self.assertRaisesRegex(ValueError, 'MKLDNN inputs are not support for forward AD gradcheck.'):
                gradcheck(lambda x: x.to_dense(), (x,), raise_exception=False, fast_mode=True, check_forward_ad=True,
                          atol=1e-1, rtol=1e-1)

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_gradcheck_test_outputs(self):
        def check(fast_mode):
            # when sparse outputs (always raise even if raise_exception=False)
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(ValueError, 'Sparse output is not supported at gradcheck yet'):
                gradcheck(lambda x: x, (x,), check_sparse_nnz=True, check_batched_grad=False, raise_exception=False,
                          fast_mode=fast_mode)

            # when mkldnn outputs (always raise even if raise_exception=False)
            root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
            with self.assertRaisesRegex(ValueError, 'MKLDNN output is not supported at gradcheck yet'):
                gradcheck(lambda x: x.to_mkldnn(), (root,), check_batched_grad=False, raise_exception=False, fast_mode=fast_mode)
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_no_differentiable_outputs(self):
        def check(fast_mode):
            # When none of the outputs are differentiable, but numerical gradient is not zero
            x = torch.ones((1,), requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'Numerical gradient for function expected to be zero'):
                gradcheck(lambda x: torch.tensor([x]), x)
            self.assertFalse(gradcheck(lambda x: torch.tensor([x]), x, raise_exception=False, fast_mode=fast_mode))

            # succeed when no outputs at all
            self.assertTrue(gradcheck(lambda x: (), (x,), fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_batched_grad(self):
        def check(fast_mode):
            x = torch.rand(10, dtype=torch.double, requires_grad=True).to_sparse()
            # runtime error while compute batched grad (print big error)
            with self.assertRaisesRegex(RuntimeError, 'gradcheck or gradgradcheck failed while testing batched gradient'):
                gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=True, check_batched_grad=True, fast_mode=fast_mode)
            self.assertFalse(gradcheck(lambda x: x.to_dense(), (x,), check_sparse_nnz=True, check_batched_grad=True,
                                       raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_backward_mul_by_grad_output(self):
        # when grad_input is sparse and has incorrect sparse_dim/dense_dim
        def check(fast_mode):
            def fn(x):
                def hook(grad):
                    if grad is not None:
                        return grad.to_dense().to_sparse(1)
                    return grad
                y = x.clone()
                y.register_hook(hook)
                return y.to_dense()
            x = torch.ones((2, 2), dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'grad is sparse tensor, but has incorrect sparse_dim'):
                gradcheck(fn, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False,
                                       raise_exception=False, fast_mode=fast_mode))

            # when backward not multiplied by grad_output (non-sparse case)
            def fn2(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'backward not multiplied by grad_output'):
                gradcheck(fn2, (x,), atol=1e-1, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn2, (x,), atol=1e-1, raise_exception=False, fast_mode=fast_mode))

            # when backward not multiplied by grad_output (sparse case)
            def fn3(x):
                y = x.clone().to_dense()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(RuntimeError, 'backward not multiplied by grad_output'):
                gradcheck(fn3, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn3, (x,), atol=1e-1, check_sparse_nnz=True, check_batched_grad=False,
                                       raise_exception=False, fast_mode=fast_mode))

            # when layout of grad_input is not the same as input
            class Test(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, x):
                    return x.to_sparse()
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'grad is incorrect layout'):
                gradcheck(Test.apply, (x,), check_batched_grad=False, fast_mode=fast_mode)
            self.assertFalse(gradcheck(Test.apply, (x,), check_batched_grad=False, raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_undefined_grad(self):
        def check(fast_mode):
            # when encounter runtime error while running backward
            def fn(x):
                def hook(x):
                    if x is None:
                        raise RuntimeError("x is undefined")
                y = x.clone()
                y.register_hook(hook)
                return y
            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertWarnsRegex(UserWarning, "Backwards compatibility: New undefined gradient support checking feature"):
                with self.assertRaisesRegex(RuntimeError, 'Expected backward function to handle undefined output grads'):
                    gradcheck(fn, (x,), fast_mode=fast_mode)
                self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_jacobian_mismatch(self):
        def check(fast_mode):
            def fn(x):  # R -> R, C -> C
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))

            x_c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, 'While considering the imaginary part of complex outputs only'):
                gradcheck(fn, (x_c,), fast_mode=False)
            self.assertFalse(gradcheck(fn, (x_c,), raise_exception=False, fast_mode=False))

            def fn2(x):  # R -> C
                y = torch.complex(x, x)
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, 'While considering the imaginary part of complex outputs only'):
                gradcheck(fn2, (x,), fast_mode=False)
            self.assertFalse(gradcheck(fn2, (x,), raise_exception=False, fast_mode=False))

            def fn3(x):  # C -> R
                y = torch.real(x)
                y.register_hook(lambda x: x + 1e-2)
                return y
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn3, (x_c,), fast_mode=False)
            self.assertFalse(gradcheck(fn3, (x_c,), raise_exception=False, fast_mode=False))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_dense_and_sparse_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x * y.coalesce().to_dense()
            a = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
            b = torch.rand(2, 2, dtype=torch.double,).to_sparse().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, b), check_sparse_nnz=True, check_batched_grad=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
    def test_gradcheck_multiple_mkldnn_inputs(self):
        def check(fast_mode):
            def fn(x, y):
                return x + y.to_dense()
            a = torch.rand(10, requires_grad=True)
            b = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, b), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode))

            def fn2(x, y):
                return x.to_dense() + y.to_dense()
            c = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            self.assertTrue(gradcheck(fn, (a, c), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_output_shape_or_dtype_depend_on_values(self):
        def check(fast_mode):
            def fn(x):
                if torch.all(x >= 1):
                    return torch.cat([x, x])
                else:
                    return x
            a = torch.ones(1, dtype=torch.double, requires_grad=True)
            with self.assertRaisesRegex(AssertionError, 'return outputs with the same shape when inputs are perturbed'):
                self.assertTrue(gradcheck(fn, (a,), fast_mode=fast_mode))

            def fn2(x):
                if torch.all(x >= 1):
                    return x.to(torch.float32)
                else:
                    return x
            with self.assertRaisesRegex(AssertionError, 'return outputs with the same dtype when inputs are perturbed'):
                self.assertTrue(gradcheck(fn2, (a,), fast_mode=fast_mode))
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_complex_non_complex_outputs(self):
        def fn(x, y):
            z = torch.complex(x, y)
            return z, x + 1
        a = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        self.assertTrue(gradcheck(fn, (a, b)))

        def fn2(z):
            return z, torch.real(z)
        c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
        self.assertTrue(gradcheck(fn2, (c)))

    def test_gradcheck_get_numerical_jacobian(self):
        # get_numerical_jacobian is deprecated and no longer used internally by gradcheck
        from torch.autograd.gradcheck import get_numerical_jacobian

        def fn(inputs):
            # get_numerical_jacobian requires fn to take inputs as a tuple
            # and returns the jacobian wrt the first output
            x = inputs[0]
            y = inputs[1]
            return 2 * x + y, x + 2 * y
        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        with self.assertWarnsRegex(UserWarning, "get_numerical_jacobian was part of PyTorch's private API"):
            jacobian = get_numerical_jacobian(fn, (a, b), target=a, eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))

        with self.assertWarnsRegex(UserWarning, "get_numerical_jacobian was part of PyTorch's private API"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6)
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobian[1], 1 * torch.eye(4, dtype=torch.double))

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6, grad_out=2.0)

    def test_gradcheck_get_analytical_jacobian(self):
        from torch.autograd.gradcheck import get_analytical_jacobian

        def fn(x, y):
            return 2 * x + y, x + 2 * y

        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)

        outputs = fn(a, b)
        with self.assertWarnsRegex(UserWarning, "get_analytical_jacobian was part of PyTorch's private API"):
            jacobians, reentrant, correct_grad_sizes, correct_grad_types = get_analytical_jacobian((a, b), outputs[0])
        self.assertEqual(jacobians[0], 2 * torch.eye(4, dtype=torch.double))
        self.assertEqual(jacobians[1], 1 * torch.eye(4, dtype=torch.double))
        self.assertTrue(reentrant)

        class NonDetFunc(Function):
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            @staticmethod
            def backward(ctx, grad_out):
                return NonDetFunc.apply(grad_out, ctx._jitter) * (1 + torch.rand_like(grad_out) * ctx._jitter), None

        outputs = NonDetFunc.apply(a, 1e-6)
        with self.assertWarnsRegex(UserWarning, "get_analytical_jacobian was part of PyTorch's private API"):
            jacobians, reentrant, correct_grad_sizes, correct_grad_types = get_analytical_jacobian((a,), outputs)
        self.assertFalse(reentrant)

        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            jacobians, _, _, _ = get_analytical_jacobian((a,), outputs, grad_out=2.0)

    def test_gradcheck_custom_error(self):
        from torch.autograd.gradcheck import GradcheckError

        def check(fast_mode):
            def fn(x):
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y
            x = torch.ones(2, 2, requires_grad=True)
            with self.assertRaisesRegex(GradcheckError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, 'Jacobian mismatch for output 0 with respect to input 0'):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            self.assertFalse(gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode))

            def fn2(x):
                raise RuntimeError("Not a GradcheckError!")
            # Checks that when raise_exception=False, non-GradcheckErrors are not caught by gradcheck
            with self.assertRaisesRegex(RuntimeError, "Not a GradcheckError!"):
                gradcheck(fn2, (x,), fast_mode=fast_mode, raise_exception=False)

        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_forward_ad(self):
        def fn(x, y):
            return x + y, y

        def bad_fn(x, y):
            # Hacky way to check if we're currently inside a forward ad level
            is_running_forward_ad = fwAD._current_level >= 0

            if is_running_forward_ad:
                y_p, y_d = fwAD.unpack_dual(y)
                y = fwAD.make_dual(y_p, y_d * 1.1)

            return x + y, y

        err_msg = "Jacobian computed with forward mode mismatch for output 0 with respect to input 1"

        for fast_mode in [True, False]:
            # Test for all inputs and outputs being real
            x = torch.rand(2, dtype=torch.double, requires_grad=True)
            y = torch.rand(2, dtype=torch.double, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            def basic_mul(x):
                return torch.view_as_real(x * 1j)
            gradcheck(basic_mul, x, check_forward_ad=True, fast_mode=fast_mode)

            # Test for one input and one output being complex
            x = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            # Test for all inputs and outputs being complex
            y = torch.rand(2, dtype=torch.cdouble, requires_grad=True)

            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

    def test_version_counter(self):
        x = torch.randn(1, 2)

        # In-place op bumps version
        x_saved_version = x._version
        x.add_(1).add_(1)
        self.assertTrue(x._version > x_saved_version)

        # Differentiable view shares version counter
        xz = x[:]
        self.assertTrue(x._version == xz._version)
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

        # `x.data = y` preserves version counter of `x`
        x_saved_version = x._version
        x.data = torch.randn(2, 3)
        self.assertTrue(x._version == x_saved_version)
        x.add_(1)
        self.assertTrue(x._version > x_saved_version)
        # Make sure `x` is still using the same version counter it shares with `xz`
        self.assertTrue(x._version == xz._version)

        # In-place op on `xz` also updates version of `x`,
        # because they share the version counter
        xz.add_(1)
        self.assertTrue(x._version == xz._version)

    def test_set_data_tensorimpl_type(self):
        # Dense tensor has impl of type `TensorImpl`, while sparse tensor has impl
        # of type `SparseTensorImpl`.
        x = torch.randn(1, 2)
        x_s = torch.sparse_coo_tensor(torch.zeros([1, 1]), torch.ones([1]))
        with self.assertRaisesRegex(RuntimeError, 'incompatible tensor type'):
            x.data = x_s

    def test_set_data_preserve_pyobj(self):
        a = torch.randn(1, 2)
        b = torch.randn(1, 2)
        b_id_saved = id(b)
        b.data = a
        self.assertTrue(b_id_saved == id(b))

    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    def test_thread_shutdown(self):
        code = """import torch
from torch.autograd import Function
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad

for shape in [(1,), ()]:
    v = torch.ones(shape, requires_grad=True)
    MyFunction.apply(v).backward()
"""
        s = TestCase.runWithPytorchAPIUsageStderr(code)
        self.assertRegex(s, "PYTORCH_API_USAGE torch.autograd.thread_shutdown")

    @unittest.skipIf(IS_MACOS, "Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941")
    def test_deep_reentrant(self):

        class DeepReentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    DeepReentrant.apply(ctx.x).sum().backward()
                return x

        # Test stack overflow escape mechanism
        v = torch.tensor(2000.0, requires_grad=True)
        # This will cause stack overflow if reentrant calls are handled
        # in the same thread recursively
        DeepReentrant.apply(v).sum().backward()

        # Test stack overflow escape mechanism multiple times
        # to ensure reusing workers in the pool works fine
        v2 = torch.tensor(200.0, requires_grad=True)
        DeepReentrant.apply(v2).sum().backward()

    def test_reentrant_priority(self):
        order = []

        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                order.append("MyFunction")
                return x

        class Reentrant(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            def backward(ctx, x):
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        v = a * b
        v.backward()
        # The tasks for the Reentrant and MyFunction backward() will be added
        # to the queue in the autograd engine at the same time. The backward
        # for Reentrant will be executed first, which will then add other
        # backward tasks to the queue. We want to ensure all the reentrant tasks
        # are prioritized over the MyFunction backward task regardless of their
        # sequence numbers
        self.assertEqual(len(order), 11)
        self.assertEqual(order.count("Reentrant"), 10)
        self.assertEqual(order[-1], "MyFunction")


    @slowTest
    def test_checkpointing(self):
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # small proxy network for some complex reasoning we want to do per input
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp)
        )

        feat_combined = []
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            feat_r = checkpoint(module, data_r)
            feat_combined.append(feat_r)

        # compute mean as a proxy for some joint reasoning
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()

    def test_checkpoint_valid_reset_on_error(self):
        a = torch.randn(2, 2, requires_grad=True)

        with self.assertRaisesRegex(Exception, "Checkpointing is not compatible with .grad()"):
            b = checkpoint(torch.exp, a).sum()
            torch.autograd.grad(b, (a,))

        c = checkpoint(torch.exp, a).sum()
        c.backward()

    def _test_reentrant_with_callbacks(self, install_callbacks_in_depths):
        counter = {}
        counter["inner"] = 0
        counter["outer"] = 0

        def inc_inner_counter():
            counter["inner"] += 1

        def inc_outer_counter():
            counter["outer"] += 1

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 1 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_inner_counter)

                return input

        class MyReentrantFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 0 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_outer_counter)
                # Reentrant backward call.
                tmp_inp = input.detach().requires_grad_()
                with torch.enable_grad():
                    tmp_out = (MyFunc.apply(tmp_inp)).sum()
                tmp_out.backward()
                return input

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = MyReentrantFunc.apply(t1)
        t3 = t2.sum()
        torch.autograd.backward([t3])

        return counter

    def test_reentrant_with_callbacks_depth_0(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([0])
        self.assertEqual(1, ret["outer"])
        self.assertEqual(0, ret["inner"])

    def test_reentrant_with_callbacks_depth_1(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([1])
        self.assertEqual(0, ret["outer"])
        self.assertEqual(1, ret["inner"])

    def test_reentrant_with_callbacks_both_depths(self):
        # Verify callback is called twice.
        ret = self._test_reentrant_with_callbacks([0, 1])
        self.assertEqual(1, ret["outer"])
        self.assertEqual(1, ret["inner"])

    def test_reentrant_with_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

        def add_gradient_penalty_to_grad(grad):
            handle.remove()
            old_param_grad = grad
            param.grad = None
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                new_param = param.detach().requires_grad_()
                out = ((g * 2) + new_param).sum()
                out.backward()
            res = g.grad + grad
            param.grad = old_param_grad
            return res

        handle = param.register_hook(add_gradient_penalty_to_grad)
        # Forward pass
        tmp = (param * param)
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()

    def test_reentrant_with_non_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

        def manual_increase_gradient(grad):
            handle.remove()
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                out = ((g * 2) + 5).sum()
                out.backward()
            res = g.grad + grad
            return res

        # Forward pass
        tmp = (param * param)
        handle = tmp.register_hook(manual_increase_gradient)
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()
        self.assertEqual(param.grad, 6 * param)

    def test_grad_fn_attr_bindings(self):
        # Check that the getter of each type returns what we want
        # See `gen_autograd_functions.py` for how the getters are generated
        #
        # This test is only meant to check if the codegen'd bindings work
        # Please help update this test if you update the names of any the fields we check!
        #
        a = torch.ones(1, requires_grad=True)
        b = torch.ones(1, requires_grad=True)
        out = torch.stack([a, b], dim=0)
        self.assertEqual(out.grad_fn._saved_tensors, (a, b))              # TensorList -> Tuple[Tensor]
        self.assertIsInstance(out.grad_fn._saved_tensors[0], torch.Tensor)
        self.assertEqual(out.grad_fn._saved_dim, 0)                       # int64_t -> int
        self.assertIsInstance(out.grad_fn._saved_dim, int)

        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out.grad_fn._saved_tensors
        self.assertEqual(out.grad_fn._saved_dim, 0)

        a = torch.ones(2, 2, requires_grad=True)
        indices = torch.tensor([0, 1])
        out = a[:, indices]
        self.assertEqual(out.grad_fn._saved_indices, (None, indices))     # c10::List<c10::optional<Tensor>> -> Tuple[Tensor?]
        self.assertIsInstance(out.grad_fn._saved_indices[1], torch.Tensor)
        self.assertEqual(out.grad_fn._saved_self_sizes, a.shape)          # IntArrayRef -> Tuple[int]
        self.assertIsInstance(out.grad_fn._saved_self_sizes[0], int)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.nn.functional.interpolate(a, 4, mode="linear")
        self.assertEqual(out.grad_fn._saved_output_size, (4,))            # c10::optional<IntArrayRef> -> int[]?
        self.assertIsInstance(out.grad_fn._saved_output_size[0], int)
        self.assertEqual(out.grad_fn._saved_align_corners, False)         # bool -> bool
        self.assertIsInstance(out.grad_fn._saved_align_corners, bool)
        self.assertIsNone(out.grad_fn._saved_scale_factors)               # c10::optional<ArrayRef<double>> -> float[]?

        out = torch.nn.functional.interpolate(a, scale_factor=0.5, mode="linear")
        self.assertIsNone(out.grad_fn._saved_output_size)
        self.assertEqual(out.grad_fn._saved_scale_factors, (0.5,))
        self.assertIsInstance(out.grad_fn._saved_scale_factors[0], float)

        a = torch.ones(2, 2, requires_grad=True)
        out = torch.pdist(a, p=1)
        self.assertEqual(out.grad_fn._saved_p, 1.)                        # double -> float
        self.assertIsInstance(out.grad_fn._saved_p, float)

        a = torch.ones(1, 1, 2, requires_grad=True)
        out = torch.logit(a, 1.)
        self.assertEqual(out.grad_fn._saved_eps, 1.)                      # c10:optional<double> -> float?
        self.assertIsInstance(out.grad_fn._saved_eps, float)
        out = torch.logit(a)
        self.assertIsNone(out.grad_fn._saved_eps)

        if torch._C.has_lapack:
            a = torch.ones(1, 1, requires_grad=True)
            q, r = torch.linalg.qr(a, mode="reduced")
            self.assertEqual(q.grad_fn._saved_mode, "reduced")                # std::string -> str

        a = torch.tensor([1.], requires_grad=True)
        out = torch.div(a, 2., rounding_mode="trunc")
        self.assertEqual(out.grad_fn._saved_rounding_mode, "trunc")       # c10::optional<std::string> -> str?
        out = torch.div(a, 2., rounding_mode=None)
        self.assertIsNone(out.grad_fn._saved_rounding_mode)               # c10::optional<std::string> -> str?

        x = torch.zeros(5, requires_grad=True)
        out = torch.threshold(x, threshold=(1 + 0j), value=(1 + 0j))
        self.assertIsInstance(out.grad_fn._saved_threshold, complex)      # Scalar(complex double) -> complex
        cfloat = torch.tensor(1 + 0j, dtype=torch.complex64)
        out = torch.threshold(x, threshold=cfloat, value=(1 + 0j))
        self.assertIsInstance(out.grad_fn._saved_threshold, complex)      # Scalar(complex float) -> complex
        out = torch.threshold(x, threshold=1., value=1.)
        self.assertIsInstance(out.grad_fn._saved_threshold, float)        # Scalar(floating point) -> float
        out = torch.threshold(x, threshold=1, value=1)
        self.assertIsInstance(out.grad_fn._saved_threshold, int)          # Scalar(integral) -> int
        out = torch.threshold(x, threshold=False, value=False)
        self.assertIsInstance(out.grad_fn._saved_threshold, bool)         # Scalar(bool) -> bool

        a = torch.ones(2, 2, requires_grad=True)
        out = a.as_strided((3,), (1,), 1)
        self.assertEqual(out.grad_fn._saved_storage_offset, 1)            # c10:optional<int64_t> -> int?
        self.assertIsInstance(out.grad_fn._saved_storage_offset, int)
        out = a.as_strided((3,), (1,))
        self.assertIsNone(out.grad_fn._saved_storage_offset)

        a = torch.ones(2, requires_grad=True)
        out = torch.tanh(a)
        self.assertEqual(out, out.grad_fn._saved_result)                  # saved variable when output

        a = torch.randn(3, 5, requires_grad=True)
        b = torch.tensor([1, 0, 4])
        loss = nn.NLLLoss()
        out = loss(a, b)
        self.assertIsNone(out.grad_fn._saved_weight)
        loss = nn.NLLLoss(weight=torch.ones((5,)))
        out = loss(a, b)
        self.assertEqual(out.grad_fn._saved_weight, torch.ones((5,)))     # c10:optional<Tensor> -> Tensor?

        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out.grad_fn._saved_weight

    def test_autograd_views_codegen(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks the behavior of two codegen functions (view_as and unbind)
        # with respect to view tracking and inplace operation on the output.

        def run_test(grad_mode, requires_grad, is_view, should_raise_tuple):
            def maybe_check_raise(fn, should_raise):
                self.assertTrue(should_raise is None or isinstance(should_raise, str))
                if should_raise is not None:
                    with self.assertRaisesRegex(RuntimeError, should_raise):
                        fn()
                else:
                    fn()

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.view_as(inp)
            # Are they differentiable views?
            self.assertTrue(out._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out.add_(1), should_raise_tuple[0])

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.unbind()
            # Are they differentiable views?
            self.assertTrue(out[0]._is_view() == is_view)
            self.assertTrue(out[1]._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out[0].add_(1), should_raise_tuple[1])
            maybe_check_raise(lambda: out[1].add_(1), should_raise_tuple[2])

        # should_raise contains None if it should not raise
        # should_raise contains a string of the error if it should raise
        # The 3 elements are for view_as, first output of unbind and second output of unbind
        run_test(grad_mode=True, requires_grad=False, is_view=True,
                 should_raise_tuple=(None, None, None))
        inp_change_err = "Output {} of UnbindBackward is a view and is being modified inplace."
        run_test(grad_mode=True, requires_grad=True, is_view=True,
                 should_raise_tuple=(None, inp_change_err.format("0"), inp_change_err.format("1")))
        leaf_grad_err = "A view was created in no_grad mode and is being modified inplace"
        run_test(grad_mode=False, requires_grad=True, is_view=True,
                 should_raise_tuple=(leaf_grad_err, leaf_grad_err, leaf_grad_err))
        run_test(grad_mode=False, requires_grad=False, is_view=True,
                 should_raise_tuple=(None, None, None))

    def test_inplace_not_requires_grad(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return grad

        # Original Tensor does not require grad
        a = torch.rand(1, 2)

        # Tensor being written does require grad
        b = torch.rand(1, requires_grad=True)

        # Take an invalid view on 'a' that should raise an error (warns during deprecation)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(RuntimeError, "This view was created inside a custom Function"):
            view_a += b

        # Extra test for copy_ that is a manual implementation and could be easily
        # forgotten when the codegen is updated (warns during deprecation)
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(RuntimeError, "This view was created inside a custom Function"):
            view_a.copy_(b)

        # Functions that should throw must properly throw
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = a.unbind()[0]
        with self.assertRaisesRegex(RuntimeError, "This view is the output of a function that returns "
                                                  "multiple views."):
            view_a.copy_(b)

        # Sanity check that views that should work still work
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        a.select(1, 0).copy_(b)

    def _do_test_autograd_simple_views_python(self, dtype):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks the autograd.Function behavior when we return one or multiple outputs
        # while one of these is an input, a view of an input or of a temporary tensor.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]
        # This indicator is used to check if the argument `ga` contains non-zero values
        ga_nz = [False]

        class IdOneOutput(Function):
            @staticmethod
            def forward(ctx, a, b, make_view):
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a

            @staticmethod
            def backward(ctx, ga):
                bw_called[0] += 1
                return ga, None, None

        class IdTwoOutput(Function):
            @staticmethod
            def forward(ctx, a, b, make_view):
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                if ga.eq(0).all():
                    ga_nz[0] = False
                else:
                    ga_nz[0] = True
                return ga + gab, gab, None

        class ViewOfTemp(Function):
            @staticmethod
            def forward(ctx, a, make_view):
                ctx.save_for_backward(a)
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                b = a.clone()
                return b.select(0, 0)

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                a, = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, 0).copy_(grad)
                return res, None

        fn_id_to_inplace_view_err_msg = {
            "one_output": ("Output 0 of IdOneOutputBackward is a view and is being "
                           "modified inplace. This view was created inside a custom Function"),
            "two_output": ("Output 0 of IdTwoOutputBackward is a view and is being modified inplace."
                           " This view is the output of a function that returns multiple views."),
            "view_of_temp": ("Output 0 of ViewOfTempBackward is a view and is being "
                             "modified inplace. This view was created inside a custom Function")
        }

        for fn_id in ["one_output", "two_output", "view_of_temp"]:
            for inplace in [True, False]:
                for make_view in [True, False]:
                    # Used for special casing the tests below
                    output_is_a_view = (make_view or fn_id == "view_of_temp")

                    def fn(a, b):
                        # never modify a, b inplace for gracheck
                        a = a.clone()
                        b = b.clone()
                        if fn_id == "two_output":
                            tmp1, tmp2 = IdTwoOutput.apply(a, b, make_view)
                            if inplace:
                                tmp1 += 3
                                tmp2 += 3
                            else:
                                tmp1 = tmp1 + 3
                                tmp2 = tmp2 + 3
                            tmp = tmp1 * tmp2
                        else:
                            if fn_id == "one_output":
                                tmp = IdOneOutput.apply(a, b, make_view)
                            else:
                                tmp = ViewOfTemp.apply(a + b, make_view)
                            if inplace:
                                tmp += 3
                            else:
                                tmp = tmp + 3

                        return tmp.sum()

                    a = torch.ones(2, dtype=dtype, requires_grad=True)
                    b = torch.ones(2, dtype=dtype, requires_grad=True)

                    err_msg = fn_id_to_inplace_view_err_msg[fn_id]

                    if not inplace or not output_is_a_view:
                        gradcheck(fn, (a, b), check_batched_grad=False)

                    # Was the custom backward called properly
                    bw_called[0] = 0
                    ga_nz[0] = True  # For the case where the backward is called

                    if inplace and output_is_a_view:
                        with self.assertRaisesRegex(RuntimeError, err_msg):
                            fn(a, b)
                    else:
                        fn(a, b).backward()

                    expected_called = 1
                    expected_ga_nz = True

                    if output_is_a_view and inplace:
                        expected_called = 0

                    self.assertTrue(bw_called[0] == expected_called)
                    self.assertTrue(ga_nz[0] == expected_ga_nz)

    def test_autograd_simple_views_python(self):
        self._do_test_autograd_simple_views_python(torch.double)
        self._do_test_autograd_simple_views_python(torch.cdouble)

    def test_autograd_complex_views_python(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks that multiples views in the forward are properly traced and how they
        # behave with respect to inplace operations.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]

        class ComplexView(Function):
            @staticmethod
            def forward(ctx, a, idx):
                res = a.narrow(0, idx, 1)
                res = a.select(0, idx)
                ctx.save_for_backward(a)
                ctx.idx = idx
                return res

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                a, = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, ctx.idx).copy_(grad)
                return res, None

        a = torch.ones(2, requires_grad=True)
        idx = 1

        bw_called[0] = 0
        out = ComplexView.apply(a.clone(), idx)
        out.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        out = ComplexView.apply(a.clone(), idx)
        with self.assertRaisesRegex(RuntimeError,
                                    "Output 0 of ComplexViewBackward is a view and is being modified inplace"):
            out += 1

    def test_autograd_inplace_views_python(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks custom autograd.Function that perform inplace operations

        bw_called = [0]

        # I) Single output
        class MyAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                return grad, grad


        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        # No extra inplace
        c = MyAdder.apply(a.clone(), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c = MyAdder.apply(a.clone(), b)
        c += 2
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        bw_called[0] = 0
        c = MyAdder.apply(a.clone().view_as(a), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # Should not give non-inputs to mark_dirty
        class MyAdderBad(Function):
            @staticmethod
            def forward(ctx, a, b):
                c = 3 * a
                c.add_(b)
                ctx.mark_dirty(c)
                return c

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                grad = 3 * grad
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            MyAdderBad.apply(a.clone(), b)
        self.assertEqual(len(w), 1)

        # II) Multiple outputs
        class MyBadAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + gab

        # No extra inplace
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        c += 2
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        inplace_on_view_err = "your Function modifies inplace an input that is a view of another Tensor"
        with self.assertRaisesRegex(RuntimeError, inplace_on_view_err):
            c, d = MyBadAdder.apply(a.clone().view_as(a), b)

        # III) Inplace + other op
        class MyOutPlaceAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a.clone(), a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + 2 * gab

        # We don't reuse the input
        def fn(a, b):
            orig_a = a.clone().view_as(a)
            c, d = MyOutPlaceAdder.apply(orig_a, b)
            return (c * d).sum()

        bad_mark_dirty_err = "Some elements marked as dirty during the forward method were not returned as output."
        with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
            fn(a, b)

    def test_named_tensor_for_complex_views(self):
        names = ["batch", "height", "width", "complex"]
        z = torch.ones((5, 12, 14, 2), requires_grad=True)
        z_named = z.refine_names(*names)
        z_complex = torch.view_as_complex(z_named.rename(None)).refine_names(*names[:-1])
        z_complex.sum().backward()
        self.assertEqual(z.grad, torch.view_as_real(torch.ones_like(z_complex).rename(None)))

    def test_custom_function_return_view_in_nograd(self):
        class Alias(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:]

            @staticmethod
            def backward(ctx, gx):
                return gx

        inp = torch.rand(2, requires_grad=True)

        with torch.no_grad():
            output = Alias.apply(inp)

        with torch.no_grad():
            expected_output = inp[:]

        # Calling the custom function should operate as if we called an equivalent op
        self.assertEqual(output.requires_grad, expected_output.requires_grad)

        # Check that in-place modification on view throws
        leaf_grad_err = "A view was created in no_grad mode and is being modified inplace"
        with self.assertRaisesRegex(RuntimeError, leaf_grad_err):
            output.zero_()

    def test_grad_mode_restored_reentrant(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, go):
                original = torch._C.is_grad_enabled()
                with torch.enable_grad():
                    self.assertTrue(torch._C.is_grad_enabled())
                    foo = torch.rand(go.size(), requires_grad=True)
                    grad, = torch.autograd.grad(
                        foo ** 3, foo, grad_outputs=go
                    )
                    self.assertTrue(torch._C.is_grad_enabled())
                self.assertTrue(torch._C.is_grad_enabled() == original)
                return grad

        inp = torch.rand(3, requires_grad=True)

        # Case where original==False
        MyFunction.apply(inp).sum().backward()
        # Case where original==True
        MyFunction.apply(inp).sum().backward(create_graph=True)

    def test_power_function(self):
        a = torch.tensor([0., 0., 0.])
        b = torch.tensor([-1., 0., 1.], requires_grad=True)
        c = torch.sum(a**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0., 0.]))

        s = 0
        b = torch.tensor([-1., 0., 1.], requires_grad=True)
        c = torch.sum(s**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0., 0.]))

    def test_nansum_with_nans(self):
        a = torch.randn(2, 2, 2, 2, dtype=torch.double)
        with torch.no_grad():
            a[a < 0.2] = float('nan')
        a.requires_grad = True

        # No args
        gradcheck(lambda x: x.nansum(), a)
        gradgradcheck(lambda x: x.nansum(), a)

        # Single dim
        gradcheck(lambda x: x.nansum((0)), a)
        gradgradcheck(lambda x: x.nansum((0)), a)

        # Multi dim
        gradcheck(lambda x: x.nansum((0, 2)), a)
        gradgradcheck(lambda x: x.nansum((0, 2)), a)

        gradcheck(lambda x: x.nansum((0, -1)), a)
        gradgradcheck(lambda x: x.nansum((0, -1)), a)

        # With keep-dim
        gradcheck(lambda x: x.nansum((0, -1), True), a)
        gradgradcheck(lambda x: x.nansum((0, -1), True), a)

    def test_nansum_dtype(self):
        inp = torch.randn(2, 2, 2, 2)
        with torch.no_grad():
            inp[inp < 0.2] = float('nan')

        def test(inp, inp_dtype, out_dtype):
            with torch.no_grad():
                a = inp.to(inp_dtype)
            a.requires_grad = True
            b = torch.sum(a, dtype=out_dtype)
            b.backward()
            self.assertEqual(a.dtype, a.grad.dtype)

        test(inp, torch.float, torch.double)
        test(inp, torch.double, torch.float)

    def test_nan_to_num(self):
        a = torch.randn(3, 3, 3, 3, dtype=torch.double)
        with torch.no_grad():
            a[torch.rand_like(a) < 0.2] = float('nan')
            a[torch.rand_like(a) < 0.2] = float('inf')
            a[torch.rand_like(a) < 0.2] = -float('inf')

        a.requires_grad = True

        gradcheck(lambda x: x.nan_to_num(), a)
        gradgradcheck(lambda x: x.nan_to_num(), a)

        gradcheck(lambda x: x.nan_to_num(nan=1.2), a)
        gradgradcheck(lambda x: x.nan_to_num(nan=1.2), a)

        gradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0), a)

        gradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0, neginf=-2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0, neginf=-2.0), a)

        gradcheck(lambda x: x.nan_to_num(posinf=2.0, neginf=-2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(posinf=2.0, neginf=-2.0), a)

        gradcheck(lambda x: x.nan_to_num(neginf=-2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(neginf=-2.0), a)

    def test_custom_function_error(self):
        class BadFw(Function):
            @staticmethod
            def backward(ctx, foo):
                return foo

        class BadBw(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        inp = torch.rand(1, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "must implement the forward"):
            BadFw.apply(inp)

        with self.assertRaisesRegex(RuntimeError, "must implement the backward"):
            BadBw.apply(inp).sum().backward()

    def test_custom_function_local_inplace(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, inplace):
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                return grad, None

        base = torch.rand(10, requires_grad=True)

        foo = MyFn.apply(base, False)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

        foo = MyFn.apply(base, True)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

    def test_integer_outputs(self):
        inp = torch.rand(4, requires_grad=True)

        out = inp.argmax()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argmin()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argsort()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.rand((), requires_grad=True)

        out = torch.searchsorted(inp, val)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        bins = torch.linspace(0, 1.0, steps=100, requires_grad=True)
        vals = torch.rand(5, 5, requires_grad=True)
        out = torch.bucketize(vals, bins)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.empty(5).requires_grad_()
        out = val.count_nonzero()
        self.assertFalse(out.requires_grad)

        def assert_only_first_requires_grad(res):
            if not isinstance(res, tuple):
                res = (res,)
            self.assertTrue(res[0].requires_grad)
            for out in res[1:]:
                if out is not None:
                    self.assertFalse(out.requires_grad)

        for sort in [True, False]:
            for return_inverse in [True, False]:
                for return_counts in [True, False]:
                    res = torch.unique(inp, sorted=sort, return_inverse=return_inverse,
                                       return_counts=return_counts)
                    assert_only_first_requires_grad(res)

                    res = torch.unique(inp, sorted=sort, return_inverse=return_inverse,
                                       return_counts=return_counts, dim=0)
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(inp, return_inverse=return_inverse,
                                                   return_counts=return_counts)
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(inp, return_inverse=return_inverse,
                                                   return_counts=return_counts, dim=0)
                    assert_only_first_requires_grad(res)

                    # Here we test the internal functions to make sure all of them are
                    # covered on top of the public API
                    res = torch._unique(inp, sorted=sort, return_inverse=return_inverse)
                    assert_only_first_requires_grad(res)

                    # This looks public but is actually manually deleted from the
                    # torch namespace in torch/functional.py
                    res = torch._VF.unique_dim(inp, dim=0, sorted=sort, return_inverse=return_inverse,
                                               return_counts=return_counts)
                    assert_only_first_requires_grad(res)

                    # We don't test `unique_dim_consecutive` here.
                    # It looks public but the python binding is actually manually disabled in
                    # tools/autograd/gen_python_functions.py

                    res = torch._unique2(inp, sorted=sort, return_inverse=return_inverse,
                                         return_counts=return_counts)
                    assert_only_first_requires_grad(res)


def index_perm_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
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
                                 input_variables, run_gradgradcheck=True, check_batched_grad=True):
    test_case.assertTrue(gradcheck(apply_method, input_variables, eps=1e-6, atol=PRECISION,
                                   check_batched_grad=check_batched_grad))
    if name in EXCLUDE_GRADGRADCHECK or test_name in EXCLUDE_GRADGRADCHECK_BY_TEST_NAME:
        return
    gradgradcheck_precision_override = gradgradcheck_method_precision_override(test_name)
    if gradgradcheck_precision_override is not None:
        atol = gradgradcheck_precision_override['atol']
        rtol = gradgradcheck_precision_override['rtol']
        test_case.assertTrue(gradgradcheck(apply_method, input_variables, None, atol=atol, rtol=rtol,
                                           gen_non_contig_grad_outputs=True,
                                           check_batched_grad=check_batched_grad))
    else:
        test_case.assertTrue(gradgradcheck(apply_method, input_variables,
                                           gen_non_contig_grad_outputs=True,
                                           check_batched_grad=check_batched_grad))


def run_functional_checks(test_case, test_name, name, apply_fn, run_grad_checks,
                          f_args_variable, f_args_tensor):
    output_variable = apply_fn(*f_args_variable)

    if run_grad_checks:
        run_grad_and_gradgrad_checks(test_case, name, test_name, apply_fn,
                                     output_variable, f_args_variable)

    self_variable = f_args_variable[0]
    if isinstance(output_variable, torch.Tensor) and output_variable.requires_grad and self_variable is not None:
        output_variable.backward(randn_like(output_variable))
        test_case.assertEqualTypeString(self_variable, self_variable.grad)
        test_case.assertEqual(self_variable.size(), self_variable.grad.size())

# this list corresponds to ops which have separate tests defined for complex dtypes in
# common_methods_invocations.py
# test for these ops with 'complex' in variant should only run for complex and
# the tests for these ops which do not have 'complex' in variant should not run for complex
# and only run for floating point

separate_complex_tests = ['div', '__rdiv__', 'sub']

# allow list for complex
complex_list = ['t', 'view', 'reshape', 'reshape_as', 'view_as', 'roll', 'clone',
                'expand', 'rot90', 'transpose',
                'permute', 'squeeze', 'unsqueeze', 'resize', 'resize_as', 'tril', 'triu',
                'chunk', 'split', 'split_with_sizes', 'zero_',
                '__radd__', 'mul', '__rmul__', 'diagonal', 'fill_', 'sub', 'narrow',
                'swapaxes', 'swapdims', 'tensor_split'] + separate_complex_tests

# deny list for batched grad computation
EXCLUDE_BATCHED_GRAD_TESTS = set([
    'test_to_sparse',
])

def add_test(
        name,
        self_size,
        args,
        variant_name='',
        check_ad=(),  # only used in test_jit
        dim_args_idx=(),
        skipTestIf=(),
        output_process_fn=lambda x: x,
        kwargs=None):
    kwargs = kwargs if kwargs else {}
    basic_test_name = 'test_' + name
    if variant_name != '':
        basic_test_name += '_' + variant_name

    if name in separate_complex_tests and 'complex' in variant_name:
        run_only_complex = True
    else:
        run_only_complex = False

    for dtype in [torch.double, torch.cdouble]:
        for dim_perm in product([-1, 1], repeat=len(dim_args_idx)):
            test_name = basic_test_name
            new_args = [arg * dim_perm[dim_args_idx.index(i)] if i in dim_args_idx else arg for i, arg in enumerate(args)]
            test_name = basic_test_name + ''.join('_neg' + str(i) for i, idx in enumerate(dim_perm) if idx < 0)

            if dtype.is_complex:
                # TODO: remove this. this is temporary while we ramp up the complex support.
                if name in complex_list:
                    if name in separate_complex_tests and 'complex' not in variant_name:
                        continue
                    if not run_only_complex:
                        test_name = test_name + '_complex'
                else:
                    continue
            elif run_only_complex:
                continue

            new_args = tuple(new_args)

            # for-loop bodies don't define scopes, so we have to save the variables
            # we want to close over in some way
            def do_test(self, device, dtype=dtype, name=name, self_size=self_size, args=new_args, test_name=test_name,
                        output_process_fn=output_process_fn):
                def check(name):
                    is_magic_method = name[:2] == '__' and name[-2:] == '__'
                    is_inplace = name[-1] == "_" and not is_magic_method
                    self_variable = create_input((self_size,), dtype=dtype, device=device)[0][0]
                    # FixMe: run grad checks on inplace self
                    if is_inplace:
                        self_variable.requires_grad = False
                    # need to record this because methods can change the size (e.g. unsqueeze)
                    args_variable, kwargs_variable = create_input(args, requires_grad=not is_inplace,
                                                                  call_kwargs=kwargs, dtype=dtype, device=device)
                    self_tensor = deepcopy(self_variable)
                    args_tensor = deepcopy(unpack_variables(args_variable))
                    if not exclude_tensor_method(name, test_name):
                        output_variable = getattr(self_variable, name)(*args_variable, **kwargs_variable)
                        output_tensor = getattr(self_tensor, name)(*args_tensor, **kwargs_variable)
                        if not isinstance(output_tensor, torch.Tensor) and not isinstance(output_tensor, tuple):
                            if dtype.is_complex:
                                output_tensor = torch.tensor((output_tensor, ), dtype=torch.cfloat, device=device)
                            else:
                                output_tensor = torch.tensor((output_tensor, ), dtype=torch.float, device=device)
                        self.assertEqual(unpack_variables(output_variable), output_tensor)
                        # TODO: check that both have changed after adding all inplace ops

                        def fn(*inputs):
                            output = getattr(inputs[0], name)(*inputs[1:], **kwargs)
                            return output_process_fn(output)

                        if not is_inplace and name not in EXCLUDE_GRADCHECK:
                            check_batched_grad = test_name not in EXCLUDE_BATCHED_GRAD_TESTS
                            run_grad_and_gradgrad_checks(self, name, test_name, fn,
                                                         output_variable, (self_variable,) + args_variable,
                                                         check_batched_grad=check_batched_grad)

                    # functional interface tests
                    torch_fn = getattr_qualified(torch, name)
                    if torch_fn is not None and name not in EXCLUDE_FUNCTIONAL:
                        def fn(*inputs):
                            output = torch_fn(*inputs, **kwargs)
                            return output_process_fn(output)

                        f_args_variable = (self_variable,) + args_variable
                        f_args_tensor = (self_tensor,) + args_tensor
                        # could run the gradchecks again, but skip since we did it for the methods above.
                        run_gradcheck = exclude_tensor_method(name, test_name) and not is_inplace and name not in EXCLUDE_GRADCHECK
                        run_functional_checks(self, test_name, name, fn,
                                              run_gradcheck, f_args_variable, f_args_tensor)

                    # check for correct type of input and input.grad
                    if not is_inplace:
                        self_variable = create_input((self_size,), requires_grad=True, dtype=dtype)[0][0]
                        args_variable, kwargs_variable = create_input(args, requires_grad=False, call_kwargs=kwargs, dtype=dtype)
                        if hasattr(self_variable, name):
                            attribute_result = getattr(self_variable, name)
                            if callable(attribute_result):
                                output_variable = attribute_result(*args_variable, **kwargs_variable)
                            else:
                                self.assertTrue(len(args_variable) == 0)
                                self.assertTrue(len(kwargs_variable) == 0)
                                output_variable = attribute_result
                        else:
                            self_and_args_variable = (self_variable,) + args_variable
                            output_variable = torch_fn(*self_and_args_variable, **kwargs_variable)
                        if isinstance(output_variable, torch.autograd.Variable):
                            if output_variable.is_sparse:
                                rand = randn_like(output_variable.to_dense()).to_sparse()
                            else:
                                rand = randn_like(output_variable)
                            output_variable.backward(rand)
                            self.assertTrue(type(self_variable) == type(self_variable.grad))
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
                                    with torch.no_grad():
                                        inp_i.grad.zero_()
                                if i.grad is not None:
                                    with torch.no_grad():
                                        i.grad.zero_()
                            for i_o, o in zip(inplace_output_variable, output_variable):
                                if dtype.is_complex:
                                    grad = randn_like(i_o).to(torch.cdouble)
                                else:
                                    grad = randn_like(i_o).double()
                                i_o.backward(grad)
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

            setattr(TestAutogradDeviceType, test_name, do_test)

class TestAutogradComplex(TestCase):
    def test_view_func_for_complex_views(self):
        # case 1: both parent and child have view_func
        x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
        y = x.detach().requires_grad_(True)

        x0 = x.clone()
        x1 = torch.view_as_complex(x0)
        x2 = torch.view_as_real(x1)
        x2.mul_(2)
        x2.sum().backward()

        y0 = y.clone()
        y0.mul_(2)
        y0.sum().backward()

        self.assertEqual(x.grad, y.grad)

        # case 2: parent has view_func but child does not
        x = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
        y = x.detach().requires_grad_(True)

        def fn(a):
            b = a.clone()
            b1 = torch.view_as_complex(b)
            b2 = b1.reshape(b1.numel())
            return b2

        x0 = fn(x)
        x0.mul_(2)
        x0.sum().backward()

        y0 = fn(y)
        y1 = y0.mul(2)
        y1.sum().backward()

        self.assertEqual(x.grad, y.grad)

        # case 3: parent does not have a view_func but child does
        x = torch.randn(10, dtype=torch.cdouble, requires_grad=True)
        y = x.detach().requires_grad_(True)

        def fn(a, dim0_size=5):
            b = a.clone()
            b1 = b.reshape(dim0_size, 2)
            b2 = torch.view_as_real(b1)
            return b2

        x0 = fn(x)
        x0.mul_(2)
        x0.sum().backward()

        y0 = fn(y)
        y1 = y0.mul(2)
        y1.sum().backward()

        self.assertEqual(x.grad, y.grad)

    def test_view_with_multi_output(self):
        x = torch.randn(2, 2, 2, dtype=torch.double)

        x1 = torch.view_as_complex(x)
        # Taking an invalid view should always be allowed as long as it is not
        # modified inplace
        res = x1.unbind(0)

        with self.assertRaisesRegex(RuntimeError, "output of a function that returns multiple views"):
            res[0] += torch.rand(2, requires_grad=True)

        x.requires_grad_(True)
        x1 = torch.view_as_complex(x)
        # Taking an invalid view should always be allowed as long as it is not
        # modified inplace
        res = x1.unbind(0)

        with self.assertRaisesRegex(RuntimeError, "output of a function that returns multiple views"):
            res[0] += torch.rand(2, requires_grad=True)

    def as_identity(self):
        # view_as_real and view_as_complex behavior should be like an identity
        def func(z):
            z_ = torch.view_as_complex(z)
            z_select = torch.select(z_, z_.dim() - 1, 0)
            z_select_real = torch.view_as_real(z_select)
            return z_select_real.sum()

        z = torch.randn(10, 2, 2, dtype=torch.double, requires_grad=True)
        gradcheck(func, [z])
        func(z).backward()

        z1 = z.clone().detach().requires_grad_(True)
        torch.select(z1, z1.dim() - 2, 0).sum().backward()

        self.assertEqual(z.grad, z1.grad)

class TestAutogradFunctional(TestCase):
    def _assert_same_struct(self, res, base):
        # base and res should be Tensors or tuple of Tensors with the same size
        if isinstance(base, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(base.size(), res.size())
        elif isinstance(base, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(base), len(res))
            for el_base, el_res in zip(base, res):
                self.assertTrue(isinstance(el_base, torch.Tensor))
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertEqual(el_base.size(), el_res.size())
        else:
            # Wrong base
            raise RuntimeError("The base given to `_assert_same_struct` doesn't have"
                               " the right structure.")

    def _assert_interleaved_struct(self, res, base1, base2):
        # base1 and base2 can be Tensors or tuples of Tensors.
        # If they are tuples, res should be a tuple as well.
        # The indexing works as follows for base1, base2 being
        # - tuple, tuple: res[i][j][k][l] = (base1[i][k], base2[j][l])
        # - tuple, Tensor: res[i][k][l] = (base1[i][k], base2[l])
        # - Tensor, tuple: res[i][j][l] = (base1[i], base2[j][l])
        # - Tensor, Tensor: res[k][l] = (base1[k], base2[l])
        if isinstance(base1, torch.Tensor) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(res.size(), base1.size() + base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base1, torch.Tensor))
                self.assertEqual(el_res.size(), el_base1.size() + base2.size())
        elif isinstance(base1, torch.Tensor) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base2))
            for el_res, el_base2 in zip(res, base2):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base2, torch.Tensor))
                self.assertEqual(el_res.size(), base1.size() + el_base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, tuple))
                self.assertEqual(len(res), len(base2))
                for el_el_res, el_base2 in zip(el_res, base2):
                    self.assertTrue(isinstance(el_el_res, torch.Tensor))
                    self.assertTrue(isinstance(el_base2, torch.Tensor))
                    self.assertEqual(el_el_res.size(), el_base1.size() + el_base2.size())
        else:
            # Wrong bases
            raise RuntimeError("The bases given to `_assert_interleaved_struct` don't have"
                               " the right structure.")

    def test_vjp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = torch.rand(4)
        v = torch.ones(3)
        with self.assertRaisesRegex(TypeError, "The inputs given to vjp must be either a Tensor"):
            res = autogradF.vjp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to vjp must"):
            res = autogradF.vjp(bar, inp, v)

        with self.assertRaisesRegex(RuntimeError, "The vector v can only be None if the user-provided function returns"):
            res = autogradF.vjp(foo, inp)

        with self.assertRaisesRegex(RuntimeError, "The given v should contain a single Tensor."):
            res = autogradF.vjp(foo, inp, (torch.ones_like(inp), torch.ones_like(inp)))

        with self.assertRaisesRegex(RuntimeError, "v has invalid size: should be torch.Size"):
            res = autogradF.vjp(foo, inp, v[:2])

        res = autogradF.vjp(foo, inp, v)[1]
        self._assert_same_struct(res, inp)

    def test_vjp_err_check_strict(self):
        def foo(a):
            return a.detach()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.vjp(foo, inp, v, strict=True)
        res = autogradF.vjp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.vjp(bar, inp, v, strict=True)
        res = autogradF.vjp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        # The Jacobian does not depend on the input
        def foo(a):
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function is independent of input 0."):
            res = autogradF.vjp(foo, inp, v, create_graph=True, strict=True)
        res = autogradF.vjp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)

    def test_vjp_no_grad(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4)
        with torch.no_grad():
            res = autogradF.vjp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_vjp_output(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4)
        res = autogradF.vjp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y

        inputs = (torch.rand(2), torch.rand(2))
        v = torch.ones(2)
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        self._assert_same_struct(vjp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vjp_val[0].grad_fn)
        self.assertIsNone(vjp_val[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y, x + y

        inputs = (torch.rand(2), torch.rand(2))
        v = (torch.tensor([1., 0.]), torch.tensor([1., 0.]))
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        self._assert_same_struct(vjp_val, inputs)
        self.assertIsNone(out[0].grad_fn)
        self.assertIsNone(out[1].grad_fn)
        self.assertIsNone(vjp_val[0].grad_fn)
        self.assertIsNone(vjp_val[1].grad_fn)

    def test_vjp_scalar(self):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        v = torch.ones([])
        res = autogradF.vjp(reducer, inputs, v)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.vjp(reducer, inputs)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

        def expander(x):
            return x.unsqueeze(0).repeat(4)
        inputs = torch.rand([])
        v = torch.ones(4)
        res = autogradF.vjp(expander, inputs, v)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

    def test_vjp_create_graph(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(2, 2, dtype=torch.double)
        v = torch.ones(2, dtype=torch.double)

        inputs.requires_grad_()
        v.requires_grad_()
        res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True), (inputs, v))

        def adder(x, y):
            return 2 * x + 3 * y, x * y

        inputs = (torch.rand(2, dtype=torch.double, requires_grad=True),
                  torch.rand(2, dtype=torch.double, requires_grad=True))
        v = (torch.tensor([1., 0.], dtype=torch.double, requires_grad=True),
             torch.tensor([1., 0.], dtype=torch.double, requires_grad=True))

        gradcheck(lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.vjp(adder, (x, y), v, create_graph=True)

            return val[0].exp() + val[1].exp() + grad[0].exp() + grad[1].exp() + x.exp() + y.exp()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def test_jvp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to jvp must be either a Tensor"):
            res = autogradF.jvp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to jvp must"):
            res = autogradF.jvp(bar, inp, v)

        with self.assertRaisesRegex(RuntimeError, "The vector v can only be None if the input to the user-provided function"):
            res = autogradF.jvp(foo, inp)

        with self.assertRaisesRegex(RuntimeError, "The given v should contain a single Tensor."):
            res = autogradF.jvp(foo, inp, (v, v))

        with self.assertRaisesRegex(RuntimeError, "v has invalid size: should be torch.Size"):
            res = autogradF.jvp(foo, inp, v[:2])

        res = autogradF.jvp(foo, inp, v)[1]
        self._assert_same_struct(res, foo(inp))

    def test_jvp_err_check_strict(self):
        def foo(a):
            return a.detach()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.jvp(foo, inp, v, strict=True)
        res = autogradF.jvp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.jvp(bar, inp, v, strict=True)
        res = autogradF.jvp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.)

        # The Jacobian does not depend on the input
        def foo(a):
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function is independent of input 0."):
            res = autogradF.jvp(foo, inp, v, create_graph=True, strict=True)
        res = autogradF.jvp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)

    def test_jvp_no_grad(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_jvp_output(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y

        inputs = (torch.rand(2), torch.rand(2))
        v = (torch.ones(2), torch.ones(2))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y, x + y

        inputs = (torch.rand(2), torch.rand(2))
        v = (torch.tensor([1., 0.]), torch.tensor([1., 0.]))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out[0].grad_fn)
        self.assertIsNone(out[1].grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

    def test_jvp_scalar(self):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[0], torch.zeros([]))
        self._assert_same_struct(res[1], res[0])

        def expander(x):
            return x.unsqueeze(0).repeat(4)
        inputs = torch.rand([])
        v = torch.ones([])
        res = autogradF.jvp(expander, inputs, v)
        self._assert_same_struct(res[0], torch.zeros(4))
        self._assert_same_struct(res[1], res[0])

        res = autogradF.jvp(expander, inputs)
        self._assert_same_struct(res[0], torch.zeros(4))
        self._assert_same_struct(res[1], res[0])

    def test_jvp_create_graph(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(2, 2, dtype=torch.double)
        v = torch.ones(2, 2, dtype=torch.double)

        inputs.requires_grad_()
        v.requires_grad_()
        res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True), (inputs, v))

        def adder(x, y):
            return 2 * x + 3 * y, x * y

        inputs = (torch.rand(2, dtype=torch.double, requires_grad=True),
                  torch.rand(2, dtype=torch.double, requires_grad=True))
        v = (torch.tensor([1., 0.], dtype=torch.double, requires_grad=True),
             torch.tensor([1., 0.], dtype=torch.double, requires_grad=True))

        gradcheck(lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.jvp(adder, (x, y), v, create_graph=True)

            return val[0].exp() + val[1].exp() + grad[0].exp() + grad[1].exp() + x.exp() + y.exp()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def _test_construct_standard_basis_for(self, inputs):
        numels = tuple(tensor.numel() for tensor in inputs)
        results = autogradF._construct_standard_basis_for(inputs, numels)
        for result, inp in zip(results, inputs):
            self.assertEqual(result.dtype, inp.dtype)
            self.assertEqual(result.device, inp.device)
        results = torch.cat([result.to(device='cpu', dtype=torch.float)
                             for result in results], dim=1)
        expected = torch.eye(results[0].shape[0], dtype=torch.float)
        self.assertEqual(results, expected)

    def test_construct_standard_basis_for(self):
        test_cases = [
            (torch.randn(2, 3),),
            (torch.randn(1),),
            (torch.randn([]),),
            (torch.randn(1), torch.randn([]), torch.randn([])),
            (torch.randn(2), torch.randn(3), torch.randn([])),
            (torch.randn(2), torch.randn([]), torch.randn(3)),
            (torch.randn(2, 3), torch.randn(3), torch.randn(3, 4, 2)),
            (torch.randn(2, dtype=torch.float64), torch.randn(3, dtype=torch.float32)),
        ]

        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_construct_standard_basis_for_cuda(self):
        test_cases = [
            (torch.randn(2), torch.randn(3, device='cuda')),
            (torch.randn(3, device='cuda'), torch.randn(2)),
        ]

        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    def _test_vectorize_raises_no_warnings(self, api):
        # vmap is an experimental prototype. When someone calls torch.vmap,
        # it raises a python warning. This test checks that
        # autogradF.{jacobian, hessian} don't raise that experimental prototype
        # warning; it is not nice for a public-facing API to raise a warning
        # no matter how it is called.
        def foo(a):
            return (a ** 2).sum()

        x = torch.randn(3)
        with warnings.catch_warnings(record=True) as wa:
            result = api(foo, x, vectorize=True)
        self.assertEqual(len(wa), 0)

    def test_jacobian_vectorize_raises_no_warnings(self):
        return self._test_vectorize_raises_no_warnings(autogradF.jacobian)

    def test_hessian_vectorize_raises_no_warnings(self):
        return self._test_vectorize_raises_no_warnings(autogradF.hessian)

    def _test_jacobian_err_check(self, vectorize):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to jacobian must be either a Tensor"):
            res = autogradF.jacobian(foo, (inp, 2), vectorize=vectorize)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to jacobian must"):
            res = autogradF.jacobian(bar, inp, vectorize=vectorize)

        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(inp), inp)

        def foo(a, b):
            return b, 3 * a.narrow(0, 0, 3)

        inp = (torch.rand(4), torch.rand(5))

        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(*inp), inp)

    def test_jacobian_err_check(self):
        return self._test_jacobian_err_check(vectorize=False)

    def test_jacobian_err_check_vectorize(self):
        return self._test_jacobian_err_check(vectorize=True)

    def test_jacobian_err_check_strict(self):
        def foo(a):
            return a.detach()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.jacobian(foo, inp, strict=True)
        res = autogradF.jacobian(foo, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function is independent of input 0."):
            res = autogradF.jacobian(bar, inp, strict=True)
        res = autogradF.jacobian(bar, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.)

        # The Jacobian does not depend on the input
        def foo(a):
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function is independent of input 0."):
            res = autogradF.jacobian(foo, inp, create_graph=True, strict=True)
        res = autogradF.jacobian(foo, inp, create_graph=True, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res, torch.eye(4))

    def test_jacobian_err_check_strict_vectorize(self):
        def foo(x):
            return x

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            res = autogradF.jacobian(foo, inp, strict=True, vectorize=True)

    def test_jacobian_no_grad(self):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        inputs = torch.rand(4, 4)
        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs)
        self.assertIsNone(res.grad_fn)
        self.assertNotEqual(res, torch.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs, create_graph=True)
        self.assertIsNotNone(res.grad_fn)
        self.assertNotEqual(res, torch.zeros(4, 4))

    def _test_jacobian_output(self, vectorize):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        inputs = torch.rand(4, 4)
        res = autogradF.jacobian(exp_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        self.assertIsNone(res.grad_fn)

        def identity(x):
            return x.clone()

        inputs = torch.rand(4)
        res = autogradF.jacobian(identity, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, identity(inputs), inputs)
        self.assertIsNone(res.grad_fn)
        self.assertEqual(res, torch.eye(4))

        def add_exp_reducer(x, y):
            return (x + y.exp()).sum(dim=1)

        inputs = (torch.rand(4, 4), torch.rand(4, 4))
        res = autogradF.jacobian(add_exp_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

    def test_jacobian_output(self):
        self._test_jacobian_output(vectorize=False)

    def test_jacobian_output_vectorize(self):
        self._test_jacobian_output(vectorize=True)

    def _test_jacobian_scalar(self, vectorize):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        res = autogradF.jacobian(reducer, inputs, vectorize=vectorize)
        self._assert_same_struct(res, inputs)

        def expander(x):
            return x.unsqueeze(0).repeat(4)
        inputs = torch.rand([])
        res = autogradF.jacobian(expander, inputs, vectorize=vectorize)
        self._assert_same_struct(res, torch.zeros(4))

    def test_jacobian_scalar(self):
        self._test_jacobian_scalar(vectorize=False)

    def test_jacobian_scalar_vectorize(self):
        self._test_jacobian_scalar(vectorize=True)

    def _test_jacobian_create_graph(self, vectorize):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        inputs = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.jacobian(exp_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        self.assertIsNotNone(res.grad_fn)

        gradcheck(lambda inp: autogradF.jacobian(exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)
        gradgradcheck(lambda inp: autogradF.jacobian(exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)

        def add_exp_reducer(x, y):
            return (x + y).exp().sum(dim=1)

        inputs = (torch.rand(4, 4, dtype=torch.double, requires_grad=True),
                  torch.rand(4, 4, dtype=torch.double, requires_grad=True))
        res = autogradF.jacobian(add_exp_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda *inp: autogradF.jacobian(add_exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)
        gradgradcheck(lambda *inp: autogradF.jacobian(add_exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)

        def foo(x, y):
            x = x.cos()
            val, jac = autogradF.jacobian(add_exp_reducer, (x, y), create_graph=True, vectorize=vectorize)

            res = val[0].exp().sum() + val[1].exp().sum() + jac[0].exp().sum()
            res = res + jac[1].exp().sum() + x.exp().sum() + y.exp().sum()
            return res

        gradcheck(foo, inputs)
        gradgradcheck(foo, inputs)

    def test_jacobian_create_graph(self):
        self._test_jacobian_create_graph(vectorize=False)

    def test_jacobian_create_graph_vectorize(self):
        self._test_jacobian_create_graph(vectorize=True)

    def _check_jacobian_vectorize_correctness(self, f, inputs):
        expected = autogradF.jacobian(f, inputs, vectorize=False)
        result = autogradF.jacobian(f, inputs, vectorize=True)
        self.assertEqual(result, expected)

    def test_jacobian_vectorize_correctness_simple(self):
        def f(x):
            return 3 * x ** 2

        x = torch.randn(2, 3, 5)
        self._check_jacobian_vectorize_correctness(f, x)

    def test_jacobian_vectorize_correctness_multi_input(self):
        def f(x, y):
            return (x.cos() * x) @ y.sin()

        x = torch.randn(2, 3)
        y = torch.randn(3, 5)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_multi_input_multi_output(self):
        def f(x, y):
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        x = torch.randn(5, 3)
        y = torch.randn(3, 5)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_unrelated_outputs(self):
        def f(x, y):
            return x, y, x, y

        x = torch.randn(2)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_zero_dim(self):
        # zero-dim output
        def f(x, y):
            return x.sum(), y.sum(), x * y

        x = torch.randn(3)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

        # zero-dim input
        def g(x):
            return torch.stack([x, x, x])

        x = torch.randn([])
        self._check_jacobian_vectorize_correctness(g, x)

        # Mixed zero-dim input / zero-dim output
        def h(x, y):
            return y.sum(), x * y

        x = torch.randn([])
        y = torch.randn(1)
        self._check_jacobian_vectorize_correctness(h, (x, y))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_jacobian_vectorize_correctness_different_devices(self):
        def f(x, y):
            return x * y, (x * y).cuda()

        x = torch.randn(3)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_different_dtype(self):
        def f(x, y):
            return (x * y).float(), (x * y).double()

        x = torch.randn(3)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def _check_hessian_vectorize_correctness(self, f, inputs):
        expected = autogradF.hessian(f, inputs, vectorize=False)
        result = autogradF.hessian(f, inputs, vectorize=True)
        self.assertEqual(result, expected)

    def test_hessian_vectorize_correctness_simple(self):
        def f(x):
            return (3 * x ** 2).sum()

        x = torch.randn(2, 3, 5)
        self._check_hessian_vectorize_correctness(f, x)

    def test_hessian_vectorize_correctness_multi_input(self):
        def f(x, y, z):
            return ((x.relu() * x) @ y.sin() @ z).sum()

        x = torch.randn(2, 3)
        y = torch.randn(3, 5)
        z = torch.randn(5, 5)
        self._check_hessian_vectorize_correctness(f, (x, y, z))

    def test_hessian_vectorize_correctness_unrelated_outputs(self):
        # output unrelated to one input
        def f(x, y):
            return (x ** 2).sum()

        x = torch.randn(2)
        y = torch.randn(3)
        self._check_hessian_vectorize_correctness(f, (x, y))

        # output unrelated to all inputs
        def f(x, y):
            return torch.randn([])

        x = torch.randn(2)
        y = torch.randn(3)
        self._check_hessian_vectorize_correctness(f, (x, y))

    def _test_hessian_err_check(self, vectorize):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            return 3 * a.narrow(0, 0, 3)

        def bar3(a):
            return 3 * a.narrow(0, 0, 3), 3 * a.narrow(0, 0, 3)

        inp = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to hessian must be either a Tensor"):
            res = autogradF.hessian(foo, (inp, 2), vectorize=vectorize)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to hessian must"):
            res = autogradF.hessian(bar, inp, vectorize=vectorize)

        err_msg_out = "The Tensor returned by the function given to hessian should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.hessian(bar2, inp, vectorize=vectorize)

        with self.assertRaisesRegex(RuntimeError, "The function given to hessian should return a single Tensor"):
            res = autogradF.hessian(bar3, inp, vectorize=vectorize)

        res = autogradF.hessian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, inp, inp)

        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (torch.rand(4), torch.rand(5))

        res = autogradF.hessian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, inp, inp)

    def test_hessian_err_check(self):
        self._test_hessian_err_check(vectorize=False)

    def test_hessian_err_check_vectorize(self):
        self._test_hessian_err_check(vectorize=True)

    def test_hessian_err_check_strict(self):
        def foo(a):
            return a.detach().sum()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.hessian(foo, inp, strict=True)
        res = autogradF.hessian(foo, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0"):
            res = autogradF.hessian(bar, inp, strict=True)
        res = autogradF.hessian(bar, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0 is"):
            res = autogradF.hessian(bar2, inp, strict=True)
        res = autogradF.hessian(bar2, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.)

    def test_hessian_err_check_strict_vectorize(self):
        def foo(x):
            return (x ** 3).sum()

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            res = autogradF.hessian(foo, inp, strict=True, vectorize=True)

    def test_hessian_no_grad(self):
        def pow_reducer(x):
            return x.pow(3).sum()

        inputs = torch.rand(2, 2)
        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs)
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)
        self.assertNotEqual(res, torch.zeros(2, 2, 2))

        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs, create_graph=True)
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)
        self.assertNotEqual(res, torch.zeros(2, 2, 2))


    def _test_hessian_output(self, vectorize):
        def pow_reducer(x):
            return x.pow(3).sum()

        inputs = torch.rand(2, 2)
        res = autogradF.hessian(pow_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNone(res.grad_fn)

        def add_pow_reducer(x, y):
            return (x + y).pow(3).sum()

        inputs = (torch.rand(2, 2), torch.rand(2, 2))
        res = autogradF.hessian(add_pow_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)

    def test_hessian_output(self):
        self._test_hessian_output(vectorize=False)

    def test_hessian_output_vectorize(self):
        self._test_hessian_output(vectorize=True)

    def _test_hessian_scalar(self, vectorize):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)

        inputs = torch.rand([])
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        self._assert_same_struct(res, inputs)

        def bad_reducer(x):
            return x.sum().view(1, 1, 1)
        inputs = torch.rand(4, 4)
        res = autogradF.hessian(bad_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)

    def test_hessian_scalar(self):
        return self._test_hessian_scalar(vectorize=False)

    def test_hessian_scalar_vectorize(self):
        return self._test_hessian_scalar(vectorize=True)

    def _test_hessian_create_graph(self, vectorize):
        def pow_reducer(x):
            return x.pow(3).sum()

        inputs = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
        res = autogradF.hessian(pow_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNotNone(res.grad_fn)

        gradcheck(lambda inp: autogradF.hessian(pow_reducer, inp, create_graph=True, vectorize=vectorize), inputs)
        gradgradcheck(lambda inp: autogradF.hessian(pow_reducer, inp, create_graph=True, vectorize=vectorize), inputs)

        def add_pow_reducer(x, y):
            return (x + y).pow(3).sum()

        inputs = (torch.rand(2, 2, dtype=torch.double, requires_grad=True),
                  torch.rand(2, 2, dtype=torch.double, requires_grad=True))
        res = autogradF.hessian(add_pow_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)

        def flatten(inp):
            return tuple(el_lvl2 for el_lvl1 in inp for el_lvl2 in el_lvl1)

        gradcheck(lambda *inp: flatten(autogradF.hessian(add_pow_reducer, inp, create_graph=True, vectorize=vectorize)), inputs)
        gradgradcheck(lambda *inp: flatten(autogradF.hessian(add_pow_reducer, inp, create_graph=True, vectorize=vectorize)), inputs)

        def foo(x, y):
            x = x.cos()
            val, hess = autogradF.hessian(add_pow_reducer, (x, y), create_graph=True, vectorize=vectorize)

            res = val[0].cos().sum() + val[1].cos().sum() + hess[0].cos().sum()
            res = res + hess[1].cos().sum() + x.cos().sum() + y.cos().sum()
            return res

        gradcheck(foo, inputs)
        gradgradcheck(foo, inputs)

    def test_hessian_create_graph(self):
        self._test_hessian_create_graph(vectorize=False)

    def test_hessian_create_graph_vectorize(self):
        self._test_hessian_create_graph(vectorize=True)

    def test_vhp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            return 3 * a.narrow(0, 0, 3)

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to vhp must be either a Tensor"):
            res = autogradF.vhp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to vhp must"):
            res = autogradF.vhp(bar, inp, v)

        err_msg_out = "The Tensor returned by the function given to vhp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.vhp(bar2, inp, v)

        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.vhp(foo, inp, torch.rand(5))

        with self.assertRaisesRegex(TypeError, "The v given to vhp must be either a Tensor or a tuple of Tensors"):
            res = autogradF.vhp(foo, inp, (v, 2))

        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (torch.rand(4), torch.rand(5))
        v = (torch.rand(4), torch.rand(5))

        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    def test_vhp_err_check_strict(self):
        def foo(a):
            return a.detach().sum()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.vhp(foo, inp, v, strict=True)
        res = autogradF.vhp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.vhp(bar, inp, v, strict=True)
        res = autogradF.vhp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0 is"):
            res = autogradF.vhp(bar2, inp, v, strict=True)
        res = autogradF.vhp(bar2, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

    def test_vhp_no_grad(self):
        def reducer(x):
            return x.exp().sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        with torch.no_grad():
            res = autogradF.vhp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.vhp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_vhp_output(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.vhp(foo, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3), torch.rand(4))
        v = (torch.ones(3), torch.ones(4))
        out, vhp_val = autogradF.vhp(bar, inputs, v)
        self._assert_same_struct(vhp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vhp_val[0].grad_fn)
        self.assertIsNone(vhp_val[1].grad_fn)

    def test_vhp_scalar(self):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.vhp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        inputs = torch.rand([])
        v = torch.rand([])
        res = autogradF.vhp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.vhp(reducer, inputs)
        self._assert_same_struct(res[1], inputs)

        def bad_reducer(x):
            return x.sum().view(1, 1, 1)
        inputs = torch.rand(4, 4)
        v = torch.rand(4, 4)
        res = autogradF.vhp(bad_reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

    def test_vhp_create_graph(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        v = torch.ones(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.vhp(foo, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v))

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3, dtype=torch.double, requires_grad=True),
                  torch.rand(4, dtype=torch.double, requires_grad=True))
        v = (torch.ones(3, dtype=torch.double, requires_grad=True),
             torch.ones(4, dtype=torch.double, requires_grad=True))
        out, vhp_val = autogradF.vhp(bar, inputs, v, create_graph=True)
        self._assert_same_struct(vhp_val, inputs)
        self.assertIsNotNone(out.grad_fn)
        self.assertIsNotNone(vhp_val[0].grad_fn)
        self.assertIsNotNone(vhp_val[1].grad_fn)

        gradcheck(lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.vhp(bar, (x, y), v, create_graph=True)

            return val.cos() + grad[0].cos().sum() + grad[1].cos() + x.cos().sum() + y.cos()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def test_hvp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            return 3 * a.narrow(0, 0, 3)

        inp = torch.rand(4)
        v = torch.rand(4)
        res = autogradF.hvp(foo, inp, v)
        with self.assertRaisesRegex(TypeError, "The inputs given to hvp must be either a Tensor"):
            res = autogradF.hvp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to hvp must"):
            res = autogradF.hvp(bar, inp, v)

        err_msg_out = "The Tensor returned by the function given to hvp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.hvp(bar2, inp, v)

        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.hvp(foo, inp, torch.rand(5))

        with self.assertRaisesRegex(TypeError, "The v given to hvp must be either a Tensor or a tuple of Tensors"):
            res = autogradF.hvp(foo, inp, (v, 2))

        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (torch.rand(4), torch.rand(5))
        v = (torch.rand(4), torch.rand(5))

        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    def test_hvp_err_check_strict(self):
        def foo(a):
            return a.detach().sum()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.hvp(foo, inp, v, strict=True)
        res = autogradF.hvp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.hvp(bar, inp, v, strict=True)
        res = autogradF.hvp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0 is"):
            res = autogradF.hvp(bar2, inp, v, strict=True)
        res = autogradF.hvp(bar2, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

    def test_hvp_no_grad(self):
        def reducer(x):
            return x.exp().sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        with torch.no_grad():
            res = autogradF.hvp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.hvp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_hvp_output(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.hvp(foo, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3), torch.rand(4))
        v = (torch.ones(3), torch.ones(4))
        out, hvp_val = autogradF.hvp(bar, inputs, v)
        self._assert_same_struct(hvp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(hvp_val[0].grad_fn)
        self.assertIsNone(hvp_val[1].grad_fn)

    def test_hvp_scalar(self):
        def reducer(x):
            return x.exp().sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.hvp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        inputs = torch.rand([])
        v = torch.rand([])
        res = autogradF.hvp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.hvp(reducer, inputs)
        self._assert_same_struct(res[1], inputs)

        def bad_reducer(x):
            return x.exp().sum().view(1, 1, 1)
        inputs = torch.rand(4, 4)
        v = torch.rand(4, 4)
        res = autogradF.hvp(bad_reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

    def test_hvp_create_graph(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        v = torch.ones(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.hvp(foo, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v))

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3, dtype=torch.double, requires_grad=True),
                  torch.rand(4, dtype=torch.double, requires_grad=True))
        v = (torch.ones(3, dtype=torch.double, requires_grad=True),
             torch.ones(4, dtype=torch.double, requires_grad=True))
        out, hvp_val = autogradF.hvp(bar, inputs, v, create_graph=True)
        self._assert_same_struct(hvp_val, inputs)
        self.assertIsNotNone(out.grad_fn)
        self.assertIsNotNone(hvp_val[0].grad_fn)
        self.assertIsNotNone(hvp_val[1].grad_fn)

        gradcheck(lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.hvp(bar, (x, y), v, create_graph=True)

            return val.cos() + grad[0].cos().sum() + grad[1].cos() + x.cos().sum() + y.cos()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def test_jacobian_match_vjp_jvp(self):
        def foo(x):
            return x ** 3 + x.sum()

        inputs = torch.rand(4)
        v = torch.rand(4)

        jac = autogradF.jacobian(foo, inputs)
        jvp = autogradF.jvp(foo, inputs, v)[1]
        vjp = autogradF.vjp(foo, inputs, v)[1]

        self.assertEqual(jvp, torch.mm(jac, v.unsqueeze(1)).squeeze(1))
        self.assertEqual(vjp, torch.mm(v.unsqueeze(0), jac).squeeze(0))

    def test_hessian_match_vhp_hvp(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4)
        v = torch.rand(4)

        hes = autogradF.hessian(foo, inputs)
        hvp = autogradF.hvp(foo, inputs, v)[1]
        vhp = autogradF.vhp(foo, inputs, v)[1]

        self.assertEqual(hvp, torch.mm(hes, v.unsqueeze(1)).squeeze(1))
        self.assertEqual(vhp, torch.mm(v.unsqueeze(0), hes).squeeze(0))

class TestAutogradForwardMode(TestCase):
    def tearDown(self):
        # Ensure that a failing test won't make others fail
        while fwAD._current_level >= 0:
            fwAD.exit_dual_level()

        super().tearDown()

    def test_forward_level_cleanup(self):
        def get_tensor_and_weak_ref():
            # Create a new Tensor and weak reference
            t = torch.rand(2, requires_grad=True)
            return t, torch._C._WeakTensorRef(t)

        # Sanity check that the helper function works as expected
        t, t_ref = get_tensor_and_weak_ref()
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        # Main test code
        foo = torch.rand(2)

        with fwAD.dual_level():
            tangent, tangent_ref = get_tensor_and_weak_ref()
            self.assertFalse(tangent_ref.expired())

            dual = fwAD.make_dual(foo, tangent)
            self.assertFalse(tangent_ref.expired())

            # Make sure that the tangent we provided has been re-used as is
            self.assertTrue(fwAD.unpack_dual(dual)[1] is tangent)

            # Make sure that dual is keeping the tangent alive
            del tangent
            self.assertFalse(tangent_ref.expired())

            # Make sure that the dual level does not keep the c++
            # version of the tangent alive
            del dual
            self.assertTrue(tangent_ref.expired())

    def test_size_check(self):
        foo = torch.rand(2)
        tangent = torch.rand(3)

        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, "Trying to set a forward gradient that has a different size"):
                dual = fwAD.make_dual(foo, tangent)

            dual = fwAD.make_dual(foo, tangent[1:])

    # The following test functions want to ensure all the following behaviors:
    #   - Ensure that default level system in the python binding works
    #   - Ensure that only level 0 exists and nesting is properly disabled
    #   - Ensure that printing works fine
    #   - Ensure that basic packing/unpacking works
    #   - Ensure that advanced packing/unpacking works
    #     - For memory / version counter share
    #     - For backward AD (regular ops)
    #   - Ensure that view + inplace for both modes work fine
    #   - Ensure we do proper cleanup on exit of a level

    def test_default_level(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        # We don't actually need to enforce that these two are the exact same python
        # object, feel free to relax in the future
        self.assertIs(baz_tangent, bar)

        baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertEqual(baz_tangent, None)

    def test_nested_level(self):
        with fwAD.dual_level() as level:
            # For now only level 0 exists
            self.assertEqual(level, 0)

        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, "Nested forward mode AD is not supported at the moment"):
                nest_level = fwAD.enter_dual_level()

    def test_print(self):
        with fwAD.dual_level() as level:
            a = torch.rand(3)
            self.assertFalse("tangent=" in str(a))

            b = fwAD.make_dual(a, torch.rand(3))
            self.assertFalse("tangent=" in str(a))
            self.assertTrue("tangent=" in str(b))

            b_primal, b_tangent = fwAD.unpack_dual(b)
            self.assertFalse("tangent=" in str(b_primal))
            self.assertFalse("tangent=" in str(b_tangent))

    def test_basic_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
            self.assertEqual(baz_primal, foo)
            self.assertIs(baz_tangent, bar)

            # Check that packing/unpacking did not change the input
            foo_primal, foo_tangent = fwAD.unpack_dual(foo)
            self.assertEqual(foo_primal, foo)
            self.assertIsNone(foo_tangent)

    def test_advanced_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.ones(2)

        # Memory and version counter check
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)

            # Ensure that they are sharing memory and version counter
            self.assertEqual(dual.storage().data_ptr(), foo.storage().data_ptr())

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual._version)

            # Unpacking should only create aliases as well
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertEqual(dual_primal.storage().data_ptr(), foo.storage().data_ptr())
            self.assertEqual(dual_tangent.storage().data_ptr(), bar.storage().data_ptr())
            # And the tangent is actually re-used as-is so it is still the same Tensor
            self.assertIs(dual_tangent, bar)

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual_primal._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual_primal._version)
            self.assertEqual(bar._version, dual_tangent._version)
            bar.add_(1)
            self.assertEqual(bar._version, dual_tangent._version)

        # backward mode check
        with fwAD.dual_level():
            foo.requires_grad_()
            bar.requires_grad_()

            # Check that backward gradients properly propagates through packing/unpacking
            dual = fwAD.make_dual(foo, bar)
            p, t = fwAD.unpack_dual(dual)

            gfoo, gbar = torch.autograd.grad(p.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertEqual(gfoo, torch.ones_like(foo))
            self.assertIsNone(gbar)

            gfoo, gbar = torch.autograd.grad(t.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertIsNone(gfoo)
            self.assertEqual(gbar, torch.ones_like(bar))

            # Check that forward gradients are impacted by detach()
            detached_dual = dual.detach()
            out = detached_dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

            # Check that forward gradients are not impacted by no_grad
            with torch.no_grad():
                out = dual * 3
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 3)
            self.assertEqual(t, bar * 3)

            # Check that forward gradients are not impacted by inplace detach
            dual = dual.clone()
            dual.detach_()
            out = dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertIsNone(t)

    def test_view_inplace_non_differentiable_views(self):
        original_foo = torch.rand(2, dtype=torch.double)
        original_bar = torch.ones(2, dtype=torch.double)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Note that in this test, we use "update" to mean computing the right tangent for the dual
            # All the inplace operations here are expected to update the primal value of the Tensors but
            # not always their tangents.
            # Also all mentions of "non differentiable view" here means non forward differentiable view
            # unless specified otherwise.
            # See note [Forward Grad View/inplace] for more details on how these views work.

            # Check that inplace ops do not update non-differentiable views
            # Non differentiable view
            dual = fwAD.make_dual(foo, bar)
            dual *= 2
            # Check that non differentiable view's tangent was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that the computed result is correct
            self.assertEqual(bar, original_bar * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            self.assertEqual(foo, original_foo * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 2)
            # Other non differentiable view
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertIsNone(fwAD.unpack_dual(dual_primal)[1])
            self.assertIsNone(fwAD.unpack_dual(dual_tangent)[1])
            dual_primal *= 2
            # Ensure dual's tangent did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            dual_tangent *= 2
            # Ensure dual's primal did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 4)


    def test_view_inplace_differentiable_views(self):
        original_foo = torch.rand(2)
        original_bar = torch.ones(2)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Check that inplace ops do update differentiable view but stop at non differentiable ones
            # A non differentiable view
            dual = fwAD.make_dual(foo, bar)
            # A differentiable view
            view = dual.narrow(0, 0, 1)
            view *= 2
            # Check that non differentiable view was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that differentiable view was updated
            self.assertEqual(fwAD.unpack_dual(dual)[1], torch.tensor([2., 1.]))
            self.assertEqual(fwAD.unpack_dual(view)[1], torch.tensor([2.]))

            # Check that we track differentiable view even for Tensors that are not dual
            baz = torch.rand(2)
            baz += dual
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])
            # Updates on view should as well
            baz = torch.rand(2)
            baz[0] = dual[0]
            self.assertEqual(fwAD.unpack_dual(baz)[1][0], fwAD.unpack_dual(dual)[1][0])
            # Unused values get a gradient of 0
            self.assertEqual(fwAD.unpack_dual(baz)[1][1], 0.)

            # Check that forward non-differentiable views do prevent gradient update
            baz = torch.rand(2)
            view = baz.detach()
            view += dual
            self.assertIsNone(fwAD.unpack_dual(baz)[1])

    def test_grad_cleanup(self):
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertIs(fwAD.unpack_dual(dual)[1], bar)

        self.assertIsNone(fwAD.unpack_dual(dual)[1])

        with fwAD.dual_level():
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            new_dual = fwAD.make_dual(foo, baz)

            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            new_dual_primal, new_dual_tangent = fwAD.unpack_dual(new_dual)
            self.assertEqual(dual_primal, new_dual_primal)
            self.assertIsNone(dual_tangent)
            self.assertEqual(new_dual_tangent, baz)

    def test_detach_view_tracking(self):
        # Default detach is both forward and backward non-differentiable
        foo = torch.rand(2)
        foo_weak = torch._C._WeakTensorRef(foo)

        out = foo.detach()

        del foo
        self.assertTrue(foo_weak.expired())


# Generic device type autograd tests.
class TestAutogradDeviceType(TestCase):

    def test_min_max_median_backprops_to_all_values(self, device):
        for f in [torch.min, torch.max, torch.median, torch.nanmedian]:
            x1 = torch.tensor([1., 0., 1., 0., 1., 0.], device=device, requires_grad=True)
            x2 = torch.tensor([float('nan'), float('nan'), float('nan')], requires_grad=True)
            for x in [x1, x2]:
                y = f(x)
                y.backward()
                self.assertEqual(x.grad.sum(), 1.)
                self.assertEqual((x.grad == 1 / 3).sum(), 3)

    def test_cdist(self, device):
        def _test_euclidean_large_cdist(sizex, sizey=None):
            if sizey is None:
                sizey = sizex
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # to avoid extremum
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            dist = torch.cdist(x, y, p=2)
            # Do a backward pass to check that it is valid for large
            # matrices
            loss = dist.sum()
            loss.backward()

        _test_euclidean_large_cdist((2000, 5))

    # Ensure that cdist backward with p<1 does not produce NaNs
    def test_cdist_grad_p_lt_1_no_nan(self, device):
        for p in [0.99, 0.7, 0.5, 0.1, 0.01]:
            x = torch.randn(1, 2, device=device)
            y = x.clone().detach() + torch.tensor([[1., 0.]], device=device)
            x.requires_grad = True
            y.requires_grad = True
            result = torch.cdist(x, y, p=p)
            result.backward(torch.ones_like(result))
            self.assertFalse(torch.isnan(x.grad).any())
            self.assertFalse(torch.isnan(y.grad).any())

    def test_cdist_same_inputs(self, device):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            eps = 1e-6
            x.requires_grad = True
            d = torch.cdist(x, y)
            d.backward(dist_grad)
            # Check that the backward passs does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()

    def test_parameter_resize(self, device):
        asd = torch.nn.Parameter(torch.ones(16, dtype=torch.double, device=device))

        for i in range(2):
            with torch.no_grad():
                asd.set_(asd[1:])
                asd.grad = None

            m = torch.cat((asd, asd))
            m.sum().backward()

    @dtypes(torch.double, torch.cdouble)
    def test_sparse_ctor_getter_backward(self, device, dtype):
        # See NOTE [ Sparse: autograd and API ] on the expected behavior of this test
        def _test(size, sparse_dim, nnz, device):
            v_size = [nnz] + list(size[sparse_dim:])
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)

            inp = torch.randn(v_size, dtype=torch.double, device=device, requires_grad=True)
            other = self.genSparseTensor(size, sparse_dim, nnz, is_uncoalesced=True, device=device,
                                         dtype=dtype)[0]

            def fn(v):
                x = torch.sparse_coo_tensor(i, v, size, dtype=dtype, device=device)
                y = (x + other).coalesce()
                yv = y.values()
                new_v = yv.tanh()
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                return z.coalesce().values()

            gradcheck(fn, (inp,), check_batched_grad=False)
            # FIXME: make gradgradcheck work.
            # gradgradcheck(fn, (inp,), check_batched_grad=False)

            # assert that _values is non-differentiable
            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.detach().requires_grad_()._values().backward(torch.ones_like(other._values()))

        for empty_i, empty_v, empty_nnz in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            _test(sparse_size + dense_size, len(sparse_size), nnz, device)

    @dtypes(torch.double, torch.cdouble)
    def test_sparse_backward(self, device, dtype):
        class FixedGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, grad_x):
                ctx.save_for_backward(grad_x)
                return x

            @staticmethod
            def backward(ctx, grad_x):
                saved_grad_x, = ctx.saved_tensors
                return saved_grad_x, None

        size = torch.Size([6, 3, 2])
        i1 = torch.tensor([
            [0, 3, 4],
            [0, 2, 2],
        ], dtype=torch.long)
        v1 = make_tensor([3, 2], dtype=dtype, device=device)
        sparse_grad1 = torch.sparse_coo_tensor(i1, v1, size, dtype=dtype, device=device)
        i2 = torch.tensor([
            [0, 1, 3, 4],
            [0, 1, 2, 2],
        ], dtype=torch.long)
        v2 = make_tensor([4, 2], dtype=dtype, device=device)
        sparse_grad2 = torch.sparse_coo_tensor(i2, v2, size, dtype=dtype, device=device)
        dense_grad = torch.rand(size, device=device, dtype=dtype)
        fn = FixedGradientFunction

        # sparse first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, dense_grad) + fn.apply(x, sparse_grad2)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # dense first
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, dense_grad) + fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().backward()
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)
        # sparse only
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        (fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().backward()
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    # autograd tests via common_method_invocations don't allow input tensors to
    # be sparse (RuntimeError: gradcheck expects all tensor inputs are dense when
    # check_sparse_nnz is set to False.)
    def test_sparse_mask_autograd(self, device):
        tensor = torch.randn(3, requires_grad=True, device=device)
        mask = torch.ones(3, device=device)
        mask[1] = 0
        mask = mask.to_sparse()
        converted = tensor.sparse_mask(mask).to_dense()
        converted.sum().backward()
        self.assertEqual(tensor.grad, mask.to_dense())

    def test_pyscalar_conversions(self, device):
        def _test_pyscalar_conversions(t, integral_conv):
            # integral -> integral
            l = t(torch.zeros(1, 1, 1, dtype=torch.long))
            pyscalar = -12345
            l[0] = pyscalar
            self.assertEqual(integral_conv(l), pyscalar)

            # floating point -> floating point
            f = Variable(t(torch.randn(1, 1, dtype=torch.double)))
            pyscalar = -12345.1
            f[0] = pyscalar
            self.assertEqual(float(f), pyscalar)
            f[0] = nan
            self.assertTrue(math.isnan(float(f)))
            f[0] = inf
            self.assertEqual(float(f), inf)
            f[0] = -inf
            self.assertEqual(float(f), -inf)

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


        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    @dtypesIfCUDA(torch.half, torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_set_requires_grad_only_for_floats(self, device, dtype):
        def f1():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad_()

        def f2():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad = True

        def f3():
            torch.ones(1, dtype=dtype, device=device, requires_grad=True)

        a = torch.ones(1, dtype=dtype, device=device)
        a.requires_grad = False  # should always work
        a.requires_grad_(False)

        for f in [f1, f2, f3]:
            if dtype.is_floating_point:
                f()
            else:
                with self.assertRaisesRegex(RuntimeError, 'floating point', msg="dt: {} device: {}".format(a.dtype, a.device)):
                    f()

    @onlyCUDA
    def test_advanced_indexing_backwards_large(self, device):
        # See https://github.com/pytorch/pytorch/issues/22843
        n = (1 << 16)
        x = torch.rand(n, 1, device=device, requires_grad=True)
        a = x[:, [0]]
        a.sum().backward()
        self.assertEqual(x.grad, torch.ones(n, 1, device=device))

    def test_advanced_indexing_backwards_memory_format(self, device):
        # See https://github.com/pytorch/pytorch/issues/36956
        shape = (2, 8, 1, 2)
        i = torch.randint(1, shape, device=device).contiguous(memory_format=torch.channels_last)
        x = torch.randn(shape, requires_grad=True, device=device)
        x[i].sum().backward()

    def _test_reentrant_parent_error_on_cpu(self, device):
        t1 = torch.rand([3, 3], requires_grad=True)
        t2 = torch.rand([3, 3], device=device, requires_grad=True)
        t3 = torch.rand([3, 3], device=device, requires_grad=True)

        # Parent graph cpu graph.
        t4 = t1 * t1
        t5 = TestAutograd.SimulateBackwardError.apply(t4)

        # Child gpu graph (much longer than parent graph).
        prev = t2 * t2
        for i in range(10):
            prev = prev * t2
        reentrant_root = prev

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will take much longer.
                reentrant_root.backward()
                return grad

        # Parent gpu graph.
        t6 = ReentrantFunc.apply(t3)
        t7 = t6 * t6

        # Parent graph will error out first, while child graph will continue executing.
        with self.assertRaisesRegex(Exception, "Simulate error"):
            torch.autograd.backward([t5.sum(), t7.sum()])

        # No grads should be accumulated since child graph will stop execution
        # after parent receives error.
        self.assertIsNone(t2.grad)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t3.grad)

    @onlyCUDA
    def test_reentrant_parent_error_on_cpu(self, device):
        before = CudaMemoryLeakCheck.get_cuda_memory_usage()

        # Run as separate function so that gc can clean up everything when we
        # check for memory usage.
        self._test_reentrant_parent_error_on_cpu(device)

        # Wait for autograd thread to cleanup failed tasks.
        after = CudaMemoryLeakCheck.get_cuda_memory_usage()
        start = time.time()
        while before != after and time.time() - start < 30:
            time.sleep(0.1)
            after = CudaMemoryLeakCheck.get_cuda_memory_usage()

        self.assertEqual(before, after)

    # test for backward in https://github.com/pytorch/pytorch/issues/15511
    def test_pdist_large(self, device):
        def func(x):
            return torch.pdist(x, p=2)

        # shape[0] should be able to be (roughly) arbitrarily large, but the kernel
        # is currently limited to smaller sizes (see issue above); this is just testing
        # a floor.
        shape = (1000, 1)
        x = torch.randn(shape, device=device).requires_grad_()
        output = torch.pdist(x, p=2)
        # just run a single backward, as gradcheck/gradgradcheck is expensive here
        output.sum().backward()

    def test_where_functional(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where(cond, x, y):
            return torch.where(cond, x, y)

        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, device=device)])

        x = torch.randn(5, 1, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, 1, dtype=torch.double, device=device, requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, 5, device=device)])

    def test_where_scalar(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        scalar = 4.
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where_scalar_first(cond, x):
            return torch.where(cond, scalar, x)

        def where_scalar_second(cond, x):
            return torch.where(cond, x, scalar)

        gradcheck(where_scalar_first, (cond, x))
        gradgradcheck(where_scalar_first, (cond, x))

        gradcheck(where_scalar_second, (cond, x))
        gradgradcheck(where_scalar_second, (cond, x))

    @skipCUDAIf(True, """Test is flaky on Linux and Windows, typical error message:
            https://github.com/pytorch/pytorch/issues/34870""")
    def test_ctc_loss(self, device):
        batch_size = 64
        num_labels = 101
        target_length = 15
        gradcheck_input_size = 10

        ZERO_NONE = 0
        ZERO_SOME = 1
        ZERO_ALL = 2

        # input_length, vary_lengths, zero_lengths
        tests = [(150, False, ZERO_NONE),
                 (150, True, ZERO_NONE),
                 (50, True, ZERO_SOME),
                 (50, True, ZERO_ALL)]

        if 'cuda' in device:
            tests += [(50, False, ZERO_NONE),
                      (50, True, ZERO_NONE),
                      (150, True, ZERO_SOME),
                      (150, True, ZERO_ALL)]

        for input_length, vary_lengths, zero_mode in tests:
            targets = torch.randint(1, num_labels, (batch_size, target_length),
                                    device=device, dtype=torch.long)
            x = torch.randn(gradcheck_input_size, dtype=torch.double, device=device, requires_grad=True)
            tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                       device=device)
            input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                              if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
            if zero_mode == ZERO_ALL:
                target_lengths = [0 for _ in range(batch_size)]
            else:
                target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                                   if vary_lengths else target_length) for _ in range(batch_size)]
                if zero_mode == ZERO_SOME:
                    idxes = torch.randint(0, batch_size, (10,))
                    for i in idxes:
                        target_lengths[i] = 0

            def ctc_after_softmax(x):
                x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                          .view(input_length, batch_size, num_labels))
                log_probs = torch.log_softmax(x_full, 2)
                return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            gradcheck(ctc_after_softmax, [x])

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(7600)
    def test_ctc_loss_cudnn(self, device):
        batch_size = 16
        input_length = 30
        num_labels = 101
        target_length = 15
        targets = torch.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=torch.long)
        log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='cuda', dtype=torch.float), 2)
        log_probs.requires_grad_()

        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]
        grad_out = torch.randn(batch_size, device='cuda', dtype=torch.float)
        with torch.backends.cudnn.flags(enabled=False):
            loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)
        loss_cudnn = torch.nn.functional.ctc_loss(log_probs, targets.to('cpu', torch.int32),
                                                  input_lengths, target_lengths, reduction='none')
        self.assertTrue("Cudnn" in str(loss_cudnn.grad_fn))
        grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    def test_leaky_relu_inplace_with_neg_slope(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), -2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.rrelu_(a.clone(), -5.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    def test_leaky_relu_inplace_with_zero_slope(self, device):
        a = torch.tensor([-2., 0., 2.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), 0.0)
        b.backward(torch.ones(3, device=device))
        expected = torch.tensor([0., 0., 1.], device=device)
        self.assertEqual(a.grad, expected)

    @onlyOnCPUAndCUDA
    def test_elu_inplace_with_neg_alpha(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.elu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.celu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    @onlyCUDA
    def test_free_unneeded_tensor(self, device):
        x = torch.randn(2, 3, 10, 10, device=device, requires_grad=True)
        m = torch.randn(1, 3, 1, 1, device=device)

        z = x.sum()
        base_mem = torch.cuda.memory_allocated()
        z = ((x + 2) * m).sum()
        end_mem = torch.cuda.memory_allocated()

        # In the end the memory usage should remain equal, because neither of
        # (x + 2) and ((x + 2) * m) should be kept alive for backward, while the
        # previous allocation of z had the same size as the current one.
        self.assertEqual(base_mem, end_mem)

    @onlyCUDA
    def test_pin_memory(self, device):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        self.assertEqual(x, x.pin_memory())
        self.assertIsNot(x, x.pin_memory())
        self.assertTrue(x.pin_memory().requires_grad)
        gradcheck(lambda x: x.pin_memory(), [x])
        gradgradcheck(lambda x: x.pin_memory(), [x])

    @skipCUDAIfRocm
    @onlyCUDA
    def test_profiler_emit_nvtx(self, device):
        # This test is not intended to ensure correctness of nvtx ranges.
        # That would require something a great deal more complex (you'd have to create a
        # profile in a subprocess, open it, and parse the sql somehow).
        # This test is merely intended to catch if emit_nvtx breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with torch.cuda.profiler.profile():
            with emit_nvtx():
                a.add(1.0)

    @onlyCUDA
    def test_rnn_backward_to_input_but_not_parameters(self, device):
        # this checks whether it is possible to not require
        # weight parameters, but require inputs, see #7722
        l = torch.nn.LSTM(2, 3).to(device)
        for p in l.parameters():
            p.requires_grad = False
        s = torch.randn(1, 1, 2, requires_grad=True, device=device)
        out, _ = l(s)
        out.sum().backward()
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    @onlyCUDA
    def test_lstmcell_backward_only_one_output_grad(self, device):
        # checks that undefined gradients doen't hamper the backward
        # see #11872
        l = torch.nn.LSTMCell(2, 3).to(device).double()
        s = torch.randn(1, 2, device=device, dtype=torch.double, requires_grad=True)
        for i in range(2):
            out = l(s)[i]
            out.sum().backward()
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    def _test_rnn_mod(self, mod, inp):
        def flatten_out(mod, inp):
            out = mod(inp)
            return tuple([t if isinstance(t, torch.Tensor) else tt for t in out for tt in t])
        gradcheckfunc = partial(flatten_out, mod)
        with torch.backends.cudnn.flags(enabled=False):
            gradcheck(gradcheckfunc, inp, check_batched_grad=False)
            gradgradcheck(gradcheckfunc, inp, check_batched_grad=False)

        if inp.is_cuda and not TEST_WITH_ROCM:
            # Assert that we have good error message around unsupported CuDNN double backward
            # NB: we trigger double backward using .backward() instead of autograd.grad due to
            # https://github.com/pytorch/pytorch/issues/37874
            with torch.backends.cudnn.flags(enabled=True):
                result = gradcheckfunc(inp)
                result[0].sum().backward(create_graph=True)
                grad0 = next(mod.parameters()).grad
                with self.assertRaisesRegex(RuntimeError,
                                            "please disable the CuDNN backend temporarily"):
                    grad0.sum().backward()

                # Here we avoid the backward(create_graph=True) memory leak
                # described in https://github.com/pytorch/pytorch/issues/7343
                for param in mod.parameters():
                    param.grad = None
                inp.grad = None

    @skipMeta  # LSTM cell reuses output which was resized
    def test_LSTM_grad_and_gradgrad(self, device):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=torch.float64, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.LSTM(hsize, hsize, bias=bias).to(device).to(torch.float64)
            self._test_rnn_mod(mod, inp)

    @skipMeta  # GRU cell reuses output which was resized
    def test_GRU_grad_and_gradgrad(self, device):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=torch.float64, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.GRU(hsize, hsize, bias=bias).to(device).to(torch.float64)
            self._test_rnn_mod(mod, inp)

    def test_copysign_subgradient(self, device):
        # Input is 0.0
        x = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Input is -0.0
        x = torch.tensor([-0.0, -0.0, -0.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is 0.0
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is -0.0
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-0.0, -0.0, -0.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

    @deviceCountAtLeast(1)
    def test_grad_assignment(self, devices):
        x = torch.randn(5, 5, device=devices[0])

        # Tests that the wrong shape raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2, device=devices[0])

        # Tests that the wrong dtype raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(5, 5, dtype=torch.long, device=devices[0])

        # Tests that self-assignment raises
        with self.assertRaises(RuntimeError):
            x.grad = x

        # Tests device -> cpu grad assignment raises
        if self.device_type != 'cpu':
            with self.assertRaises(RuntimeError):
                t_cpu = torch.rand(5, 5)
                t_cpu.grad = torch.randn(5, 5, device=devices[0])

        # Tests half type on CUDA
        if self.device_type == 'cuda':
            x = x.to(dtype=torch.half, device=devices[0])
            x.grad = torch.zeros_like(x)

        # Tests cross-device assignment raises
        if len(devices) > 1:
            x = torch.randn(5, 5, device=devices[0])
            with self.assertRaises(RuntimeError):
                x.grad = torch.randn(5, 5, device=devices[1])

    @deviceCountAtLeast(1)
    @dtypes(torch.float, torch.double)
    def test_requires_grad_factory(self, devices, dtype):
        fns = [torch.ones_like, torch.testing.randn_like]
        x = torch.randn(2, 3, dtype=dtype, device=devices[0])

        for fn in fns:
            for requires_grad in [True, False]:
                output = fn(x, dtype=dtype, device=devices[0], requires_grad=requires_grad)
                self.assertEqual(requires_grad, output.requires_grad)
                self.assertIs(dtype, output.dtype)
                self.assertEqual(devices[0], str(x.device))

    @deviceCountAtLeast(2)
    def test_unused_output_device(self, devices):
        from torch.nn.parallel._functions import Broadcast
        x = torch.randn(5, 5, dtype=torch.float, device=devices[0], requires_grad=True)
        outputs = Broadcast.apply(list(range(len(devices))), x)
        y = outputs[-1] * 2
        y.sum().backward()
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(x.grad, torch.ones(5, 5) * 2)

    @deviceCountAtLeast(2)
    def test_backward_device(self, devices):
        # check that current device matches the variable's device
        device = [None]

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                device[0] = grad_output.device
                return grad_output.clone()

        v = torch.randn(1, device=devices[1], requires_grad=True)
        Identity.apply(v).backward()
        self.assertEqual(str(device[0]), devices[1])

    @deviceCountAtLeast(2)
    def test_inputbuffer_add_multidevice(self, devices):
        input = torch.randn(1, device=devices[0], requires_grad=True)
        output = input.to(device=devices[1]) + input.to(device=devices[1])
        output.backward()

    @onlyCPU
    def test_copy_(self, device):
        # At the time of writing this test, copy_ is not generated from native_functions.yaml
        # there was a bug that bfloat16 was not recognized as floating.
        x = torch.randn(10, device=device, requires_grad=True)
        floating_dt = [dt for dt in torch.testing.get_all_dtypes() if dt.is_floating_point]
        for dt in floating_dt:
            y = torch.empty(10, device=device, dtype=dt)
            y.copy_(x)
            self.assertTrue(y.requires_grad)
            z = x.to(torch.bfloat16)
            self.assertTrue(z.requires_grad)

    @onlyCUDA
    def test_simple_reentrant_cross_device(self, device):
        class ReentrantFunc(Function):
            _cpu_mode = True

            @staticmethod
            def forward(ctx, x):
                return x * (x + 2)

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    if ReentrantFunc._cpu_mode:
                        new_param = torch.randn(2, 2, requires_grad=True)
                        (new_param ** 2).sum().backward()
                    else:
                        new_param = torch.randn(2, 2, device=device, requires_grad=True)
                        (new_param ** 2).sum().backward()
                return grad_output

        # Reentrant starts on GPU thread, finishs on GPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on CPU thread, finishs on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        # set ReentrantFunc node to GPU to emit tasks to GPU queue
        ReentrantFunc._cpu_mode = False
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on GPU thread, finishs on CPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        # set ReentrantFunc node to CPU to emit tasks to CPU queue
        ReentrantFunc._cpu_mode = True
        out = ReentrantFunc.apply(x)
        out.sum().backward()

    @onlyCUDA
    def test_cross_device_reentrant_autograd(self, device):
        # Output on gpu so that this task will be associated with the gpu thread
        def fn_on_gpu(inp):
            # Artificially increase the priority of the next op to make sure it runs
            # as soon as we reach it before the ops of branch1.
            dummy = inp * 2 * 2 * 2 * 2
            return inp.to(device=device)

        def parent_on_cpu(inp):
            # Slow branch of ops on gpu so that the work queue for the gpu thread
            # won't empty too quickly. They also have smaller priorities than the
            # ones created by fn_on_gpu
            branch1 = inp.to(device=device)
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            # Perform checkpoint on cpu tensors. So the last op performed in the reentrant
            # autograd is an AccumulateGrad that runs on the cpu thread for the gpu thread.
            # So the cpu thread will notify the gpu thread with an empty NodeTask.
            branch2 = checkpoint(fn_on_gpu, inp)
            out = branch2 + branch1
            return out

        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)
        # This will segfault if the empty NodeTask is not handled properly in the
        # gpu thread ReadyQueue
        out.sum().backward()

    def test_inplace_view_backprop_base(self, device):
        # modify view and back-prop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v1.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [1, 1]])

    def test_inplace_view_backprop_view_of_view(self, device):
        # modify view and backprop through view-of-view
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = x.narrow(0, 0, 1)
        v1.mul_(2)
        v2.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

    def test_inplace_view_of_view(self, device):
        # modify view-of-view and backprop through base
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1]])

    def test_inplace_view_then_no_grad(self, device):
        # Perform an in-place operation on a view of a non-leaf variable.
        a = torch.ones(3, 1, dtype=torch.double, device=device, requires_grad=True)
        b = a * 2
        c = b.view_as(b)
        c[0][0] = 3

        # Force a graph update with grad disabled.
        with torch.no_grad():
            c.grad_fn

        c.sum().backward()

    def test_inplace_view_gradcheck(self, device):
        # gradcheck modifications to views
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            x = root.clone()
            x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
            x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_view_multiple_outputs(self, device):
        root = torch.arange(9., dtype=torch.double).reshape(3, 3).requires_grad_()
        x = root.clone()
        v1 = x.unbind()
        with self.assertRaises(RuntimeError):
            v1[0].mul_(2)

    def test_inplace_view_of_multiple_output_view(self, device):
        a = torch.rand(10, dtype=torch.double, device=device, requires_grad=True).clone()
        b = a.unbind(0)
        c = b[0].view_as(b[0])
        with self.assertRaises(RuntimeError):
            c.mul_(2)

    def test_inplace_multiple_output_view_of_view(self, device):
        a = torch.rand(10, dtype=torch.double, device=device, requires_grad=True).clone()
        b = a.view_as(a)
        c = b.unbind(0)
        with self.assertRaises(RuntimeError):
            c[0].mul_(2)

    def test_inplace_view_makes_base_require_grad(self, device):
        # in-place modification to view makes base require grad
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=False)
        b = torch.randn(4, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            x = root.clone()
            self.assertFalse(x.requires_grad)
            x.narrow(1, 2, 2).mul_(b)
            self.assertTrue(x.requires_grad)
            return x

        gradcheck(func, [a, b], raise_exception=True)
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_view_backprop_view(self, device):
        # modify view and backprop through view
        a = torch.tensor([2., 5.], device=device, requires_grad=False)
        b = torch.tensor([3.], device=device, requires_grad=True)
        res = a.narrow(0, 1, 1).mul_(b)
        res.sum().backward()
        self.assertEqual(b.grad.tolist(), [5])
        self.assertIsNone(a.grad)

    def test_inplace_view_modify_base(self, device):
        # Test that an in-place operation on a base that forced it to require
        # grad also forces any previous views to require grad and backprop
        # correctly
        r = torch.ones(1, dtype=torch.double, device=device, requires_grad=True)

        def fn(r):
            x = torch.ones(5, dtype=torch.double, device=device)
            v = x.select(0, 1)
            self.assertFalse(v.requires_grad)
            self.assertIsNone(v.grad_fn)
            x.add_(r)  # v is now dependent on r due to the in-place op on x
            self.assertTrue(v.requires_grad)
            return v

        gradcheck(fn, [r])
        gradgradcheck(fn, [r])

    def test_inplace_view_python(self, device):
        # in-place modifications of Python-autograd created view
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

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
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_view_non_contig(self, device):
        root = torch.ones(2, 3, 2, device=device).select(2, 1).t().requires_grad_(True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = v1.narrow(1, 1, 1)
        v2.mul_(2)
        x.sum().backward()
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1], [1, 1]])

    def test_inplace_view_multi_output_unsafe(self, device):
        for f in [lambda t: t.unsafe_split(1),
                  lambda t: t.unsafe_split_with_sizes((1, 1, 1)),
                  lambda t: t.unsafe_chunk(3)]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            s1.mul_(s2)
            s1.sum().backward()

    def test_inplace_view_multi_output_safe(self, device):
        for f in [lambda t: t.split(1),
                  lambda t: t.split_with_sizes((1, 1, 1)),
                  lambda t: t.chunk(3)]:
            a = torch.randn(3, 3, device=device, requires_grad=True)
            b = a + a
            s1, s2, s3 = f(b)
            error_msg = 'This view is an output of a function that returns multiple views.'
            with self.assertRaisesRegex(RuntimeError, error_msg):
                s1.mul_(s2)

    def test_mv_grad_stride_0(self, device):
        # Reference: https://github.com/pytorch/pytorch/issues/38315
        mat = torch.randn(2, 2, dtype=torch.double, device=device)
        vec = torch.randn(1, dtype=torch.double, device=device).requires_grad_(True)

        def fn(vec):
            # Expand inside the function to make sure the input to
            # gradcheck does not have overlapping memory
            vec = vec.expand(2)
            return (mat @ vec).sum()

        gradcheck(fn, (vec))
        gradgradcheck(fn, (vec))

    @onlyCUDA
    def test_gradcheck_input_output_different_device(self, device):
        x = torch.ones((1,), dtype=torch.double, device="cuda", requires_grad=True)
        gradcheck(lambda x: x.to("cpu"), (x,))

        x = torch.ones((1,), dtype=torch.double, device="cpu", requires_grad=True)
        gradcheck(lambda x: x.to("cuda"), (x,))

    def test_logcumsumexp_large_value(self, device):
        a = torch.rand(4, 4, 4, dtype=torch.double, requires_grad=True)
        with torch.no_grad():
            # Large Number
            a[0] = 10000

        gradcheck(lambda x: x.logcumsumexp(0), a)
        gradgradcheck(lambda x: x.logcumsumexp(0), a)

        gradcheck(lambda x: x.logcumsumexp(1), a)
        gradgradcheck(lambda x: x.logcumsumexp(1), a)

        gradcheck(lambda x: x.logcumsumexp(2), a)
        gradgradcheck(lambda x: x.logcumsumexp(2), a)

    def test_strided_leaf_grad_layout(self, device):
        # (1) If leaf is non-overlapping and dense, grad's layout should match its leaf.
        for fmt_a in (torch.contiguous_format, torch.channels_last):
            for fmt_b in (torch.contiguous_format, torch.channels_last):
                a = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_a)
                b = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_b)
                a.requires_grad_()
                b.requires_grad_()
                # checks (1) for broadcasted gradients
                a.sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                b.sum().backward()
                self.assertEqual(b.grad.stride(), b.stride())
                # checks (1) for non-broadcasted gradients
                a.grad = None
                b.grad = None
                (a * b).sum().backward()
                self.assertEqual(a.grad.stride(), a.stride())
                self.assertEqual(b.grad.stride(), b.stride())

        # (2) If leaf isn't dense, checks that grads are rowmajor contiguous.
        c = torch.empty_strided((2, 2), (4, 2), device=device).copy_(torch.rand((2, 2), device=device))
        c.requires_grad_()
        d = torch.rand((2, 2), device=device)
        # checks (2) for broadcasted gradients
        c.sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))
        # checks (2) for non-broadcasted gradients
        c.grad = None
        (c * d).sum().backward()
        self.assertEqual(c.grad.stride(), (2, 1))

    def _test_atleast(self, device, torch_fn):
        # 0-dim
        s = torch.tensor(0.5, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), s)
        gradgradcheck(lambda x: torch_fn(x), s)

        # 1-dim
        a = torch.rand(4, dtype=torch.double, requires_grad=True)

        gradcheck(lambda x: torch_fn(x), a)
        gradgradcheck(lambda x: torch_fn(x), a)

        # 2,3,4-dim
        b = torch.rand(4, 3, dtype=torch.double, requires_grad=True)
        c = torch.rand(4, 3, 2, dtype=torch.double, requires_grad=True)
        d = torch.rand(4, 3, 2, 1, dtype=torch.double, requires_grad=True)

        input_tuple = (s, a, b, c, d)
        gradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)
        gradgradcheck(lambda s, w, x, y, z: torch_fn(s, w, x, y, z), input_tuple)

    def test_atleast(self, device):
        self._test_atleast(device, torch.atleast_1d)
        self._test_atleast(device, torch.atleast_2d)
        self._test_atleast(device, torch.atleast_3d)

    def test_xlogy(self, device):

        def _tensor_tensor_helper(x, y):
            gradcheck(lambda x, y: torch.xlogy(x, y), (x, y))
            gradgradcheck(lambda x, y: torch.xlogy(x, y), (x, y))

            with torch.no_grad():
                x = x.clone()
                x[torch.rand_like(x) > 0.5] = 0

            gradcheck(lambda y: torch.xlogy(x, y), (y))
            gradgradcheck(lambda y: torch.xlogy(x, y), (y))

        shapes = ((4,), (1, 4), (1, 1, 4), (1, 1, 1, 4))

        # For broadcastible shapes and scalar.
        for x_shape, y_shape in permutations(shapes, 2):
            x = torch.rand(*x_shape, dtype=torch.double, device=device, requires_grad=True)
            y = torch.rand(*y_shape, dtype=torch.double, device=device, requires_grad=True)

            _tensor_tensor_helper(x, y)
            _tensor_tensor_helper(y, x)

            gradcheck(lambda y: torch.xlogy(0, y), (y))
            gradgradcheck(lambda y: torch.xlogy(0, y), (y))

            gradcheck(lambda y: torch.xlogy(2, y), (y))
            gradgradcheck(lambda y: torch.xlogy(2, y), (y))
            gradcheck(lambda y: torch.xlogy(y, 2), (y))
            gradgradcheck(lambda y: torch.xlogy(y, 2), (y))

        # Different shape
        x = torch.rand(2, 3, 4, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.rand(4, 5, dtype=torch.double, device=device, requires_grad=True)
        _tensor_tensor_helper(x, y)
        _tensor_tensor_helper(y, x)
        _tensor_tensor_helper(x, x)
        _tensor_tensor_helper(y, y)

        # Same shape
        x = torch.rand(4, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.rand(4, 5, dtype=torch.double, device=device, requires_grad=True)
        _tensor_tensor_helper(x, y)
        _tensor_tensor_helper(y, x)
        _tensor_tensor_helper(x, x)
        _tensor_tensor_helper(y, y)


class TestAutogradInferenceMode(TestCase):
    def _is_inference_tensor(self, tensor):
        try:
            err_msg = "Inference tensors do not track version counter"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                tensor._version
            return True
        except AssertionError as e:
            return False

    def test_inference_mode_context_manager(self):
        self.assertFalse(torch.is_inference_mode_enabled())
        with torch.inference_mode():
            self.assertTrue(torch.is_inference_mode_enabled())
            with torch.inference_mode(False):
                self.assertFalse(torch.is_inference_mode_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_inference_mode_decorator(self):
        @torch.inference_mode()
        def func(x):
            self.assertTrue(torch.is_inference_mode_enabled())
            return x * x

        for requires_grad in (True, False):
            c = torch.ones(1, 2, 3, requires_grad=requires_grad)
            d = func(c)
            self.assertTrue(self._is_inference_tensor(d))
            self.assertFalse(d.requires_grad)

    def test_inference_mode_tensor_creation(self):
        with torch.inference_mode():
            # new tensors created through constructors are inference tensors
            c = torch.ones(1, 2, 3)
            self.assertFalse(c.requires_grad)
            self.assertTrue(self._is_inference_tensor(c))

            # requires_grad doesn't change inference tensor behavior in InferenceMode
            tmp = torch.ones(1, 2, 3, requires_grad=True)
            self.assertTrue(tmp.requires_grad)
            self.assertTrue(self._is_inference_tensor(tmp))

            tmp = torch.ones(1, 2, 3).requires_grad_(False)
            self.assertFalse(tmp.requires_grad)
            self.assertTrue(self._is_inference_tensor(tmp))

    def test_inference_mode_existing_autograd_session(self):
        s = torch.ones(1, 2, 3, requires_grad=True)
        a = s.clone()

        # `a` gets saved outside of inference mode
        out = a * a
        with torch.inference_mode():
            a.add_(2)

        self.assertFalse(self._is_inference_tensor(a))
        # tensors created outside of inference mode aren't
        # inference tensors, so they will still have their
        # version counters tracked
        err_msg = ("one of the variables needed for gradient computation has been "
                   "modified by an inplace operation")
        with self.assertRaisesRegex(RuntimeError, err_msg):
            out.backward(torch.ones_like(out))

    def test_inference_mode_inf_tensor_in_inf_mode_functional_op(self):
        def functional_op(x):
            return x * x

        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # performing a non-view operation produces a inference tensor
                # that does not require grad
                func_out = functional_op(c)
                self.assertTrue(self._is_inference_tensor(func_out))
                self.assertFalse(func_out.requires_grad)

    def test_inference_mode_inf_tensor_in_inf_mode_inplace_op(self):
        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # after perform inplace operation, tensor is still
                # an inference tensor
                c.add_(2)
                self.assertTrue(self._is_inference_tensor(c))
                self.assertEqual(c.requires_grad, requires_grad)

    def test_inference_mode_inf_tensor_in_inf_mode_view_op(self):
        with torch.inference_mode():
            for requires_grad in (True, False):
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # perform view operation produces inference tensor
                # that does not require grad
                view_out = c.view(-1)
                self.assertTrue(self._is_inference_tensor(view_out))
                self.assertFalse(view_out.requires_grad)

    def test_inference_mode_inf_tensor_in_normal_mode_functional_op(self):
        def functional_op(x):
            return x * x

        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

        func_out = functional_op(c)
        self.assertFalse(self._is_inference_tensor(func_out))
        self.assertFalse(func_out.requires_grad)
        self.assertTrue(func_out.is_leaf)

    def test_inference_mode_inf_tensor_in_normal_mode_inplace_op(self):
        for requires_grad in (False, True):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            if requires_grad:
                # leaf variable that requires grad is being used in an inplace
                # operation when requires_grad=True
                pass
            else:
                err_msg = "Inplace update to inference tensor outside InferenceMode"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    c.add_(2)

    def test_inference_mode_inf_tensor_in_normal_mode_view_op(self):
        for requires_grad in (True, False):
            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            out = c.view(-1)
            self.assertTrue(self._is_inference_tensor(out))
            self.assertFalse(out.requires_grad)
            self.assertFalse(out._is_view())
            self.assertTrue(out.is_leaf)

    def test_normal_tensor_inplace_output_in_inference_mode(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                a.add_(2)
                self.assertFalse(self._is_inference_tensor(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace
                a.add_(2)
                self.assertFalse(self._is_inference_tensor(a))
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace -> inplace -> view
                view_out = a.view(-1)
                self.assertFalse(self._is_inference_tensor(view_out))
                self.assertEqual(view_out.requires_grad, requires_grad)

    def test_normal_tensor_inplace_output_in_normal_mode(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                a.add_(2)
                self.assertFalse(self._is_inference_tensor(a))
                self.assertEqual(a.requires_grad, requires_grad)

            a.add_(2)
            self.assertFalse(self._is_inference_tensor(a))
            self.assertEqual(a.requires_grad, requires_grad)

            # inplace -> inplace
            a.add_(2)
            self.assertFalse(self._is_inference_tensor(a))
            self.assertEqual(a.requires_grad, requires_grad)

            # inplace -> inplace -> view
            view_out = a.view(-1)
            self.assertFalse(self._is_inference_tensor(view_out))
            self.assertEqual(view_out.requires_grad, requires_grad)

    def test_normal_tensor_view_output_in_inference_mode(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(self._is_inference_tensor(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())

                # view -> view
                tmp = out.view(-1)
                self.assertFalse(self._is_inference_tensor(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                self.assertTrue(tmp._is_view())
                self.assertTrue(tmp.is_leaf)

                # view -> view -> inplace
                self.assertTrue(torch.is_inference_mode_enabled())
                tmp.add_(2)
                self.assertFalse(self._is_inference_tensor(tmp))
                self.assertEqual(tmp.requires_grad, requires_grad)
                # Accessing is_leaf in python tries to update grad_fn and raises:
                # A view was created in inference mode and its base or
                # another view of its base has been modified inplace in normal mode
                # tmp.is_leaf
                self.assertEqual(a._version, tmp._version)

    def test_normal_tensor_view_output_in_normal_mode(self):
        def functional_op(x):
            return x * x

        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                out = a.view(-1)
                self.assertFalse(self._is_inference_tensor(out))
                self.assertEqual(out.requires_grad, requires_grad)
                self.assertTrue(out._is_view())
                self.assertTrue(out.is_leaf)

            tmp = functional_op(out)
            self.assertFalse(self._is_inference_tensor(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

            if requires_grad:
                err_msg = "A view was created in inference mode and is being modified inplace"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    out.add_(2)
                pass
            else:
                out.add_(2)

            tmp = out.view(2, 3)
            self.assertFalse(self._is_inference_tensor(tmp))
            self.assertEqual(tmp.requires_grad, requires_grad)

    def test_mix_inference_and_normal_tensor_functional_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            # add is safe since it doesn't save any variable for backward
            out = c.add(s)
            self.assertFalse(self._is_inference_tensor(out))
            self.assertEqual(out.requires_grad, requires_grad)
            if requires_grad:
                # leaf inference tensor with requires_grad=True can still have gradient
                out.backward(torch.ones_like(out))
                self.assertEqual(c.grad, torch.ones_like(c))

            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    c * s

                # inference tensor in TensorList input
                inputs = [s, c]
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.stack(inputs)


    def test_mix_inference_and_normal_tensor_inplace_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            self.assertTrue(self._is_inference_tensor(c))
            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mul_(c)

                # inference tensor in TensorList input
                err_msg = ("out=... arguments don't support automatic differentiation, "
                           "but one of the arguments requires grad")
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)
            else:
                a.mul_(c)
                err_msg = "Inplace update to inference tensor outside InferenceMode is not allowed"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)

    def test_mix_inference_and_normal_tensor_view_op(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            with torch.inference_mode():
                c = torch.ones(1, 2, 3)

            # view_as is a composite op which calls view with only one
            # tensor argument. So there isn't a mixed inference and normal
            # tensor inputs for view ops
            tmp1 = c.view_as(s)
            self.assertTrue(self._is_inference_tensor(tmp1))
            self.assertFalse(tmp1.requires_grad)

            # this is fine since its equivalent as s.view(c.sizes()) which
            # isn't a mixed input scenario
            tmp2 = s.view_as(c)
            self.assertFalse(self._is_inference_tensor(tmp2))
            self.assertEqual(tmp2.requires_grad, requires_grad)

    def test_inference_mode_handle_direct_view_on_rebase(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                view_out = a.view(-1)

            if requires_grad:
                err_msg = "A view was created in inference mode and is being modified inplace"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    view_out.add_(2)
                pass
            else:
                view_out.add_(2)

    def test_inference_mode_handle_indirect_view_on_rebase(self):
        for requires_grad in (True, False):
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            a = s.clone()

            with torch.inference_mode():
                view_out = a.view(-1)

            a.add_(2)
            if requires_grad:
                err_msg = "A view was created in inference mode and its base or another view "
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    view_out.grad_fn
                pass
            else:
                view_out.grad_fn

class TestMultithreadAutograd(TestCase):
    def _run_py_multithread_fn(self, fn, args=(), num_threads=10, kwargs=None):
        threads = []
        for _ in range(num_threads):
            p = threading.Thread(target=fn, args=(args))
            p.start()
            threads.append(p)

        for p in threads:
            p.join()

    def test_simple_backward(self):
        # simple multithreaded backward that create threads in the beginning of training
        # and everything else is training separately, i.e. inputs, operations, etc.
        def train_fn():
            x = torch.ones(5, 5, requires_grad=True)
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()
            self.assertEqual(x.grad, x + 3.5)

        self._run_py_multithread_fn(train_fn)

    def test_simple_backward_same_input(self):
        # simple multithreaded backward with only shared inputs (i.e. This is common
        # for things like Hogwild multithreaded training with multiple CPU threads)
        def train_fn_backward(x):
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()

        x = torch.ones(5, 5, requires_grad=True)
        self._run_py_multithread_fn(train_fn_backward, (x,))
        # Since we are calling backward from multiple threads
        # and all threads share the same input, when we do backward
        # concurrently, different backwards will all accumulate to
        # the same .grad for each input, and the gradients should
        # be equal to num_threads * gradient
        self.assertEqual(x.grad, 10 * (x + 3.5))

        def train_fn_grad(x):
            y = (x + 3) * (x + 4) * 0.5
            grads = torch.autograd.grad(y.sum(), x)
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0], x + 3.5)

        # since we use functional grad() api, gradients will not
        # be accumulate to the same place and should be the same
        self._run_py_multithread_fn(train_fn_grad, (x,))

    def test_python_thread_in_middle(self):
        # User might write a network that starts on one CPU thread, then runs its second half
        # concurrently with other threads (either via python threading or fork/join calls),
        # then calls backward()/grad() on BOTH threads, like a Y pattern from input at the
        # bottom to output at the top. This way part of the GraphTask is being shared across
        # different threads and we need to ensure user specify retain_graph=True, otherwise
        # error out with the correct error message

        # Case 1: multiple backward with python threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        success_vs_raises = [0, 0]

        def train_fn_no_retain_graph(x):
            y = x + x ** 2
            try:
                y.sum().backward()
                success_vs_raises[0] += 1
            except RuntimeError as error:
                success_vs_raises[1] += 1
                self.assertRegex(str(error), "Specify retain_graph=True")

        x_no_retain = torch.ones(5, 5, requires_grad=True)
        y_no_retain = x_no_retain + x_no_retain ** 2
        self._run_py_multithread_fn(train_fn_no_retain_graph, (y_no_retain,), num_threads=5)
        # at least one thread will be success in this case, all other threads should raise
        # with the error that throw to user to recommend them specify retain_graph=True
        self.assertTrue(success_vs_raises[0] >= 1)

        # multiple backward with python threads, no error with retain_graph=True
        def train_fn_retain_graph(x):
            y = x + x ** 2
            y.sum().backward(retain_graph=True)

        x_retain = torch.ones(5, 5, requires_grad=True)
        y_retain = x_retain + x_retain ** 2
        self._run_py_multithread_fn(train_fn_retain_graph, (y_retain,), num_threads=5)
        # result should equal to num_thread * gradients
        self.assertEqual(x_retain.grad, 5 * (4 * x_retain ** 3 + 6 * (x_retain ** 2) + 4 * x_retain + 1))

    def test_fork_join_in_middle(self):
        # multiple backward with jit threads (fork/join primitive)
        # similar to test_python_thread_in_middle, we test with retain_graph=False/True

        # Case 1: multiple grad() calls with jit threads, retain_graph=False
        # should throw error in some threads with no retain_graph.
        @torch.jit.script
        def train_fn_jit_no_retain(middle, orig_x):
            y = middle + middle ** 2
            return torch.autograd.grad([y.sum()], [orig_x])

        @torch.jit.script
        def train_fn_fork_join_calls_no_retain(x):
            y_no_retain = (x + 3) * (x + 4) * 0.5

            fut = torch.jit._fork(train_fn_jit_no_retain, y_no_retain, x)
            grad_hat = train_fn_jit_no_retain(y_no_retain, x)
            grad = torch.jit._wait(fut)
            return grad, grad_hat

        try:
            train_fn_fork_join_calls_no_retain(torch.randn(5, 5, requires_grad=True))
        except RuntimeError as error:
            self.assertRegex(str(error), "Specify retain_graph=True")

        # Case 2: no error with retain_graph=True
        @torch.jit.script
        def train_fn_jit_retain(middle, orig_x):
            y = middle + middle ** 2
            return torch.autograd.grad([y.sum()], [orig_x], retain_graph=True)

        @torch.jit.script
        def train_fn_fork_join_calls_retain(x):
            y_retain = (x + 3) * (x + 4) * 0.5
            fut1 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            fut2 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            grad = train_fn_jit_retain(y_retain, x)
            grad1 = torch.jit._wait(fut1)
            grad2 = torch.jit._wait(fut2)
            return grad, grad1, grad2

        grad, grad1, grad2 = train_fn_fork_join_calls_retain(torch.randn(5, 5, requires_grad=True))
        self.assertEqual(grad, grad1)
        self.assertEqual(grad, grad2)

    def test_preserve_backtrace(self):
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, *grad):
                raise ValueError("something")

        t = torch.rand(10, requires_grad=True)
        try:
            Foo.apply(t).sum().backward()
        except Exception:
            import traceback
            tb = sys.exc_info()[2]
            tb_str = "\n".join(traceback.format_tb(tb))
            self.assertTrue('raise ValueError("something")' in tb_str)

    # TODO(@anjali411): add an OpInfo based test for torch.cat
    # Issue: https://github.com/pytorch/pytorch/issues/51627
    def test_cat_r_to_c(self):
        inp_c = torch.rand(3, 2, dtype=torch.cdouble, requires_grad=True)
        inp_r = torch.randn(3, 2, dtype=torch.double, requires_grad=True)

        def fn(x1, x2):
            return torch.cat((x1, x2), dim=-1)

        torch.autograd.gradcheck(fn, [inp_r, inp_c])
        torch.autograd.gradcheck(fn, [inp_c, inp_r])

for test in method_tests():
    add_test(*test)


# e.g., TestAutogradDeviceTypeCPU and TestAutogradDeviceTypeCUDA
instantiate_device_type_tests(
    TestAutogradDeviceType,
    globals(),
    except_for=None
)

if __name__ == '__main__':
    run_tests()
