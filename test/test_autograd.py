import contextlib
import gc
import sys
import math
import torch
import unittest
from copy import deepcopy
from collections import OrderedDict

from common import make_jacobian, TestCase, iter_tensors, \
    get_numerical_jacobian, run_tests
from torch.autograd._functions import *
from torch.autograd import Variable, Function

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

PRECISION = 1e-4


def iter_gradients(x):
    if isinstance(x, Variable):
        if x.requires_grad:
            yield x.grad.data
    else:
        for elem in x:
            for result in iter_gradients(elem):
                yield result


def zero_gradients(i):
    for t in iter_gradients(i):
        t.zero_()


def get_analytical_jacobian(input, output):
    jacobian = make_jacobian(input, output.numel())
    grad_output = output.data.clone().zero_()
    flat_grad_output = grad_output.view(-1)

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        zero_gradients(input)
        output.backward(grad_output, retain_variables=True)
        for jacobian_x, d_x in zip(jacobian, iter_gradients(input)):
            jacobian_x[:, i] = d_x

    return jacobian


@contextlib.contextmanager
def backward_engine(engine):
    _prev_engine = Variable._execution_engine
    Variable._execution_engine = engine()
    try:
        yield
    finally:
        Variable._execution_engine = _prev_engine


class TestAutograd(TestCase):

    def test_hooks(self):
        x = Variable(torch.ones(5, 5), requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        counter = [0]

        def bw_hook(inc, grad):
            self.assertIsInstance(grad, Variable)
            counter[0] += inc

        z = x ** 2 + x * 2 + x * y + y
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(counter[0], 1)

        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(counter[0], 4)

        test2.remove()
        z.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            return grad.mul(2)

        test.remove()
        z.register_hook(bw_hook_modify)
        y.grad.data.zero_()
        z.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(y.grad.data, (x.data + 1) * 2)

        y.register_hook(bw_hook_modify)
        y.grad.data.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad.data, (x.data + 1) * 4)

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
        fn._backward_hooks = OrderedDict()
        was_called = [False]

        def hook(grad_input, grad_output):
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertIsNotNone(grad_input[0])
            self.assertIsNone(grad_input[1])
            self.assertIsNotNone(grad_output[0])
            self.assertIsNotNone(grad_output[1])
            was_called[0] = True
        fn._backward_hooks[id(hook)] = hook

        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5))
        sum(fn(x, y)).sum().backward()
        self.assertTrue(was_called[0])

    def _test_backward(self):
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

    def test_backward(self):
        self._test_backward()

    @unittest.skip("BasicEngine is out of date")
    def test_backward_basic_engine(self):
        with backward_engine(torch.autograd.engine.BasicEngine):
            self._test_backward()

    def test_multi_backward(self):
        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5), requires_grad=True)

        q = Variable(torch.randn(5, 5), requires_grad=True)

        a = Variable(torch.randn(5, 5), requires_grad=True)
        b = Variable(torch.randn(5, 5), requires_grad=True)

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

    def test_multi_backward_stochastic(self):
        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5), requires_grad=True)

        z = x + y
        q = torch.normal(x)
        q.reinforce(torch.randn(5, 5))

        torch.autograd.backward([z, q], [torch.ones(5, 5), None])

    def test_multi_backward_no_grad(self):
        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5), requires_grad=False)

        z = x + y
        q = y * 2

        torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])
        self.assertEqual(x.grad.data, torch.ones(5, 5))

    def test_volatile(self):
        x = Variable(torch.ones(5, 5), requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, volatile=True)

        z = x ** 2
        self.assertFalse(z.volatile)
        self.assertTrue(z.requires_grad)
        self.assertIsNotNone(z.creator)
        z.backward(torch.ones(5, 5))
        self.assertEqual(x.grad.data, torch.ones(5, 5) * 2)

        w = z + y
        self.assertTrue(w.volatile)
        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.creator)

    def test_indexing(self):
        x = torch.range(1, 16).resize_(4, 4)
        y = Variable(x)
        self.assertEqual(x[1], y[1].data)
        self.assertEqual(x[1, 1], y[1, 1].data[0])
        self.assertEqual(x[1:], y[1:].data)
        self.assertEqual(x[:2], y[:2].data)
        self.assertEqual(x[:2, 2], y[:2, 2].data)
        self.assertEqual(x[1:2, 2], y[1:2, 2].data)
        self.assertEqual(x[1, 2:], y[1, 2:].data)

    def test_requires_grad(self):
        x = Variable(torch.randn(5, 5))
        y = Variable(torch.randn(5, 5))
        z = Variable(torch.randn(5, 5), requires_grad=True)
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

    def test_inplace(self):
        x = Variable(torch.ones(5, 5), requires_grad=True)
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        z = x * y
        q = z + y
        w = z * y
        z.add_(2)
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5), retain_variables=True)
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5), retain_variables=True)
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5), retain_variables=True)
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
        r.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(x.grad.data, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5), retain_variables=True)
        self.assertEqual(x.grad.data, torch.Tensor(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        leaf = Variable(torch.ones(5, 5), requires_grad=True)
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

    def test_shared_storage(self):
        x = Variable(torch.ones(5, 5))
        y = x.t()
        z = x[1]
        self.assertRaises(RuntimeError, lambda: x.add_(2))
        self.assertRaises(RuntimeError, lambda: y.add_(2))
        self.assertRaises(RuntimeError, lambda: z.add_(2))

    def _test_setitem(self, size, index):
        x = Variable(torch.ones(*size), requires_grad=True)
        y = x + 2
        y_version = y._version
        y[index] = 2
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad = torch.ones(*size)
        if isinstance(index, Variable):
            index = index.data
        expected_grad[index] = 0
        self.assertEqual(x.grad.data, expected_grad)

    def _test_setitem_tensor(self, size, index):
        x = Variable(torch.ones(*size), requires_grad=True)
        y = x + 2
        y_version = y._version
        value = Variable(torch.Tensor(x[index].size()).fill_(7), requires_grad=True)
        y[index] = value
        self.assertNotEqual(y._version, y_version)
        y.backward(torch.ones(*size))
        expected_grad_input = torch.ones(*size)
        if isinstance(index, Variable):
            index = index.data
        expected_grad_input[index] = 0
        self.assertEqual(x.grad.data, expected_grad_input)
        self.assertEqual(value.grad.data, torch.ones(value.size()))

    def test_setitem(self):
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5,), 3)

    def test_setitem_mask(self):
        mask = torch.ByteTensor(5, 5).bernoulli_()
        self._test_setitem((5, 5), Variable(mask))
        self._test_setitem((5,), Variable(mask[0]))
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        self._test_setitem_tensor((5, 5), Variable(mask))
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_unused_output(self):
        x = Variable(torch.randn(10, 10), requires_grad=True)
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
            Variable(torch.randn(10, 10), creator=CollectOnDelete())

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                     "CUDA not available or <2 GPUs detected")
    def test_unused_output_gpu(self):
        from torch.nn.parallel._functions import Broadcast
        x = Variable(torch.randn(5, 5).float().cuda(), requires_grad=True)
        outputs = Broadcast(list(range(torch.cuda.device_count())))(x)
        y = outputs[-1] * 2
        y.sum().backward()
        self.assertEqual(x.grad.data, torch.ones(5, 5) * 2)

    def test_detach(self):
        x = Variable(torch.randn(10, 10), requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = Variable(torch.randn(10, 10), requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertFalse(y.creator.requires_grad)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(x.grad.data, torch.ones(10, 10))

    def test_type_conversions(self):
        import torch.cuda
        x = Variable(torch.randn(5, 5))
        self.assertIs(type(x.float().data), torch.FloatTensor)
        self.assertIs(type(x.int().data), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIs(type(x.float().cuda().data), torch.cuda.FloatTensor)
            self.assertIs(type(x.int().cuda().data), torch.cuda.IntTensor)
            self.assertIs(type(x.int().cuda().cpu().data), torch.IntTensor)
            if torch.cuda.device_count() > 2:
                x2 = x.float().cuda(1)
                self.assertIs(type(x2.data), torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIs(type(x2.data), torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIs(type(x2.data), torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)

    def test_return_leaf(self):
        class Identity(Function):

            def forward(self, a, b):
                return a, a + b

            def backward(self, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        class Inplace(InplaceFunction):

            def forward(self, a, b):
                self.mark_dirty(a)
                return a.add_(b), b + 2

            def backward(self, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = Variable(torch.randn(5, 5), requires_grad=True)

        q, p = Identity()(x, y)
        # Make sure hooks only receive grad from usage of q, not x.
        q.register_hook(
            lambda grad: self.assertEqual(grad.data, torch.ones(5, 5)))
        (q + p + x).sum().backward()
        self.assertEqual(x.grad.data, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad.data, torch.ones(5, 5))
        del q, p  # these need to be freed, or next part will raise an error

    def test_return_leaf_inplace(self):
        class Inplace(InplaceFunction):

            def forward(self, a, b):
                self.mark_dirty(a)
                return a.add_(b), b + 2

            def backward(self, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = Variable(torch.randn(5, 5))
        y = Variable(torch.randn(5, 5), requires_grad=True)

        fn = Inplace(True)
        q, p = fn(x, y)
        self.assertIs(q, x)
        self.assertIs(q.creator, fn)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad.data, torch.ones(5, 5))

    def test_leaf_assignment(self):
        x = Variable(torch.randn(5, 5))
        y = Variable(torch.randn(5), requires_grad=True)
        z = Variable(torch.randn(5), requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.creator, None)
        x.sum().backward()
        self.assertEqual(y.grad.data, torch.ones(5))
        self.assertEqual(z.grad.data, torch.ones(5) * 2)

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
        x = Variable(torch.ones(5, 5), requires_grad=True)
        y = Variable(torch.ones(5, 5), requires_grad=True)
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
        self.assertEqual(x.grad.data, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad.data, torch.ones(5, 5) * 17)

    def test_functional_blas(self):
        def compare(fn, *args):
            unpacked_args = tuple(arg.data if isinstance(arg, Variable) else arg
                                  for arg in args)
            self.assertEqual(fn(*args).data, fn(*unpacked_args))

        def test_blas_add(fn, x, y, z):
            # Checks all signatures
            compare(fn, x, y, z)
            compare(fn, 0.5, x, y, z)
            compare(fn, 0.5, x, 0.25, y, z)

        def test_blas(fn, x, y):
            compare(fn, x, y)

        test_blas(torch.mm, Variable(torch.randn(2, 10)),
                  Variable(torch.randn(10, 4)))
        test_blas_add(torch.addmm, Variable(torch.randn(2, 4)),
                      Variable(torch.randn(2, 10)), Variable(torch.randn(10, 4)))
        test_blas(torch.bmm, Variable(torch.randn(4, 2, 10)),
                  Variable(torch.randn(4, 10, 4)))
        test_blas_add(torch.addbmm, Variable(torch.randn(2, 4)),
                      Variable(torch.randn(4, 2, 10)), Variable(torch.randn(4, 10, 4)))
        test_blas_add(torch.baddbmm, Variable(torch.randn(4, 2, 4)),
                      Variable(torch.randn(4, 2, 10)), Variable(torch.randn(4, 10, 4)))
        test_blas(torch.mv, Variable(torch.randn(2, 10)),
                  Variable(torch.randn(10)))
        test_blas_add(torch.addmv, Variable(torch.randn(2)),
                      Variable(torch.randn(2, 10)), Variable(torch.randn(10)))
        test_blas(torch.ger, Variable(torch.randn(5)),
                  Variable(torch.randn(6)))
        test_blas_add(torch.addr, Variable(torch.randn(5, 6)),
                      Variable(torch.randn(5)), Variable(torch.randn(6)))

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

        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = MyFn()(x)
        y.sum().backward()
        self.assertEqual(x.grad.data, 2 * x.data)

    def test_too_many_grads(self):
        class MyFn(Function):

            def forward(self, input):
                return input

            def backward(self, grad_output):
                return grad_output, None, None

        x = Variable(torch.randn(5, 5), requires_grad=True)
        y = MyFn()(x)
        y.sum().backward()
        self.assertEqual(x.grad.data, x.data.clone().fill_(1))

    def test_stochastic(self):
        x = Variable(torch.rand(2, 10), requires_grad=True)
        stddevs = Variable(torch.rand(2, 10) * 5, requires_grad=True)
        y = (x * 2).clamp(0, 1)
        y = y / y.sum(1).expand_as(y)
        samples_multi = y.multinomial(5)
        samples_multi_flat = y[0].multinomial(5)
        samples_bernoulli = y.bernoulli()
        samples_norm = torch.normal(y)
        samples_norm_std = torch.normal(y, stddevs)
        z = samples_multi * 2 + 4
        z = z + samples_multi_flat.unsqueeze(0).expand_as(samples_multi)
        z = torch.cat([z, z], 1)
        z = z.double()
        z = z + samples_bernoulli + samples_norm + samples_norm_std
        last_sample = torch.normal(z, 4)
        z = last_sample + 2
        self.assertFalse(z.requires_grad)

        self.assertRaises(RuntimeError, lambda: z.backward(retain_variables=True))
        samples_multi.reinforce(torch.randn(2, 5))
        self.assertRaises(RuntimeError, lambda: z.backward(retain_variables=True))
        samples_multi_flat.reinforce(torch.randn(5))
        self.assertRaises(RuntimeError, lambda: z.backward(retain_variables=True))
        samples_bernoulli.reinforce(torch.randn(2, 10))
        self.assertRaises(RuntimeError, lambda: z.backward(retain_variables=True))
        samples_norm.reinforce(torch.randn(2, 10))
        self.assertRaises(RuntimeError, lambda: z.backward(retain_variables=True))
        samples_norm_std.reinforce(torch.randn(2, 10))
        # We don't have to specify rewards w.r.t. last_sample - it doesn't
        # require gradient

        last_sample.backward(retain_variables=True)
        z.backward()

        self.assertGreater(x.grad.data.abs().sum(), 0)

    def test_stochastic_sequence(self):
        x = Variable(torch.rand(10).clamp_(0, 1), requires_grad=True)
        b = x.bernoulli()
        n1 = torch.normal(b, x)
        n2 = torch.normal(n1, 2)

        b.reinforce(torch.randn(10))
        n1.reinforce(torch.randn(10))
        n2.reinforce(torch.randn(10))

        n2.backward()

        self.assertGreater(x.grad.data.abs().sum(), 0)

    def test_stochastic_output(self):
        x = Variable(torch.rand(10), requires_grad=True)
        b = x.clone().clamp(0, 1).bernoulli()
        b.reinforce(torch.randn(10))
        b.backward()
        self.assertGreater(x.grad.data.abs().sum(), 0)

    def test_pickle(self):
        x = Variable(torch.randn(10, 10), requires_grad=True)
        y = Variable(torch.randn(10, 10), volatile=True)
        z = Variable(torch.randn(10, 10), requires_grad=False)

        def assert_strict_equal(var1, var2):
            self.assertEqual(var1.data, var2.data)
            self.assertEqual(var1.requires_grad, var2.requires_grad)
            self.assertEqual(var1.volatile, var2.volatile)

        serialized = [pickle.dumps([x, y, z], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc, zc = pickle.loads(dump)
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)
            assert_strict_equal(zc, z)

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

        x = Variable(torch.randn(5), requires_grad=True)
        a, b = F1()(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2()(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad.data, torch.ones(x.size()))


def index_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape).mul_(max_indices).floor_().long()
    return Variable(index, requires_grad=False)


def gather_variable(shape, index_dim, max_indices):
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = torch.LongTensor(*shape)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(
            torch.randperm(max_indices)[:shape[batch_dim]])
    return Variable(index, requires_grad=False)


L = 20
M = 10
S = 5
function_tests = [
    (Add, (), ((M, M), (M, M))),
    (Sub, (), ((M, M), (M, M))),
    (Mul, (), ((M, M), (M, M))),
    (Div, (), ((M, M), torch.rand(M, M) + 5e-2)),
    (Pow, (), (torch.rand(M, M) + 1e-3, torch.rand(M, M) + 0.1)),
    (AddConstant, (3.14,), ((L, L),)),
    (SubConstant, (3.14,), ((L, L),)),
    (SubConstant, (3.14, True), ((L, L),), 'from_tensor'),
    (MulConstant, (3.14,), ((L, L),)),
    (DivConstant, (3.14, True), (torch.rand(L, L) + 1e-1,), 'by_tensor'),
    (PowConstant, (3.14,), (torch.rand(L, L),)),
    (PowConstant, (3.14, True), (torch.rand(L, L),), 'tensor_power'),
    (Transpose, (0, 1), (torch.rand(L, L),)),
    (Transpose, (2, 0), (torch.rand(S, S, S),), '3d'),
    (Permute, ((0, 4, 3, 5, 1, 2),), ((1, 2, 3, 4, 5, 6),)),
    (Index, ((1, 2),), (torch.rand(S, S, S),)),
    (Index, (slice(0, 3),), (torch.rand(S, S, S),), 'slice'),
    (Index, ((slice(0, 3), 1),), (torch.rand(S, S, S),), 'slice_index'),
    (View, (S * S, S), (torch.rand(S, S, S),)),
    (Expand, ((S, 5, S, 5),), ((S, 1, S, 1),)),
    (Exp, (), (torch.rand(S, S, S),)),
    (Log, (), (torch.rand(S, S, S) + 1e-2,)),
    (Log1p, (), (torch.rand(S, S, S),)),
    (Tanh, (), ((S, S, S),)),
    (Sigmoid, (), ((S, S, S),)),
    (Sinh, (), ((S, S, S),)),
    (Cosh, (), ((S, S, S),)),
    (Abs, (), ((S, S, S),)),
    (Clamp, (0, 1), ((S, S, S),)),
    (Sqrt, (), (torch.rand(S, S, S) + 5e-4,)),
    (Sin, (), ((S, S, S),)),
    (Cos, (), ((S, S, S),)),
    (Tan, (), (torch.randn(S, S, S).clamp(-1, 1),)),
    (Asin, (), (torch.randn(S, S, S).clamp(-0.9, 0.9),)),
    (Acos, (), (torch.randn(S, S, S).clamp(-0.9, 0.9),)),
    (Atan, (), ((S, S, S),)),
    (Reciprocal, (), (torch.rand(S, S, S) + 0.1,)),
    (Cmax, (), ((S, S, S), (S, S, S))),
    (Cmin, (), ((S, S, S), (S, S, S))),
    (Round, (), ((S, S, S),)),
    (Sign, (), ((S, S, S),)),
    (Trunc, (), ((S, S, S),)),
    (Floor, (), ((S, S, S),)),
    (Ceil, (), ((S, S, S),)),
    (Frac, (), ((S, S, S),)),
    (Fmod, (1.5,), ((S, S, S),)),
    (Lerp, (0.2,), ((S, S, S), (S, S, S))),
    (Rsqrt, (), (torch.rand(S, S, S) + 1e-2,)),
    (Remainder, (1.5,), ((S, S, S),)),
    (CmaxConstant, (0.5,), ((S, S, S),)),
    (CminConstant, (0.5,), ((S, S, S),)),
    (Mean, (), ((S, S, S),)),
    (Mean, (1,), ((S, S, S),), 'dim'),
    (Sum, (), ((S, S, S),)),
    (Sum, (1,), ((S, S, S),), 'dim'),
    (Prod, (), ((S, S, S),)),
    (Prod, (1,), ((S, S, S),), 'dim'),
    (Addmm, (), ((S, M), (S, S), (S, M)),),
    (Addmm, (0.1, 1), ((S, M), (S, S), (S, M)), 'coef'),
    (Addbmm, (), ((S, M), (S, S, S), (S, S, M)),),
    (Addbmm, (0.1, 0.4), ((S, M), (S, S, S), (S, S, M)), 'coef'),
    (Baddbmm, (), ((S, S, M), (S, S, S), (S, S, M)),),
    (Baddbmm, (0.1, 0.4), ((S, S, M), (S, S, S), (S, S, M)), 'coef'),
    (Addmv, (), ((S,), (S, M), (M,)),),
    (Addmv, (0.1, 0.4), ((S,), (S, M), (M,)), 'coef'),
    (Addr, (), ((S, M), (S,), (M,)),),
    (Addr, (0.1, 0.4), ((S, M), (S,), (M,)), 'coef'),
    (Dot, (), ((L,), (L,)),),
    (Max, (), ((S, S, S),),),
    (Repeat, (torch.Size([2, 3, 1, 4]),), ((S, S, S, S),)),
    (Min, (), ((S, S, S),),),
    (Max, (0,), ((S, S, S),), 'dim'),
    (Min, (0,), ((S, S, S),), 'dim'),
    (Mode, (0,), ((S, S, S),),),
    (Kthvalue, (2, 0), ((S, S, S),),),
    (Median, (0,), ((S, S, S),),),
    (Norm, (1.5,), (torch.rand(S, S, S),), '1_5'),
    (Norm, (), ((S, S, S),), '2'),
    (Norm, (3,), ((S, S, S),), '3'),
    (Norm, (1.5, 0), (torch.rand(S, S, S),), '1_5_dim'),
    (Norm, (2, 0), ((S, S, S),), '2_dim'),
    (Norm, (3, 0), ((S, S, S),), '3_dim'),
    (Addcmul, (), ((S, S), (S, S), (S, S))),
    (Addcmul, (0.6,), ((S, S), (S, S), (S, S)), 'scale'),
    (Addcdiv, (), ((S, S), (S, S), torch.rand(S, S) + 5e-2)),
    (Addcdiv, (0.6,), ((S, S), (S, S), torch.rand(S, S) + 5e-2), 'scale'),
    (IndexAdd, (0,), ((S, S), index_variable(2, S), (2, S))),
    # (IndexCopy,     (0,),               ((S, S), index_variable(2, S), (2, S))      ),
    (IndexFill, (0, 2), ((S, S), index_variable(2, S))),
    (IndexSelect, (0,), ((S, S), index_variable(2, S))),
    (Gather, (0,), ((M, S), gather_variable((S, S), 1, M))),
    (Gather, (1,), ((M, S), gather_variable((M, S // 2), 0, S)), 'dim1'),
    (Scatter, (0,), ((M, S), gather_variable((S, S), 1, M), (S, S))),
    (Scatter, (1,), ((M, S), gather_variable((M, S // 2), 0, S), (M, S // 2)), 'dim1'),
    (Concat, (0,), ((1, S, S), (2, S, S), (3, S, S))),
    (Resize, (S * S, S), ((S, S, S),)),
    (Diag, (), ((S, S),), '2d'),
    (Diag, (), ((S,),), '1d'),
    (Tril, (), ((S, S),)),
    (Tril, (2,), ((S, S),), 'idx'),
    (Triu, (), ((S, S),)),
    (Triu, (2,), ((S, S),), 'idx'),
    (Clone, (), ((S, M, S),)),
    (Squeeze, (), ((S, 1, M, 1),)),
    (Squeeze, (1,), ((S, 1, M, 1),), 'dim'),
    (Unsqueeze, (0,), ((S, M, S),), '0'),
    (Unsqueeze, (1,), ((S, M, S),), '1'),
    # (MaskedCopy,    (),                 ((S, S), Variable(torch.randn(S, S).gt(0), requires_grad=False), (S, S),)),
    (MaskedFill, (10,), ((S, S), Variable(torch.randn(S, S).gt(0), requires_grad=False))),
    (MaskedSelect, (), ((S, S), Variable(torch.randn(S, S).gt(0), requires_grad=False))),
    (Sort, (), ((S, M, S),)),
    (Sort, (1,), ((S, M, S),), 'dim'),
    (Sort, (1, True), ((S, M, S),), 'dim_desc'),
    (Topk, (3,), ((S, M, S),)),
    (Topk, (3, 1), ((S, M, S),), 'dim'),
    (Topk, (3, 1, True), ((S, M, S),), 'dim_desc'),
    (Topk, (3, 1, True, True), ((S, M, S),), 'dim_desc_sort'),
]


method_tests = [
    ('add', (S, S, S), ((S, S, S),)),
    ('add', (S, S, S), (3.14,), 'constant'),
    ('sub', (S, S, S), ((S, S, S),)),
    ('sub', (S, S, S), (3.14,), 'constant'),
    ('mul', (S, S, S), ((S, S, S),)),
    ('mul', (S, S, S), (3.14,), 'constant'),
    ('div', (S, S, S), ((S, S, S),)),
    ('div', (S, S, S), (3.14,), 'constant'),
    ('pow', (S, S, S), ((S, S, S),)),
    ('pow', (S, S, S), (3.14,), 'constant'),
    ('transpose', (1, 2, 3), (1, 2)),
    ('t', (1, 2), ()),
    ('view', (S, S, S), (S * S, S),),
    ('view_as', (S, S, S), ((S * S, S),)),
    ('expand', (S, 1, S), (S, S, S)),
    ('expand', (torch.Size([S, 1, S]),), (S, S, S), 'size'),
    ('exp', (S, S, S), ()),
    ('log', (S, S, S), ()),
    ('log1p', (S, S, S), ()),
    ('tanh', (S, S, S), ()),
    ('sigmoid', (S, S, S), ()),
    ('sinh', (S, S, S), ()),
    ('cosh', (S, S, S), ()),
    ('abs', (S, S, S), ()),
    ('clamp', (S, S, S), (0, 1)),
    ('sqrt', (S, S, S), ()),
    ('sin', (S, S, S), ()),
    ('cos', (S, S, S), ()),
    ('tan', (S, S, S), ()),
    ('asin', (S, S, S), ()),
    ('acos', (S, S, S), ()),
    ('atan', (S, S, S), ()),
    ('reciprocal', (S, S, S), ()),
    ('round', (S, S, S), ()),
    ('sign', (S, S, S), ()),
    ('trunc', (S, S, S), ()),
    ('floor', (S, S, S), ()),
    ('ceil', (S, S, S), ()),
    ('rsqrt', (S, S, S), ()),
    ('fmod', (S, S, S), (1.5,)),
    ('remainder', (S, S, S), (1.5,)),
    ('lerp', (S, S, S), ((S, S, S), 0.4)),
    ('max', (S, S, S), ()),
    ('max', (S, S, S), ((S, S, S),), 'elementwise'),
    ('min', (S, S, S), ()),
    ('min', (S, S, S), ((S, S, S),), 'elementwise'),
    ('mean', (S, S, S), ()),
    ('mean', (S, S, S), (1,), 'dim'),
    ('sum', (S, S, S), ()),
    ('sum', (S, S, S), (1,), 'dim'),
    ('prod', (S, S, S), ()),
    ('prod', (S, S, S), (1,), 'dim'),
    ('var', (S, S, S), ()),
    ('var', (S, S, S), (1,), 'dim'),
    ('std', (S, S, S), ()),
    ('std', (S, S, S), (1,), 'dim'),
    ('renorm', (S, S, S), (2, 1, 0.5)),
    ('renorm', (S, S, S), (1, 2, 3), 'norm_1'),
    ('repeat', (S, S, S, S), (2, 3, 1, 4)),
    ('addmm', (S, M), ((S, S), (S, M)),),
    ('addmm', (S, M), (0.2, 0.6, (S, S), (S, M)), 'coef'),
    ('addbmm', (S, M), ((S, S, S), (S, S, M)),),
    ('addbmm', (S, M), (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'),
    ('baddbmm', (S, S, M), ((S, S, S), (S, S, M)),),
    ('baddbmm', (S, S, M), (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'),
    ('addmv', (S,), ((S, M), (M,)),),
    ('addmv', (S,), (0.2, 0.6, (S, M), (M,)), 'coef'),
    ('addr', (S, M), ((S,), (M,)),),
    ('addr', (S, M), (0.2, 0.6, (S,), (M,)), 'coef'),
    ('dot', (L,), ((L,),),),
    ('addcmul', (S, S), ((S, S), (S, S))),
    ('addcmul', (S, S), (0.5, (S, S), (S, S)), 'scale'),
    ('addcdiv', (S, S), ((S, S), (S, S))),
    ('addcdiv', (S, S), (0.5, (S, S), (S, S)), 'scale'),
    ('norm', (S, S, S), (2,)),
    ('norm', (S, S, S), (2, 1), 'dim'),
    ('dist', (S, S, S), ((S, S, S),)),
    ('dist', (S, S, S), ((S, S, S), 4), '4'),
    ('index_select', (S, S, S), (0, index_variable(2, S))),
    ('diag', (M, M), (), '2d'),
    ('diag', (M,), (), '1d'),
    ('tril', (M, M), ()),
    ('triu', (M, M), ()),
    ('clone', (S, M, S), ()),
    ('eq', (S, S, S), ((S, S, S),)),
    ('ne', (S, S, S), ((S, S, S),)),
    ('gt', (S, S, S), ((S, S, S),)),
    ('ge', (S, S, S), ((S, S, S),)),
    ('lt', (S, S, S), ((S, S, S),)),
    ('le', (S, S, S), ((S, S, S),)),
    ('eq', (S, S, S), (0,), 'scalar'),
    ('ne', (S, S, S), (0,), 'scalar'),
    ('gt', (S, S, S), (0,), 'scalar'),
    ('ge', (S, S, S), (0,), 'scalar'),
    ('lt', (S, S, S), (0,), 'scalar'),
    ('le', (S, S, S), (0,), 'scalar'),
    ('permute', (1, 2, 3, 4), (0, 2, 3, 1)),
    ('select', (S, S, S), (1, 2)),
    ('narrow', (S, S, S), (1, 2, 2)),
    ('squeeze', (S, 1, S, 1), ()),
    ('squeeze', (S, 1, S, 1), (1,), '1_dim'),
    ('squeeze', (S, 1, S, 1), (2,), 'not_1_dim'),
    ('unsqueeze', (S, S, S), (0,), 'first'),
    ('unsqueeze', (S, S, S), (1,), 'middle'),
    ('unsqueeze', (S, S, S), (3,), 'last'),
    ('masked_select', (M, M), (Variable(torch.ByteTensor(M, M).bernoulli_(), requires_grad=False),)),
    ('masked_fill_', (M, M), (Variable(torch.ByteTensor(M, M).bernoulli_(), requires_grad=False), 10)),
    ('masked_copy_', (M, M), (Variable(torch.ByteTensor(M, M).bernoulli_(), requires_grad=False), (M, M))),
]
# TODO: mm, bmm, mv, ger
# TODO: max, min with dim (problem with indices)
# TODO: mode, median, sort, kthvalue, topk (problem with indices)
# TODO: indexAdd, indexCopy, indexFill
# TODO: resize, resize_as (tensors only have resize_ and resize_as_)
# TODO: clamp with min/max


def create_input(call_args):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)

    def map_arg(arg):
        if isinstance(arg, tuple) and not isinstance(arg[0], Variable):
            return Variable(torch.randn(*arg).double(), requires_grad=True)
        elif torch.is_tensor(arg):
            if isinstance(arg, torch.FloatTensor):
                return Variable(arg.double(), requires_grad=True)
            else:
                return Variable(arg, requires_grad=True)
        else:
            return arg
    return tuple(map_arg(arg) for arg in call_args)


def unpack_variables(args):
    if isinstance(args, Variable):
        return args.data
    elif isinstance(args, tuple):
        return tuple(unpack_variables(elem) for elem in args)
    else:
        return args


ignore_inplace = set((
    'test_DivConstant_by_tensor',
))


for test in function_tests:
    cls, constructor_args, call_args = test[:3]
    test_name = 'test_' + cls.__name__ + ('_' + test[3] if len(test) == 4 else '')

    def do_test(self, cls=cls, constructor_args=constructor_args,
                call_args=call_args, test_name=test_name):
        input = create_input(call_args)
        output = cls(*constructor_args)(*input)
        if not isinstance(output, tuple):
            output = (output,)
        for i, o in enumerate(output):
            if not o.requires_grad:
                continue
            analytical = get_analytical_jacobian(input, o)

            def fn(input):
                tmp = cls(*constructor_args)(*input)
                if not isinstance(tmp, tuple):
                    tmp = (tmp,)
                return tmp[i].data
            numerical = get_numerical_jacobian(fn, input, input)
            self.assertLessEqual(
                max(a.add(-1, n).abs().max() for a, n in zip(analytical, numerical)),
                PRECISION
            )

        if test_name not in ignore_inplace and issubclass(cls, InplaceFunction):
            inplace_input = deepcopy(input)
            inplace_input_copy = tuple(i + 0 for i in inplace_input)
            fn = cls(*constructor_args, inplace=True)
            inplace_output = fn(*inplace_input_copy)
            if not isinstance(inplace_output, tuple):
                inplace_output = (inplace_output,)
            self.assertEqual(inplace_output, output)
            # Check that gradient is the same
            for inp_i, i in zip(inplace_input, input):
                if inp_i.grad is not None:
                    inp_i.grad.data.zero_()
                if i.grad is not None:
                    i.grad.data.zero_()
            for io, o in zip(inplace_output, output):
                grad = torch.randn(*io.size()).double()
                io.backward(grad)
                o.backward(grad)
            for inp_i, i in zip(inplace_input, input):
                self.assertEqual(inp_i.grad, i.grad)

    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)


EXCLUDE_FUNCTIONAL = {
    'addmm',
    'addbmm',
    'baddbmm',
    'addmv',
    'addr',
}
for test in method_tests:
    name, self_size, args = test[:3]
    test_name = 'test_' + name + ('_' + test[3] if len(test) == 4 else '')

    def do_test(self, name=name, self_size=self_size, args=args, test_name=test_name):
        def check(name):
            self_variable = create_input((self_size,))[0]
            args_variable = create_input(args)
            self_tensor = deepcopy(self_variable.data)
            args_tensor = deepcopy(unpack_variables(args_variable))
            output_variable = getattr(self_variable, name)(*args_variable)
            output_tensor = getattr(self_tensor, name)(*args_tensor)
            if not torch.is_tensor(output_tensor) and not isinstance(output_tensor, tuple):
                output_tensor = torch.DoubleTensor((output_tensor,))
            self.assertEqual(unpack_variables(output_variable), output_tensor)
            # TODO: check that both have changed after adding all inplace ops

            # functional interface tests
            if hasattr(torch, name) and name not in EXCLUDE_FUNCTIONAL:
                f_args_variable = (self_variable,) + args_variable
                f_args_tensor = (self_tensor,) + args_tensor
                output_variable = getattr(torch, name)(*f_args_variable)
                output_tensor = getattr(torch, name)(*f_args_tensor)
                if not torch.is_tensor(output_tensor) and not isinstance(output_tensor, tuple):
                    output_tensor = torch.DoubleTensor((output_tensor,))
                self.assertEqual(unpack_variables(output_variable), output_tensor)

        check(name)
        inplace_name = name + '_'
        if hasattr(Variable(torch.ones(1)), inplace_name):
            try:
                check(inplace_name)
            except Exception as e:
                if 'only supports scalar' not in e.args[0]:
                    raise

    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)


if __name__ == '__main__':
    run_tests()
