import unittest

from common import make_jacobian, TestCase, iter_tensors, get_numerical_jacobian
from torch.autograd.functions import *

PRECISION = 1e-3

def iter_gradients(x):
    if isinstance(x, Variable):
        yield x.grad
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
        output.backward(grad_output)
        for jacobian_x, d_x in zip(jacobian, iter_gradients(input)):
            jacobian_x[:,i] = d_x

    return jacobian

class TestAutograd(TestCase):

    def test_hooks(self):
        x = Variable(torch.ones(5, 5))
        y = Variable(torch.ones(5, 5) * 4)

        counter = [0]
        def bw_hook(inc, grad):
            self.assertTrue(torch.isTensor(grad))
            counter[0] += inc

        z = x ** 2 + x * 2 + x * y + y
        z.register_hook('test', lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5))
        self.assertEqual(counter[0], 1)

        z.register_hook('test2', lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5))
        self.assertEqual(counter[0], 4)

        z.remove_hook('test2')
        z.backward(torch.ones(5, 5))
        self.assertEqual(counter[0], 5)

    def test_backward(self):
        x_t = torch.randn(5, 5)
        y_t = torch.rand(5, 5) + 0.1
        z_t = torch.randn(5, 5)
        grad_output = torch.randn(5, 5)
        x = Variable(x_t)
        y = Variable(y_t)
        z = Variable(z_t)

        a = x + (y * z) + 4 * z**2 * x / y
        a.backward(grad_output)
        x_grad = 4 * z_t.pow(2) / y_t + 1
        y_grad = z_t - 4 * x_t * z_t.pow(2) / y_t.pow(2)
        z_grad = 8 * x_t * z_t / y_t + y_t
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_volatile(self):
        x = Variable(torch.ones(5, 5))
        y = Variable(torch.ones(5, 5) * 4, volatile=True)

        z = x ** 2
        self.assertFalse(z.volatile)
        self.assertTrue(z.requires_grad)
        self.assertIsNotNone(z.creator)
        z.backward(torch.ones(5, 5))
        self.assertEqual(x.grad, torch.ones(5, 5) * 2)

        w = z + y
        self.assertTrue(w.volatile)
        self.assertFalse(w.requires_grad)
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
        self.assertIsNone(w.creator)


L = 20
M = 10
S = 5
tests = [
    (Add, (), ((M, M), (M, M))),
    (Sub, (), ((M, M), (M, M))),
    (Mul, (), ((M, M), (M, M))),
    (Div, (), ((M, M), torch.rand(M, M) + 1e-2)),
    (Pow, (), (torch.rand(M, M) + 1e-3, torch.rand(M, M) + 0.1)),
    (AddConstant, (3.14,), ((L, L),)),
    (SubConstant, (3.14,), ((L, L),)),
    (SubConstant, (3.14, True), ((L, L),), 'from_tensor'),
    (MulConstant, (3.14,), ((L, L),)),
    (DivConstant, (3.14, True), (torch.rand(L, L) + 1e-2,), 'by_tensor'),
    (PowConstant, (3.14,), (torch.rand(L, L),)),
    (Transpose, (0, 1), (torch.rand(L, L),)),
    (Transpose, (2, 0), (torch.rand(S, S, S),), '3d'),
    (Index, (1, 2), (torch.rand(S, S, S),)),
    (Index, (slice(0, 3),), (torch.rand(S, S, S),), 'slice'),
    (View, (S*S, S), (torch.rand(S, S, S),)),
    (Exp,  (), (torch.rand(S, S, S),)),
    (Log,  (), (torch.rand(S, S, S) + 1e-2,)),
    (Log1p,  (), (torch.rand(S, S, S),)),
]

def create_input(call_args):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)
    def map_arg(arg):
        if isinstance(arg, tuple):
            return Variable(torch.randn(*arg))
        else:
            return Variable(arg)
    return tuple(map_arg(arg) for arg in call_args)


for test in tests:
    cls, constructor_args, call_args = test[:3]
    def do_test(self, cls=cls, constructor_args=constructor_args, call_args=call_args):
        input = create_input(call_args)
        output = cls(*constructor_args)(*input)
        for i, o in enumerate(output):
            analytical = get_analytical_jacobian(input, o)
            def fn(input):
                return cls(*constructor_args)(*input)[i].data
            numerical = get_numerical_jacobian(fn, input, input)
            self.assertLessEqual(
                max(a.add(-1, n).abs().max() for a, n in zip(analytical, numerical)),
                PRECISION
            )

    test_name = 'test_' + cls.__name__ + ('_' + test[3] if len(test) == 4 else '')
    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)


if __name__ == '__main__':
    unittest.main()
