import math
import unittest
from copy import deepcopy

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

    def test_inplace(self):
        x = Variable(torch.ones(5, 5))
        y = Variable(torch.ones(5, 5) * 4)

        z = x * y
        q = z + y
        w = z * y
        z.dirty = True
        # Add doesn't need it's inputs to do backward, so it shouldn't raise
        q.backward(torch.ones(5, 5))
        # Mul saves both inputs in forward, so it should raise
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        z = x * y
        q = z * y
        r = z + y
        w = z.add_(y)
        # w is a the last expression, so this should succeed
        w.backward(torch.ones(5, 5))
        # r doesn't use the modified value in backward, so it should succeed
        r.backward(torch.ones(5, 5))
        # q uses dirty z, so it should raise
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        x.grad.zero_()
        m = x / 2
        z = m + y / 8
        q = z * y
        r = z + y
        w = z.exp_()
        self.assertTrue(z.dirty)
        r.backward(torch.ones(5, 5))
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        w.backward(torch.ones(5, 5))
        self.assertEqual(x.grad, torch.Tensor(5, 5).fill_((1 + math.e) / 2))
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))


L = 20
M = 10
S = 5
function_tests = [
    (Add,           (),                 ((M, M), (M, M))                            ),
    (Sub,           (),                 ((M, M), (M, M))                            ),
    (Mul,           (),                 ((M, M), (M, M))                            ),
    (Div,           (),                 ((M, M), torch.rand(M, M) + 1e-2)           ),
    (Pow,           (),                 (torch.rand(M, M) + 1e-3, torch.rand(M, M) + 0.1)),
    (AddConstant,   (3.14,),            ((L, L),)                                   ),
    (SubConstant,   (3.14,),            ((L, L),)                                   ),
    (SubConstant,   (3.14, True),       ((L, L),),                  'from_tensor'   ),
    (MulConstant,   (3.14,),            ((L, L),)),
    (DivConstant,   (3.14, True),       (torch.rand(L, L) + 1e-2,), 'by_tensor'     ),
    (PowConstant,   (3.14,),            (torch.rand(L, L),)                         ),
    (Transpose,     (0, 1),             (torch.rand(L, L),)                         ),
    (Transpose,     (2, 0),             (torch.rand(S, S, S),),     '3d'            ),
    (Permute,       (0, 4, 3, 5, 1, 2), ((1, 2, 3, 4, 5, 6),),                      ),
    (Index,         (1, 2),             (torch.rand(S, S, S),)                      ),
    (Index,         (slice(0, 3),),     (torch.rand(S, S, S),),     'slice'         ),
    (View,          (S*S, S),           (torch.rand(S, S, S),)                      ),
    (Expand,        (S, 5, S, 5),       ((S, 1, S, 1),)                             ),
    (Exp,           (),                 (torch.rand(S, S, S),)                      ),
    (Log,           (),                 (torch.rand(S, S, S) + 1e-2,)               ),
    (Log1p,         (),                 (torch.rand(S, S, S),)                      ),
    (Tanh,          (),                 ((S, S, S),)                                ),
    (Sigmoid,       (),                 ((S, S, S),)                                ),
    (Sinh,          (),                 ((S, S, S),)                                ),
    (Cosh,          (),                 ((S, S, S),)                                ),
    (Abs,           (),                 ((S, S, S),)                                ),
    (Clamp,         (0, 1),             ((S, S, S),)                                ),
    (Sqrt,          (),                 (torch.rand(S, S, S),)                      ),
    (Sin,           (),                 ((S, S, S),)                                ),
    (Cos,           (),                 ((S, S, S),)                                ),
    (Tan,           (),                 (torch.randn(S, S, S).clamp(-1, 1),)        ),
    (Asin,          (),                 (torch.randn(S, S, S).clamp(-0.9, 0.9),)    ),
    (Acos,          (),                 (torch.randn(S, S, S).clamp(-0.9, 0.9),)    ),
    (Atan,          (),                 ((S, S, S),)                                ),
    (Cinv,          (),                 (torch.rand(S, S, S) + 0.1,)                ),
    (Cmax,          (),                 ((S, S, S), (S, S, S))                      ),
    (Cmin,          (),                 ((S, S, S), (S, S, S))                      ),
    (CmaxConstant,  (0.5,),             ((S, S, S),)                                ),
    (CminConstant,  (0.5,),             ((S, S, S),)                                ),
    (Mean,          (),                 ((S, S, S),)                                ),
    (Mean,          (1,),               ((S, S, S),),               'dim'           ),
    (Sum,           (),                 ((S, S, S),)                                ),
    (Sum,           (1,),               ((S, S, S),),               'dim'           ),
    (Addmm,         (),                 ((S, M), (S, S), (S, M)),                   ),
    (Addmm,         (0.1, 1),           ((S, M), (S, S), (S, M)),   'coef'          ),
    (Addbmm,        (),                 ((S, M), (S, S, S), (S, S, M)),             ),
    (Addbmm,        (0.1, 0.4),         ((S, M), (S, S, S), (S, S, M)), 'coef'      ),
    (Baddbmm,       (),                 ((S, S, M), (S, S, S), (S, S, M)),          ),
    (Baddbmm,       (0.1, 0.4),         ((S, S, M), (S, S, S), (S, S, M)), 'coef'   ),
    (Addmv,         (),                 ((S,), (S, M), (M,)),                       ),
    (Addmv,         (0.1, 0.4),         ((S,), (S, M), (M,)),       'coef'          ),
    (Addr,          (),                 ((S, M), (S,), (M,)),                       ),
    (Addr,          (0.1, 0.4),         ((S, M), (S,), (M,)),       'coef'          ),
    (Dot,           (),                 ((L,), (L,)),                               ),
    (Max,           (0,),               ((S, S, S),),                               ),
    (Min,           (0,),               ((S, S, S),),                               ),
    (Mode,          (0,),               ((S, S, S),),                               ),
    (Median,        (0,),               ((S, S, S),),                               ),
]


method_tests = [
    ('add',         (S, S, S),          ((S, S, S),)                                ),
    ('add',         (S, S, S),          (3.14,),                    'constant'      ),
    ('sub',         (S, S, S),          ((S, S, S),)                                ),
    ('sub',         (S, S, S),          (3.14,),                    'constant'      ),
    ('mul',         (S, S, S),          ((S, S, S),)                                ),
    ('mul',         (S, S, S),          (3.14,),                    'constant'      ),
    ('div',         (S, S, S),          ((S, S, S),)                                ),
    ('div',         (S, S, S),          (3.14,),                    'constant'      ),
    ('pow',         (S, S, S),          ((S, S, S),)                                ),
    ('pow',         (S, S, S),          (3.14,),                    'constant'      ),
    ('transpose',   (1, 2, 3),          (1, 2)                                      ),
    ('t',           (1, 2),             ()                                          ),
    ('view',        (S, S, S),          (S*S, S),                                   ),
    ('viewAs',      (S, S, S),          ((S*S, S),)                                 ),
    ('expand',      (S, 1, S),          (S, S, S)                                   ),
    ('exp',         (S, S, S),          ()                                          ),
    ('log',         (S, S, S),          ()                                          ),
    ('log1p',       (S, S, S),          ()                                          ),
    ('tanh',        (S, S, S),          ()                                          ),
    ('sigmoid',     (S, S, S),          ()                                          ),
    ('sinh',        (S, S, S),          ()                                          ),
    ('cosh',        (S, S, S),          ()                                          ),
    ('abs',         (S, S, S),          ()                                          ),
    ('clamp',       (S, S, S),          (0, 1)                                      ),
    ('sqrt',        (S, S, S),          ()                                          ),
    ('sin',         (S, S, S),          ()                                          ),
    ('cos',         (S, S, S),          ()                                          ),
    ('tan',         (S, S, S),          ()                                          ),
    ('asin',        (S, S, S),          ()                                          ),
    ('acos',        (S, S, S),          ()                                          ),
    ('atan',        (S, S, S),          ()                                          ),
    ('cinv',        (S, S, S),          ()                                          ),
    ('cmax',        (S, S, S),          ((S, S, S),)                                ),
    ('cmax',        (S, S, S),          (0.5,),                     'constant'      ),
    ('cmin',        (S, S, S),          ((S, S, S),)                                ),
    ('cmin',        (S, S, S),          (0.5,),                     'constant'      ),
    ('mean',        (S, S, S),          ()                                          ),
    ('mean',        (S, S, S),          (1,),                       'dim'           ),
    ('sum',         (S, S, S),          ()                                          ),
    ('sum',         (S, S, S),          (1,),                       'dim'           ),
    ('addmm',       (S, M),             ((S, S), (S, M)),                           ),
    ('addmm',       (S, M),             (0.2, 0.6, (S, S), (S, M)), 'coef'          ),
    ('addbmm',      (S, M),             ((S, S, S), (S, S, M)),                     ),
    ('addbmm',      (S, M),             (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'    ),
    ('baddbmm',     (S, S, M),          ((S, S, S), (S, S, M)),                     ),
    ('baddbmm',     (S, S, M),          (0.2, 0.6, (S, S, S), (S, S, M)), 'coef'    ),
    ('addmv',       (S,),               ((S, M), (M,)),                             ),
    ('addmv',       (S,),               (0.2, 0.6, (S, M), (M,)),   'coef'          ),
    ('addr',        (S, M),             ((S,), (M,)),                               ),
    ('addr',        (S, M),             (0.2, 0.6, (S,), (M,)),     'coef'          ),
    ('dot',         (L,),               ((L,),),                                    ),
    ('max',         (S, S, S),          ()                                          ),
    ('min',         (S, S, S),          ()                                          ),
]
# TODO: max, min with dim
# TODO: mode, median


def create_input(call_args):
    if not isinstance(call_args, tuple):
        call_args = (call_args,)
    def map_arg(arg):
        if isinstance(arg, tuple):
            return Variable(torch.randn(*arg))
        elif torch.isTensor(arg):
            return Variable(arg)
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
            inplace_output = cls(*constructor_args, inplace=True)(*input)
            if not isinstance(inplace_output, tuple):
                inplace_output = (inplace_output,)
            self.assertEqual(inplace_output, output)

    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)


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
            if not torch.isTensor(output_tensor) and not isinstance(output_tensor, tuple):
                output_tensor = torch.DoubleTensor((output_tensor,))
            self.assertEqual(unpack_variables(output_variable), output_tensor)
            # TODO: check that both have changed after adding all inplace ops

        check(name)
        inplace_name = name + '_'
        if hasattr(Variable(torch.ones(1)), inplace_name):
            try:
                check(inplace_name)
            except Exception as e:
                if not 'only supports scalar' in e.args[0]:
                    raise


    assert not hasattr(TestAutograd, test_name), 'Two tests have the same name: ' + test_name
    setattr(TestAutograd, test_name, do_test)



if __name__ == '__main__':
    unittest.main()

