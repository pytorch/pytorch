import math
import tempfile
import unittest

import torch
import torch.cuda

from common import TestCase, get_gpu_type, to_gpu

def is_floating(t):
    return type(t) in [torch.FloatTensor, torch.DoubleTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]

types = [
    torch.FloatTensor,
    torch.DoubleTensor,
    torch.LongTensor,
    torch.IntTensor,
    torch.ShortTensor,
    torch.CharTensor,
    torch.ByteTensor,
]

def number(floating, integer, t):
    name = type(t).__name__
    if 'Double' in name or 'Float' in name or 'Half' in name:
        return floating
    else:
        return integer
# TODO: check HalfTensor

S = 10
M = 50

def make_tensor(t, *sizes):
    return t(*sizes).copy_(torch.randn(*sizes))

def small_2d(t):
    return make_tensor(t, S, S)

def small_3d(t):
    return make_tensor(t, S, S, S)

def medium_1d(t):
    return make_tensor(t, M)

def medium_2d(t):
    return make_tensor(t, M, M)

def small_3d_ones(t):
    return t(S, S, S).copy_(torch.ones(S, S, S))

def small_3d_positive(t):
    min_val = 1e-3 if is_floating(t) else 2
    return make_tensor(t, S, S, S).clamp_(min_val, 120)

def small_3d_unique(t):
    return t(S, S, S).copy_(torch.range(1, S*S*S))

def new_t(*sizes):
    def tmp(t):
        return t(*sizes).copy_(torch.randn(*sizes))
    return tmp

tests = [
    ('add',           small_3d,           lambda t: [number(3.14, 3, t)]                                    ),
    ('add',           small_3d,           lambda t: [small_3d_positive(t)],                 'tensor'        ),
    ('add',           small_3d,           lambda t: [number(0.2, 2, t), small_3d_positive(t)], 'scalar_tensor' ),
    ('sub',           small_3d,           lambda t: [number(3.14, 3, t)],                                   ),
    ('sub',           small_3d,           lambda t: [small_3d_positive(t)],                 'tensor'        ),
    ('mul',           small_3d,           lambda t: [number(3.14, 3, t)],                                   ),
    ('mul',           small_3d,           lambda t: [small_3d_positive(t)],                 'tensor'        ),
    ('div',           small_3d,           lambda t: [number(3.14, 3, t)],                                   ),
    ('div',           small_3d,           lambda t: [small_3d_positive(t)],                 'tensor'        ),
    ('pow',           small_3d,           lambda t: [number(3.14, 3, t)],                                   ),
    ('pow',           small_3d,           lambda t: [small_3d(t).abs_()],                   'tensor'        ),
    ('addbmm',        small_2d,           lambda t: [small_3d(t), small_3d(t)],                             ),
    ('addbmm',        small_2d,           lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar' ),
    ('addbmm',        small_2d,           lambda t: [number(0.5, 3, t), number(0.4, 2, t), small_3d(t), small_3d(t)], 'two_scalars' ),
    ('baddbmm',       small_3d,           lambda t: [small_3d(t), small_3d(t)],                             ),
    ('baddbmm',       small_3d,           lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar' ),
    ('baddbmm',       small_3d,           lambda t: [number(0.5, 3, t), number(0.4, 2, t), small_3d(t), small_3d(t)], 'two_scalars' ),
    ('addcdiv',       small_3d,           lambda t: [small_3d(t), small_3d(t)],                             ),
    ('addcdiv',       small_3d,           lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar' ),
    ('addcmul',       small_3d,           lambda t: [small_3d(t), small_3d(t)],                             ),
    ('addcmul',       small_3d,           lambda t: [number(0.4, 2, t), small_3d(t), small_3d(t)], 'scalar' ),
    ('addmm',         medium_2d,          lambda t: [medium_2d(t), medium_2d(t)],                           ),
    ('addmm',         medium_2d,          lambda t: [number(0.4, 2, t), medium_2d(t), medium_2d(t)], 'scalar' ),
    ('addmm',         medium_2d,          lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_2d(t), medium_2d(t)], 'two_scalars'   ),
    ('addmv',         medium_1d,          lambda t: [medium_2d(t), medium_1d(t)],                           ),
    ('addmv',         medium_1d,          lambda t: [number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'scalar' ),
    ('addmv',         medium_1d,          lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'two_scalars'   ),
    ('addmv',         medium_1d,          lambda t: [medium_2d(t), medium_1d(t)],                           ),
    ('addmv',         medium_1d,          lambda t: [number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'scalar' ),
    ('addmv',         medium_1d,          lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_2d(t), medium_1d(t)], 'two_scalars'   ),
    ('addr',          medium_2d,          lambda t: [medium_1d(t), medium_1d(t)],                           ),
    ('addr',          medium_2d,          lambda t: [number(0.4, 2, t), medium_1d(t), medium_1d(t)], 'scalar' ),
    ('addr',          medium_2d,          lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_1d(t), medium_1d(t)], 'two_scalars'   ),
    ('addr',          medium_2d,          lambda t: [number(0.5, 3, t), number(0.4, 2, t), medium_1d(t), medium_1d(t)], 'two_scalars'   ),
    ('atan2',         medium_2d,          lambda t: [medium_2d(t)],                                         ),
    ('chunk',         medium_2d,          lambda t: [4],                                                    ),
    ('chunk',         medium_2d,          lambda t: [4, 1],                                 'dim'           ),
    ('clamp',         medium_2d,          lambda t: [-0.1, 0.5],                                            ),
    ('clone',         medium_2d,          lambda t: [],                                                     ),
    ('cmax',          medium_2d,          lambda t: [medium_2d(t)],                                         ),
    ('cmin',          medium_2d,          lambda t: [medium_2d(t)],                                         ),
    ('contiguous',    medium_2d,          lambda t: [],                                                     ),
    ('cross',         new_t(M, 3, M),     lambda t: [new_t(M, 3, M)(t)],                                    ),
    ('cumprod',       small_3d,           lambda t: [1],                                                    ),
    ('cumsum',        small_3d,           lambda t: [1],                                                    ),
    ('dim',           small_3d,           lambda t: [],                                                     ),
    ('dist',          small_2d,           lambda t: [small_2d(t)],                                          ),
    ('dist',          small_2d,           lambda t: [small_2d(t), 3],                       '3_norm'        ),
    ('dist',          small_2d,           lambda t: [small_2d(t), 2.5],                     '2.5_norm'      ),
    ('dot',           medium_1d,          lambda t: [medium_1d(t)],                                         ),
    ('element_size',  medium_1d,          lambda t: [],                                                     ),
    ('eq',            small_3d_ones,      lambda t: [small_3d(t)],                                          ),
    ('eq',            small_3d_ones,      lambda t: [small_3d_ones(t)],                     'equal'         ),
    ('ne',            small_3d_ones,      lambda t: [small_3d(t)],                                          ),
    ('ne',            small_3d_ones,      lambda t: [small_3d_ones(t)],                     'equal'         ),
    ('equal',         small_3d_ones,      lambda t: [small_3d_ones(t)],                                     ),
    ('equal',         small_3d_ones,      lambda t: [small_3d(t)],                                          ),
    ('expand',        new_t(M, 1, M),     lambda t: [M, 4, M],                                              ),
    ('expand_as',     new_t(M, 1, M),     lambda t: [new_t(M, 4, M)(t)],                                    ),
    ('fill',          medium_2d,          lambda t: [number(3.14, 3, t)],                                   ),
    ('ge',            medium_2d,          lambda t: [medium_2d(t)],                                         ),
    ('le',            medium_2d,          lambda t: [medium_2d(t)],                                         ),
    ('gt',            medium_2d,          lambda t: [medium_2d(t)],                                         ),
    ('lt',            medium_2d,          lambda t: [medium_2d(t)],                                         ),
    ('is_contiguous', medium_2d,          lambda t: [],                                                     ),
    # TODO: can't check negative case - GPU copy will be contiguous
    ('is_same_size',  medium_2d,          lambda t: [small_3d(t)],                          'negative'      ),
    ('is_same_size',  medium_2d,          lambda t: [medium_2d(t)],                         'positive'      ),
    ('is_set_to',     medium_2d,          lambda t: [medium_2d(t)],                                         ),
    # TODO: positive case
    ('is_size',       medium_2d,          lambda t: [torch.LongStorage((M, M))],                            ),
    ('kthvalue',      small_3d_unique,    lambda t: [3],                                                    ),
    ('kthvalue',      small_3d_unique,    lambda t: [3, 1],                                 'dim'           ),
    ('lerp',          small_3d,           lambda t: [small_3d(t), 0.3],                                     ),
    ('max',           small_3d_unique,    lambda t: [],                                                     ),
    ('max',           small_3d_unique,    lambda t: [1],                                    'dim'           ),
    ('min',           small_3d_unique,    lambda t: [],                                                     ),
    ('min',           small_3d_unique,    lambda t: [1],                                    'dim'           ),
    ('mean',          small_3d,           lambda t: [],                                                     ),
    ('mean',          small_3d,           lambda t: [1],                                    'dim'           ),
    ('mode',          small_3d,           lambda t: [],                                                     ),
    ('mode',          small_3d,           lambda t: [1],                                    'dim'           ),
    ('std',           small_3d,           lambda t: [],                                                     ),
    ('std',           small_3d,           lambda t: [1],                                    'dim'           ),
    ('var',           small_3d,           lambda t: [],                                                     ),
    ('var',           small_3d,           lambda t: [1],                                    'dim'           ),
    ('ndimension',    small_3d,           lambda t: [],                                                     ),
    ('nelement',      small_3d,           lambda t: [],                                                     ),
    ('numel',         small_3d,           lambda t: [],                                                     ),
    ('narrow',        small_3d,           lambda t: [1, 3, 2],                                              ),
    ('nonzero',       small_3d,           lambda t: [],                                                     ),
    ('norm',          small_3d,           lambda t: [],                                                     ),
    ('norm',          small_3d,           lambda t: [3],                                    '3_norm'        ),
    ('norm',          small_3d,           lambda t: [3, 0],                                 '3_norm_dim'    ),
    ('ones',          small_3d,           lambda t: [1, 2, 3, 4, 5],                                        ),
    ('permute',       new_t(1, 2, 3, 4),  lambda t: [2, 1, 3, 0],                                           ),
    ('prod',          small_3d,           lambda t: [],                                                     ),
    ('prod',          small_3d,           lambda t: [1],                                    'dim'           ),
    ('sum',           small_2d,           lambda t: [],                                                     ),
    ('sum',           small_3d,           lambda t: [1],                                    'dim'           ),
    ('renorm',        small_3d,           lambda t: [2, 1, 1],                              '2_norm'        ),
    ('renorm',        small_3d,           lambda t: [1.5, 1, 1],                            '1.5_norm'      ),
    ('repeat',        small_2d,           lambda t: [2, 2, 2],                                              ),
    ('size',          new_t(1, 2, 3, 4),  lambda t: [],                                                     ),
    ('sort',          small_3d_unique,    lambda t: [],                                                     ),
    ('sort',          small_3d_unique,    lambda t: [1],                                    'dim'           ),
    ('sort',          small_3d_unique,    lambda t: [1, True],                              'dim_descending'),
    ('split',         small_3d,           lambda t: [2],                                                    ),
    ('split',         small_3d,           lambda t: [2, 1],                                 'dim'           ),
    ('squeeze',       new_t(1, 2, 1, 4),  lambda t: [],                                                     ),
    ('squeeze',       new_t(1, 2, 1, 4),  lambda t: [2],                                    'dim'           ),
    ('t',             new_t(1, 2),        lambda t: [],                                                     ),
    ('transpose',     new_t(1, 2, 3, 4),  lambda t: [1, 2],                                                 ),
    ('to_list',       small_3d,           lambda t: [],                                                     ),
    ('topk',          small_3d,           lambda t: [2, 1, False, True],                    'dim_sort'      ),
    ('topk',          small_3d,           lambda t: [2, 1, True, True],                     'dim_desc_sort' ),
    ('trace',         medium_2d,          lambda t: [],                                                     ),
    ('tril',          medium_2d,          lambda t: [],                                                     ),
    ('tril',          medium_2d,          lambda t: [2],                                    'positive'      ),
    ('tril',          medium_2d,          lambda t: [-2],                                   'negative'      ),
    ('triu',          medium_2d,          lambda t: [],                                                     ),
    ('triu',          medium_2d,          lambda t: [2],                                    'positive'      ),
    ('triu',          medium_2d,          lambda t: [-2],                                   'negative'      ),
    ('view',          small_3d,           lambda t: [100, 10],                                              ),
    ('view_as',       small_3d,           lambda t: [t(100, 10)],                                           ),
    ('zero',          small_3d,           lambda t: [],                                                     ),
    ('zeros',         small_3d,           lambda t: [1, 2, 3, 4],                                           ),
    ('rsqrt',         lambda t: small_3d(t) + 1,                lambda t: [],                               ),
    ('sinh',          lambda t: small_3d(t).clamp(-1, 1),       lambda t: [],                               ),
    ('tan',           lambda t: small_3d(t).clamp(-1, 1),       lambda t: [],                               ),
]

# TODO: random functions, cat, gather, scatter, index*, masked*, resize, resizeAs, storage_offset, storage, stride, unfold

custom_precision = {
    'addbmm': 1e-4,
    'addmm': 1e-4,
    'rsqrt': 1e-4,
    'cumprod': 1e-4,
}

simple_pointwise = [
    'abs',
    'acos',
    'asin',
    'atan',
    'ceil',
    'cinv',
    'cos',
    'cosh',
    'exp',
    'floor',
    'fmod',
    'frac',
    'log',
    'log1p',
    'neg',
    'remainder',
    'round',
    'sigmoid',
    'sign',
    'sin',
    'sqrt',
    'tanh',
    'trunc',
]
for fn in simple_pointwise:
    tests.append((fn, small_3d, lambda t: []))

def compare_cpu_gpu(tensor_constructor, arg_constructor, fn, t, precision=1e-5):
    def tmp(self):
        cpu_tensor = tensor_constructor(t)
        gpu_tensor = to_gpu(cpu_tensor)
        cpu_args = arg_constructor(t)
        gpu_args = [to_gpu(arg) for arg in cpu_args]
        cpu_result = getattr(cpu_tensor, fn)(*cpu_args)
        try:
            gpu_result = getattr(gpu_tensor, fn)(*gpu_args)
        except RuntimeError as e:
            reason = e.args[0]
            if 'unimplemented data type' in reason:
                raise unittest.SkipTest('unimplemented data type')
            raise
        # If one changes, another should change as well
        self.assertEqual(cpu_tensor, gpu_tensor, precision)
        self.assertEqual(cpu_args, gpu_args, precision)
        # Compare results
        self.assertEqual(cpu_result, gpu_result, precision)
    return tmp

class TestCuda(TestCase):

    def test_autogpu(self):
        if torch.cuda.device_count() > 1:
            x = torch.randn(5, 5).cuda()
            y = torch.randn(5, 5).cuda()
            self.assertEqual(x.getDevice(), 0)
            self.assertEqual(x.getDevice(), 0)
            with torch.cuda.device(1):
                z = torch.randn(5, 5).cuda()
                self.assertEqual(z.getDevice(), 1)
                q = x.add(y)
                self.assertEqual(q.getDevice(), 0)
                w = torch.randn(5, 5).cuda()
                self.assertEqual(w.getDevice(), 1)
            z = z.cuda()
            self.assertEqual(z.getDevice(), 0)

    def test_serialization(self):
        x = torch.randn(5, 5).cuda()
        y = torch.IntTensor(2, 5).fill_(0).cuda()
        q = [x, y, x, y.storage()]
        with tempfile.NamedTemporaryFile() as f:
            torch.save(q, f)
            f.seek(0)
            q_copy = torch.load(f)
        self.assertEqual(q_copy, q, 0)
        q_copy[0].fill_(5)
        self.assertEqual(q_copy[0], q_copy[2], 0)
        self.assertTrue(isinstance(q_copy[0], torch.cuda.DoubleTensor))
        self.assertTrue(isinstance(q_copy[1], torch.cuda.IntTensor))
        self.assertTrue(isinstance(q_copy[2], torch.cuda.DoubleTensor))
        self.assertTrue(isinstance(q_copy[3], torch.cuda.IntStorage))
        q_copy[1].fill_(10)
        self.assertTrue(q_copy[3], torch.cuda.IntStorage(10).fill_(10))

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIs(type(x.float()), torch.FloatTensor)
        self.assertIs(type(x.cuda()), torch.cuda.DoubleTensor)
        self.assertIs(type(x.cuda().float()), torch.cuda.FloatTensor)
        self.assertIs(type(x.cuda().float().cpu()), torch.FloatTensor)
        self.assertIs(type(x.cuda().float().cpu().int()), torch.IntTensor)

        y = x.storage()
        self.assertIs(type(y.float()), torch.FloatStorage)
        self.assertIs(type(y.cuda()), torch.cuda.DoubleStorage)
        self.assertIs(type(y.cuda().float()), torch.cuda.FloatStorage)
        self.assertIs(type(y.cuda().float().cpu()), torch.FloatStorage)
        self.assertIs(type(y.cuda().float().cpu().int()), torch.IntStorage)

    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_type_conversions_same_gpu(self):
        x = torch.randn(5, 5).cuda(1)
        self.assertEqual(x.int().getDevice(), 1)


for decl in tests:
    for t in types:
        tensor = t()
        gpu_tensor = get_gpu_type(t)()
        for inplace in (True, False):
            if len(decl) == 3:
                name, constr, arg_constr = decl
                desc = ''
            elif len(decl) == 4:
                name, constr, arg_constr, desc = decl
            if inplace:
                name = name + '_'
            if not hasattr(tensor, name):
                continue
            if not hasattr(gpu_tensor, name):
                print("Ignoring {}, because it's not implemented by torch.cuda.{}".format(name, gpu_tensor.__class__.__name__))
                continue

            test_name = 'test_' + t.__name__ + '_' + name
            if desc:
                test_name += '_' + desc

            precision = custom_precision.get(name, TestCuda.precision)

            assert not hasattr(TestCase, test_name)
            setattr(TestCuda, test_name, compare_cpu_gpu(constr, arg_constr, name, t, precision))

if __name__ == '__main__':
    unittest.main()
