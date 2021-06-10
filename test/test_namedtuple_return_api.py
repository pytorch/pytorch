import os
import re
import yaml
import textwrap
import torch

from torch.testing._internal.common_utils import TestCase, run_tests
from collections import namedtuple


path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')
all_operators_with_namedtuple_return = {
    'max', 'min', 'median', 'nanmedian', 'mode', 'kthvalue', 'svd', 'symeig', 'eig',
    'qr', 'geqrf', 'solve', 'slogdet', 'sort', 'topk', 'lstsq', 'linalg_inv_ex',
    'triangular_solve', 'cummax', 'cummin', 'linalg_eigh', "_unpack_dual", 'linalg_qr',
    '_svd_helper', 'linalg_svd', 'linalg_slogdet', 'fake_quantize_per_tensor_affine_cachemask',
    'fake_quantize_per_channel_affine_cachemask', 'linalg_lstsq', 'linalg_eig', 'linalg_cholesky_ex',
    'frexp', 'lu_unpack'
}


class TestNamedTupleAPI(TestCase):

    def test_native_functions_yaml(self):
        operators_found = set()
        regex = re.compile(r"^(\w*)(\(|\.)")
        file = open(aten_native_yaml, 'r')
        for f in yaml.safe_load(file.read()):
            f = f['func']
            ret = f.split('->')[1].strip()
            name = regex.findall(f)[0][0]
            if name in all_operators_with_namedtuple_return:
                operators_found.add(name)
                continue
            if '_backward' in name or name.endswith('_forward'):
                continue
            if not ret.startswith('('):
                continue
            if ret == '()':
                continue
            ret = ret[1:-1].split(',')
            for r in ret:
                r = r.strip()
                self.assertEqual(len(r.split()), 1,
                                 'only allowlisted operators are allowed to have named return type, got ' + name)
        file.close()
        self.assertEqual(all_operators_with_namedtuple_return, operators_found, textwrap.dedent("""
        Some elements in the `all_operators_with_namedtuple_return` of test_namedtuple_return_api.py
        could not be found. Do you forget to update test_namedtuple_return_api.py after renaming some
        operator?
        """))

    def test_namedtuple_return(self):
        a = torch.randn(5, 5)
        per_channel_scale = torch.randn(5)
        per_channel_zp = torch.zeros(5, dtype=torch.int64)

        op = namedtuple('op', ['operators', 'input', 'names', 'hasout'])
        operators = [
            op(operators=['max', 'min', 'median', 'nanmedian', 'mode', 'sort', 'topk', 'cummax', 'cummin'], input=(0,),
               names=('values', 'indices'), hasout=True),
            op(operators=['kthvalue'], input=(1, 0),
               names=('values', 'indices'), hasout=True),
            op(operators=['svd', '_svd_helper'], input=(), names=('U', 'S', 'V'), hasout=True),
            op(operators=['linalg_svd'], input=(), names=('U', 'S', 'Vh'), hasout=True),
            op(operators=['slogdet'], input=(), names=('sign', 'logabsdet'), hasout=False),
            op(operators=['qr', 'linalg_qr'], input=(), names=('Q', 'R'), hasout=True),
            op(operators=['solve'], input=(a,), names=('solution', 'LU'), hasout=True),
            op(operators=['geqrf'], input=(), names=('a', 'tau'), hasout=True),
            op(operators=['symeig', 'eig'], input=(True,), names=('eigenvalues', 'eigenvectors'), hasout=True),
            op(operators=['triangular_solve'], input=(a,), names=('solution', 'cloned_coefficient'), hasout=True),
            op(operators=['lstsq'], input=(a,), names=('solution', 'QR'), hasout=True),
            op(operators=['linalg_eig'], input=(), names=('eigenvalues', 'eigenvectors'), hasout=True),
            op(operators=['linalg_eigh'], input=("L",), names=('eigenvalues', 'eigenvectors'), hasout=True),
            op(operators=['linalg_slogdet'], input=(), names=('sign', 'logabsdet'), hasout=True),
            op(operators=['linalg_cholesky_ex'], input=(), names=('L', 'info'), hasout=True),
            op(operators=['linalg_inv_ex'], input=(), names=('inverse', 'info'), hasout=True),
            op(operators=['fake_quantize_per_tensor_affine_cachemask'],
               input=(0.1, 0, 0, 255), names=('output', 'mask',), hasout=False),
            op(operators=['fake_quantize_per_channel_affine_cachemask'],
               input=(per_channel_scale, per_channel_zp, 1, 0, 255),
               names=('output', 'mask',), hasout=False),
            op(operators=['_unpack_dual'], input=(0,), names=('primal', 'tangent'), hasout=False),
            op(operators=['linalg_lstsq'], input=(a,), names=('solution', 'residuals', 'rank', 'singular_values'), hasout=False),
            op(operators=['frexp'], input=(), names=('mantissa', 'exponent'), hasout=True),
            op(operators=['lu_unpack'],
               input=(torch.tensor([3, 2, 1, 4, 5], dtype=torch.int32), True, True),
               names=('P', 'L', 'U'), hasout=True),
        ]

        def get_func(f):
            "Return either torch.f or torch.linalg.f, where 'f' is a string"
            mod = torch
            if f.startswith('linalg_'):
                mod = torch.linalg
                f = f[7:]
            if f.startswith('_'):
                mod = torch._VF
            return getattr(mod, f, None)

        def check_namedtuple(tup, names):
            "Check that the namedtuple 'tup' has the given names"
            for i, name in enumerate(names):
                self.assertIs(getattr(tup, name), tup[i])

        for op in operators:
            for f in op.operators:
                # 1. check the namedtuple returned by calling torch.f
                func = get_func(f)
                if func:
                    ret1 = func(a, *op.input)
                    check_namedtuple(ret1, op.names)
                #
                # 2. check the out= variant, if it exists
                if func and op.hasout:
                    ret2 = func(a, *op.input, out=tuple(ret1))
                    check_namedtuple(ret2, op.names)
                #
                # 3. check the Tensor.f method, if it exists
                meth = getattr(a, f, None)
                if meth:
                    ret3 = meth(*op.input)
                    check_namedtuple(ret3, op.names)

        all_covered_operators = set([x for y in operators for x in y.operators])

        self.assertEqual(all_operators_with_namedtuple_return, all_covered_operators, textwrap.dedent('''
        The set of covered operators does not match the `all_operators_with_namedtuple_return` of
        test_namedtuple_return_api.py. Do you forget to add test for that operator?
        '''))


if __name__ == '__main__':
    run_tests()
