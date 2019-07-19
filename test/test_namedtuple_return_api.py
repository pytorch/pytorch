import os
import re
import yaml
import unittest
import textwrap
import torch
from collections import namedtuple


path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, '../aten/src/ATen/native/native_functions.yaml')
all_operators_with_namedtuple_return = {
    'max', 'min', 'median', 'mode', 'kthvalue', 'svd', 'symeig', 'eig',
    'qr', 'geqrf', 'solve', 'slogdet', 'sort', 'topk', 'gels',
    'triangular_solve'
}


class TestNamedTupleAPI(unittest.TestCase):

    def test_native_functions_yaml(self):
        operators_found = set()
        regex = re.compile(r"^(\w*)\(")
        file = open(aten_native_yaml, 'r')
        for f in yaml.load(file.read()):
            f = f['func']
            ret = f.split('->')[1].strip()
            name = regex.findall(f)[0]
            if name in all_operators_with_namedtuple_return:
                operators_found.add(name)
                continue
            if name.endswith('_backward') or name.endswith('_forward'):
                continue
            if not ret.startswith('('):
                continue
            ret = ret[1:-1].split(',')
            for r in ret:
                r = r.strip()
                self.assertEqual(len(r.split()), 1,
                                 'only whitelisted operators are allowed to have named return type, got ' + name)
        file.close()
        self.assertEqual(all_operators_with_namedtuple_return, operators_found, textwrap.dedent("""
        Some elements in the `all_operators_with_namedtuple_return` of test_namedtuple_return_api.py
        could not be found. Do you forget to update test_namedtuple_return_api.py after renaming some
        operator?
        """))

    def test_namedtuple_return(self):
        a = torch.randn(5, 5)

        op = namedtuple('op', ['operators', 'input', 'names', 'hasout'])
        operators = [
            op(operators=['max', 'min', 'median', 'mode', 'sort', 'topk'], input=(0,),
               names=('values', 'indices'), hasout=True),
            op(operators=['kthvalue'], input=(1, 0),
               names=('values', 'indices'), hasout=True),
            op(operators=['svd'], input=(), names=('U', 'S', 'V'), hasout=True),
            op(operators=['slogdet'], input=(), names=('sign', 'logabsdet'), hasout=False),
            op(operators=['qr'], input=(), names=('Q', 'R'), hasout=True),
            op(operators=['solve'], input=(a,), names=('solution', 'LU'), hasout=True),
            op(operators=['geqrf'], input=(), names=('a', 'tau'), hasout=True),
            op(operators=['symeig', 'eig'], input=(True,), names=('eigenvalues', 'eigenvectors'), hasout=True),
            op(operators=['triangular_solve'], input=(a,), names=('solution', 'cloned_coefficient'), hasout=True),
            op(operators=['gels'], input=(a,), names=('solution', 'QR'), hasout=True),
        ]

        for op in operators:
            for f in op.operators:
                ret = getattr(a, f)(*op.input)
                for i, name in enumerate(op.names):
                    self.assertIs(getattr(ret, name), ret[i])
                if op.hasout:
                    ret1 = getattr(torch, f)(a, *op.input, out=tuple(ret))
                    for i, name in enumerate(op.names):
                        self.assertIs(getattr(ret, name), ret[i])

        all_covered_operators = set([x for y in operators for x in y.operators])

        self.assertEqual(all_operators_with_namedtuple_return, all_covered_operators, textwrap.dedent('''
        The set of covered operators does not match the `all_operators_with_namedtuple_return` of
        test_namedtuple_return_api.py. Do you forget to add test for that operator?
        '''))


if __name__ == '__main__':
    unittest.main()
