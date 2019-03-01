import torch
import unittest
import os
import re
import ast
import _ast


path = os.path.dirname(os.path.realpath(__file__))
rstpath = os.path.join(path, '../docs/source/')
pypath = os.path.join(path, '../torch/_torch_docs.py')
r1 = re.compile(r'\.\. autofunction:: (\w*)')
r2 = re.compile(r'\.\. auto(method|attribute):: (\w*)')


class TestDocCoverage(unittest.TestCase):

    def test_torch(self):
        # get symbols documented in torch.rst
        whitelist = [
            'set_printoptions', 'get_rng_state', 'is_storage', 'initial_seed',
            'set_default_tensor_type', 'load', 'save', 'set_default_dtype',
            'is_tensor', 'compiled_with_cxx11_abi', 'set_rng_state',
            'manual_seed'
        ]
        everything = set()
        filename = os.path.join(rstpath, 'torch.rst')
        with open(filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                name = r1.findall(l)
                if name:
                    everything.add(name[0])
        everything -= set(whitelist)
        # get symbols in functional.py and _torch_docs.py
        whitelist2 = ['product', 'inf', 'math', 'reduce', 'warnings', 'torch', 'annotate']
        everything2 = set()
        with open(pypath, 'r') as f:
            body = ast.parse(f.read()).body
            for i in body:
                if not isinstance(i, _ast.Expr):
                    continue
                i = i.value
                if not isinstance(i, _ast.Call):
                    continue
                if i.func.id != 'add_docstr':
                    continue
                i = i.args[0]
                if i.value.id != 'torch':
                    continue
                i = i.attr
                everything2.add(i)
            for p in dir(torch.functional):
                if not p.startswith('_') and p[0].islower():
                    everything2.add(p)
            everything2 -= set(whitelist2)
        # assert they are equal
        for p in everything:
            self.assertIn(p, everything2, 'in torch.rst but not in python')
        for p in everything2:
            self.assertIn(p, everything, 'in python but not in torch.rst')

    def test_tensor(self):
        # get symbols documented in tensors.rst
        whitelist = []
        everything = set()
        filename = os.path.join(rstpath, 'tensors.rst')
        with open(filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                name = r2.findall(l)
                if name:
                    everything.add(name[0][1])
        everything -= set(whitelist)
        # get symbols in tensor.py and _tensor_docs.py
        classes = [torch.FloatTensor, torch.LongTensor, torch.ByteTensor]
        whitelist2 = []
        everything2 = set(x for c in classes for x in dir(c) if not x.startswith('_') and getattr(c, x).__doc__)
        everything2 -= set(whitelist2)
        # assert they are equal
        for p in everything:
            self.assertIn(p, everything2, 'in tensors.rst but not in python')
        for p in everything2:
            self.assertIn(p, everything, 'in python but not in tensors.rst')


if __name__ == '__main__':
    unittest.main()
