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
r2 = re.compile(r'\.\. auto(?:method|attribute):: (\w*)')


class TestDocCoverage(unittest.TestCase):

    @staticmethod
    def parse_rst(filename, regex):
        filename = os.path.join(rstpath, filename)
        ret = set()
        with open(filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                name = regex.findall(l)
                if name:
                    ret.add(name[0])
        return ret

    def test_torch(self):
        # get symbols documented in torch.rst
        whitelist = {
            'set_printoptions', 'get_rng_state', 'is_storage', 'initial_seed',
            'set_default_tensor_type', 'load', 'save', 'set_default_dtype',
            'is_tensor', 'compiled_with_cxx11_abi', 'set_rng_state',
            'manual_seed'
        }
        in_rst = self.parse_rst('torch.rst', r1) - whitelist
        # get symbols in functional.py and _torch_docs.py
        whitelist2 = {'product', 'inf', 'math', 'reduce', 'warnings', 'torch', 'annotate'}
        has_docstring = set(a for a in dir(torch) if getattr(torch, a).__doc__ and not a.startswith('_'))
        nn_functional = set(dir(torch.nn.functional))
        has_docstring -= nn_functional | whitelist2
        # assert they are equal
        for p in in_rst:
            self.assertIn(p, has_docstring, 'in torch.rst but not in python')
        for p in has_docstring:
            self.assertIn(p, in_rst, 'in python but not in torch.rst')

    def test_tensor(self):
        in_rst = self.parse_rst('tensors.rst', r2)
        classes = [torch.FloatTensor, torch.LongTensor, torch.ByteTensor]
        has_docstring = set(x for c in classes for x in dir(c) if not x.startswith('_') and getattr(c, x).__doc__)
        for p in in_rst:
            self.assertIn(p, has_docstring, 'in tensors.rst but not in python')
        for p in has_docstring:
            self.assertIn(p, in_rst, 'in python but not in tensors.rst')


if __name__ == '__main__':
    unittest.main()
