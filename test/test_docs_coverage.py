import torch
import unittest
import os
import re


path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../docs/source/')

r1 = re.compile(r'\.\. autofunction:: (\w*)')

class TestTorchDocCoverage(unittest.TestCase):

    def test_torch(self):
        whitelist = [
            'all', 'any', 'as_strided', 'autograd', 'backends', 'clamp_max',
            'clamp_min', 'complex128', 'complex32', 'complex64', 'cpp', 'cuda',
            'default_generator', 'device', 'distributed', 'distributions',
            'float', 'double',
        ]
        everything = set(whitelist)
        filename = os.path.join(path, 'torch.rst')
        with open(filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                name = r1.findall(l)
                if name:
                    everything.add(name[0])
        for p in everything:
            self.assertIn(p, dir(torch))
        for p in dir(torch):
            if p.startswith('_') or p[0].isupper() or p.endswith('_'):
                continue
            self.assertIn(p, everything)


if __name__ == '__main__':
    unittest.main()
