import torch
import unittest
import os
import re


path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, '../docs/source/')

r1 = re.compile(r'\.\. autofunction:: (\w*)')

class TestTorchDocCoverage(unittest.TestCase):

    def test_torch(self):
        whitelist = []
        everything = set(whitelist)
        filename = os.path.join(path, 'torch.rst')
        with open(filename, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                name = r1.findall(l)
                if name:
                    name = name[0]
                else:
                    continue
                everything.add(name)
        for p in dir(torch):
            if p.startswith('_') or p[0].isupper() or p.endswith('_'):
                continue
            self.assertIn(p, everything)


if __name__ == '__main__':
    unittest.main()
