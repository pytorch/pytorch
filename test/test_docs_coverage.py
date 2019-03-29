import torch
import unittest
import os
import re
import ast
import _ast
import textwrap


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
        in_rst = self.parse_rst('torch.rst', r1)
        # get symbols in functional.py and _torch_docs.py
        whitelist = {
            # below are some jit functions
            'wait', 'fork', 'parse_type_comment', 'import_ir_module',
            'import_ir_module_from_buffer', 'merge_type_from_type_comment',
            'parse_ir',

            # below are symbols mistakely binded to torch.*, but should
            # go to torch.nn.functional.* instead
            'avg_pool1d', 'conv_transpose2d', 'conv_transpose1d', 'conv3d',
            'relu_', 'pixel_shuffle', 'conv2d', 'selu_', 'celu_', 'threshold_',
            'cosine_similarity', 'rrelu_', 'conv_transpose3d', 'conv1d', 'pdist',
            'adaptive_avg_pool1d', 'conv_tbc'
        }
        has_docstring = set(
            a for a in dir(torch)
            if getattr(torch, a).__doc__ and not a.startswith('_') and
            'function' in type(getattr(torch, a)).__name__)
        self.assertEqual(
            has_docstring & whitelist, whitelist,
            textwrap.dedent('''
            The whitelist in test_docs_coverage.py contains something
            that don't have docstring or not in torch.*. If you just
            removed something from torch.*, please remove it from whiltelist
            in test_docs_coverage.py'''))
        has_docstring -= whitelist
        # assert they are equal
        self.assertEqual(
            has_docstring, in_rst,
            textwrap.dedent('''
            List of functions documented in torch.rst and in python are different.
            Do you forget to add new thing to torch.rst, or whitelist things you
            don't want to document?''')
        )

    def test_tensor(self):
        in_rst = self.parse_rst('tensors.rst', r2)
        classes = [torch.FloatTensor, torch.LongTensor, torch.ByteTensor]
        has_docstring = set(x for c in classes for x in dir(c) if not x.startswith('_') and getattr(c, x).__doc__)
        self.assertEqual(
            has_docstring, in_rst,
            textwrap.dedent('''
            List of tensor methods documented in tensor.rst and in python are
            different. Do you forget to add new thing to tensor.rst, or whitelist
            things you don't want to document?''')
        )


if __name__ == '__main__':
    unittest.main()
