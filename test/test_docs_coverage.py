import torch
import unittest
import os
import re
import textwrap


path = os.path.dirname(os.path.realpath(__file__))
rstpath = os.path.join(path, '../docs/source/')
pypath = os.path.join(path, '../torch/_torch_docs.py')
r1 = re.compile(r'\.\. autofunction:: (\w*)')
r2 = re.compile(r'\.\. auto(?:method|attribute):: (\w*)')


class TestDocCoverage(unittest.TestCase):

    @staticmethod
    def parse_rst(filename, regex):
        path = os.path.join(os.getenv('DOCS_SRC_DIR', ''), filename)
        if not os.path.exists(path):
            # Try to find the file using a relative path.
            path = os.path.join(rstpath, filename)

        ret = set()
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip()
                name = regex.findall(l)
                if name:
                    ret.add(name[0])
        return ret

    def test_torch(self):
        # TODO: The algorithm here is kind of unsound; we don't assume
        # every identifier in torch.rst lives in torch by virtue of
        # where it lives; instead, it lives in torch because at the
        # beginning of the file we specified automodule.  This means
        # that this script can get confused if you have, e.g., multiple
        # automodule directives in the torch file.  "Don't do that."
        # (Or fix this to properly handle that case.)

        # get symbols documented in torch.rst
        in_rst = self.parse_rst('torch.rst', r1)
        # get symbols in functional.py and _torch_docs.py
        whitelist = {
            # below are some jit functions
            'wait', 'fork', 'parse_type_comment', 'import_ir_module',
            'import_ir_module_from_buffer', 'merge_type_from_type_comment',
            'parse_ir', 'parse_schema',

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
            that doesn't have a docstring or isn't in torch.*. If you just
            removed something from torch.*, please remove it from the whitelist
            in test_docs_coverage.py'''))
        has_docstring -= whitelist
        # https://github.com/pytorch/pytorch/issues/32014
        # The following context manager classes are imported on top leve torch
        # and are referred in docs as torch.no_grad. So we would like to have them
        # included in docs too. has_docstring only contains functions and no classes
        # so adding some them manually here.
        has_docstring |= {'no_grad', 'enable_grad', 'set_grad_enabled'}
        # assert they are equal
        self.assertEqual(
            has_docstring, in_rst,
            textwrap.dedent('''
            The lists of functions documented in torch.rst and in python are different.
            Did you forget to add a new thing to torch.rst, or whitelist things you
            don't want to document?''')
        )

    def test_tensor(self):
        in_rst = self.parse_rst('tensors.rst', r2)
        whitelist = {
            'names', 'unflatten', 'align_as', 'rename_', 'refine_names', 'align_to',
            'has_names', 'rename',
        }
        classes = [torch.FloatTensor, torch.LongTensor, torch.ByteTensor]
        has_docstring = set(x for c in classes for x in dir(c) if not x.startswith('_') and getattr(c, x).__doc__)
        has_docstring -= whitelist
        self.assertEqual(
            has_docstring, in_rst,
            textwrap.dedent('''
            The lists of tensor methods documented in tensors.rst and in python are
            different. Did you forget to add a new thing to tensors.rst, or whitelist
            things you don't want to document?''')
        )


if __name__ == '__main__':
    unittest.main()
