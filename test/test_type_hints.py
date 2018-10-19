from __future__ import print_function
import unittest
from common_utils import TestCase, run_tests, download_file
import tempfile
import torch
import re
import os
import sys
import subprocess
import inspect


def get_examples_from_docstring(docstr):
    """
Extracts all runnable python code from the examples
in docstrings an returns a list of lines.
"""
    example_file_lines = []
    # the detection is a bit hacky because there isn't a nice way of detecting
    # where multiline commands end. Thus we keep track of how far we got in beginning
    # and continue to add lines until we have a compileable Python statement.
    exampleline_re = re.compile(r"^\s+(?:>>>|\.\.\.) (.*)$")
    beginning = ""
    for l in docstr.split('\n'):
        if beginning:
            m = exampleline_re.match(l)
            if m:
                beginning += m[1]
            else:
                beginning += l
        else:
            m = exampleline_re.match(l)
            if m:
                beginning += m[1]
        if beginning:
            complete = True
            try:
                compile(beginning, "", "exec")
            except SyntaxError:
                complete = False
            if complete:
                # found one
                example_file_lines += beginning.split('\n')
                beginning = ""
            else:
                beginning += "\n"
    return ['    ' + l for l in example_file_lines]


def get_all_examples():
    """get_all_examples() -> str
This function grabs (hopefully all) examples from the torch documentation
strings and puts them in one nonsensical module returned as a string.
    """
    blacklist = {"load", "save", "_np"}
    allexamples = ""

    example_file_lines = [
        "import torch",
        "import torch.nn.functional as F",
        "import math  # type: ignore",  # mypy complains about floats where SupportFloat is expected
        "import numpy  # type: ignore",
        "",
        "def preprocess(inp : torch.Tensor) -> torch.Tensor:",  # for requires_grad_ example
        "  return inp",
    ]

    for fname in dir(torch):
        fn = getattr(torch, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blacklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append("\n\ndef example_torch_{}():".format(fname))
                example_file_lines += e

    for fname in dir(torch.Tensor):
        fn = getattr(torch.Tensor, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blacklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append("\n\ndef example_torch_tensor_{}():".format(fname))
                example_file_lines += e

    return "\n".join(example_file_lines)


class TestTypeHints(TestCase):
    def test_doc_examples(self):
        """run documentation examples through mypy.
mypy can be picky about its environment, so we need
to set up a temporary dir and link the torch module
into it"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fn = os.path.join(tmp_dir, 'test.py')
            curdir = os.getcwd()
            os.chdir(tmp_dir)
            with open(fn, "w") as f:
                print(get_all_examples(), file=f)
            try:
                os.symlink(os.path.dirname(torch.__file__), './torch', target_is_directory=True)
            except OSError:
                raise unittest.SkipTest('cannot symlink')
            try:
                result = subprocess.run(['python3', '-mmypy', '--follow-imports',
                                         'silent', '--check-untyped-defs', 'test.py'],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except subprocess.CalledProcessError as e:
                raise AssertionError("mypy failed" + "\n\n" + e.output.decode())
            os.chdir(curdir)

if __name__ == '__main__':
    run_tests()
