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

try:
    import mypy
    HAVE_MYPY = True
except ImportError:
    HAVE_MYPY = False


def get_examples_from_docstring(docstr):
    """
    Extracts all runnable python code from the examples
    in docstrings; returns a list of lines.
    """
    # TODO: Figure out if there's a way to use doctest directly to
    # implement this
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
                beginning += m.group(1)
            else:
                beginning += l
        else:
            m = exampleline_re.match(l)
            if m:
                beginning += m.group(1)
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
        # for requires_grad_ example
        # NB: We are parsing this file as Python 2, so we must use
        # Python 2 type annotation syntax
        "def preprocess(inp):",
        "    # type: (torch.Tensor) -> torch.Tensor",
        "    return inp",
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
    @unittest.skipIf(sys.version_info[0] == 2, "no type hints for Python 2")
    @unittest.skipIf(not HAVE_MYPY, "need mypy")
    def test_doc_examples(self):
        """
        Run documentation examples through mypy.
        """
        fn = os.path.join('test', 'data', 'type_hints_smoketest.py')
        with open(fn, "w") as f:
            print(get_all_examples(), file=f)
        try:
            # TODO: I'm not sure we actually need to start a new process
            # for mypy itself haha
            result = subprocess.run([
                sys.executable,
                '-mmypy',
                '--follow-imports', 'silent',
                '--check-untyped-defs',
                fn],
                check=True)
        except subprocess.CalledProcessError as e:
            raise AssertionError("mypy failed.  Look above this error for mypy's output.")


if __name__ == '__main__':
    run_tests()
