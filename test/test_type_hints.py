from __future__ import print_function
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests
import tempfile
import torch
import re
import os
import sys
import subprocess
import inspect

try:
    import mypy  # noqa: F401
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
    blacklist = {
        "_np",
        "refine_names",
        "rename",
        "names",
        "align_as",
        "align_to",
        "unflatten",
    }
    allexamples = ""

    example_file_lines = [
        "import torch",
        "import torch.nn.functional as F",
        "import math  # type: ignore",  # mypy complains about floats where SupportFloat is expected
        "import numpy  # type: ignore",
        "import io  # type: ignore",
        "import itertools  # type: ignore",
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
        fn = os.path.join(os.path.dirname(__file__), 'generated_type_hints_smoketest.py')
        with open(fn, "w") as f:
            print(get_all_examples(), file=f)

        # OK, so here's the deal.  mypy treats installed packages
        # and local modules differently: if a package is installed,
        # mypy will refuse to use modules from that package for type
        # checking unless the module explicitly says that it supports
        # type checking. (Reference:
        # https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports
        # )
        #
        # Now, PyTorch doesn't support typechecking, and we shouldn't
        # claim that it supports typechecking (it doesn't.) However, not
        # claiming we support typechecking is bad for this test, which
        # wants to use the partial information we get from the bits of
        # PyTorch which are typed to check if it typechecks.  And
        # although mypy will work directly if you are working in source,
        # some of our tests involve installing PyTorch and then running
        # its tests.
        #
        # The guidance we got from Michael Sullivan and Joshua Oreman,
        # and also independently developed by Thomas Viehmann,
        # is that we should create a fake directory and add symlinks for
        # the packages that should typecheck.  So that is what we do
        # here.
        #
        # If you want to run mypy by hand, and you run from PyTorch
        # root directory, it should work fine to skip this step (since
        # mypy will preferentially pick up the local files first).  The
        # temporary directory here is purely needed for CI.  For this
        # reason, we also still drop the generated file in the test
        # source folder, for ease of inspection when there are failures.
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                os.symlink(
                    os.path.dirname(torch.__file__),
                    os.path.join(tmp_dir, 'torch'),
                    target_is_directory=True
                )
            except OSError:
                raise unittest.SkipTest('cannot symlink')
            try:
                subprocess.run([
                    sys.executable,
                    '-mmypy',
                    '--follow-imports', 'silent',
                    '--check-untyped-defs',
                    os.path.abspath(fn)],
                    cwd=tmp_dir,
                    check=True)
            except subprocess.CalledProcessError as e:
                raise AssertionError("mypy failed.  Look above this error for mypy's output.")

    @unittest.skipIf(sys.version_info[0] == 2, "no type hints for Python 2")
    @unittest.skipIf(not HAVE_MYPY, "need mypy")
    def test_type_hint_examples(self):
        """
        Runs mypy over all the test examples present in
        `type_hint_tests` directory.
        """
        test_path = os.path.dirname(os.path.realpath(__file__))
        examples_folder = os.path.join(test_path, "type_hint_tests")
        examples = os.listdir(examples_folder)
        for example in examples:
            try: 
                example_path = os.path.join(examples_folder, example)
                subprocess.run([ 
                    sys.executable, 
                    '-mmypy', 
                    '--follow-imports', 'silent', 
                    '--check-untyped-defs', 
                    example_path],  
                    check=True) 
            except subprocess.CalledProcessError as e: 
                raise AssertionError("mypy failed for example {}.  Look above this error for mypy's output.".format(example))

if __name__ == '__main__':
    run_tests()
