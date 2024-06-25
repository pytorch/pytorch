# mypy: allow-untyped-defs
# Owner(s): ["module: typing"]

import doctest
import inspect
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_utils import run_tests, set_cwd, TestCase

try:
    import mypy.api

    HAVE_MYPY = True
except ImportError:
    HAVE_MYPY = False


def get_examples_from_docstring(docstr):
    """
    Extracts all runnable python code from the examples
    in docstrings; returns a list of lines.
    """
    examples = doctest.DocTestParser().get_examples(docstr)
    return [f"    {l}" for e in examples for l in e.source.splitlines()]


def get_all_examples():
    """get_all_examples() -> str

    This function grabs (hopefully all) examples from the torch documentation
    strings and puts them in one nonsensical module returned as a string.
    """
    blocklist = {
        "_np",
    }
    allexamples = ""

    example_file_lines = [
        "import torch",
        "import torch.nn.functional as F",
        "import math",
        "import numpy",
        "import io",
        "import itertools",
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
        if docstr and fname not in blocklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append(f"\n\ndef example_torch_{fname}():")
                example_file_lines += e

    for fname in dir(torch.Tensor):
        fn = getattr(torch.Tensor, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blocklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append(f"\n\ndef example_torch_tensor_{fname}():")
                example_file_lines += e

    return "\n".join(example_file_lines)


class TestTypeHints(TestCase):
    @unittest.skipIf(not HAVE_MYPY, "need mypy")
    def test_doc_examples(self):
        """
        Run documentation examples through mypy.
        """
        fn = Path(__file__).resolve().parent / "generated_type_hints_smoketest.py"
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
                    os.path.join(tmp_dir, "torch"),
                    target_is_directory=True,
                )
            except OSError:
                raise unittest.SkipTest("cannot symlink") from None
            repo_rootdir = Path(__file__).resolve().parent.parent
            # TODO: Would be better not to chdir here, this affects the
            # entire process!
            with set_cwd(str(repo_rootdir)):
                (stdout, stderr, result) = mypy.api.run(
                    [
                        "--cache-dir=.mypy_cache/doc",
                        "--no-strict-optional",  # needed because of torch.lu_unpack, see gh-36584
                        str(fn),
                    ]
                )
            if result != 0:
                self.fail(f"mypy failed:\n{stderr}\n{stdout}")


if __name__ == "__main__":
    run_tests()
