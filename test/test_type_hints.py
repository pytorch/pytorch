# mypy: allow-untyped-defs
# Owner(s): ["module: typing"]

import doctest
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.testing._internal.common_utils import (
    run_tests,
    set_cwd,
    TestCase,
    xfailIfS390X,
)


HAVE_PYREFLY = importlib.util.find_spec("pyrefly") is not None


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
        "_InputT",
    }

    example_file_lines = [
        "# pyrefly: allow-untyped-defs",
        "",
        "import math",
        "import io",
        "import itertools",
        "",
        "from typing import Any, ClassVar, Generic, List, Tuple, Union",
        "from typing_extensions import Literal, get_origin, TypeAlias",
        "T: TypeAlias = object",
        "",
        "import numpy",
        "",
        "import torch",
        "import torch.nn.functional as F",
        "",
        "from typing_extensions import ParamSpec as _ParamSpec",
        "ParamSpec = _ParamSpec",
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
                example_file_lines.append(f"\n\ndef example_torch_{fname}() -> None:")
                example_file_lines += e

    for fname in dir(torch.Tensor):
        fn = getattr(torch.Tensor, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blocklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append(
                    f"\n\ndef example_torch_tensor_{fname}() -> None:"
                )
                example_file_lines += e

    return "\n".join(example_file_lines)


class TestTypeHints(TestCase):
    # when this test fails on s390x, it also leads to OOM on test reruns
    @xfailIfS390X
    @unittest.skipIf(not HAVE_PYREFLY, "need pyrefly")
    def test_doc_examples(self):
        """
        Run documentation examples through pyrefly.
        """
        fn = Path(__file__).resolve().parent / "generated_type_hints_smoketest.py"
        fn.write_text(get_all_examples())

        # OK, so here's the deal.  Type checkers like mypy and pyrefly treat
        # installed packages and local modules differently: if a package is
        # installed, they will refuse to use modules from that package for type
        # checking unless the module explicitly says that it supports
        # type checking.
        #
        # Now, PyTorch doesn't support typechecking, and we shouldn't
        # claim that it supports typechecking (it doesn't.) However, not
        # claiming we support typechecking is bad for this test, which
        # wants to use the partial information we get from the bits of
        # PyTorch which are typed to check if it typechecks.  And
        # although type checkers will work directly if you are working in source,
        # some of our tests involve installing PyTorch and then running
        # its tests.
        #
        # The guidance is that we should create a fake directory and add
        # symlinks for the packages that should typecheck.  So that is what
        # we do here.
        #
        # If you want to run pyrefly by hand, and you run from PyTorch
        # root directory, it should work fine to skip this step (since
        # pyrefly will preferentially pick up the local files first).  The
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
            # Use permissive config for documentation examples (equivalent to mypy's --no-strict-optional)
            permissive_config = repo_rootdir / "pyrefly-permissive.toml"

            # TODO: Would be better not to chdir here, this affects the
            # entire process!
            with set_cwd(str(repo_rootdir)):
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pyrefly",
                            "check",
                            "--config",
                            str(permissive_config),
                            str(fn),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,  # 2 minute timeout
                    )
                    stdout = result.stdout
                    stderr = result.stderr
                    exit_code = result.returncode
                except subprocess.TimeoutExpired:
                    self.fail("pyrefly timed out")
                except subprocess.SubprocessError as e:
                    self.fail(f"pyrefly subprocess error: {e}")

            if exit_code != 0:
                self.fail(f"pyrefly failed:\n{stderr}\n{stdout}")


if __name__ == "__main__":
    run_tests()
