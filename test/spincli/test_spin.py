# Owner(s): ["module: spin"]

import ast
import os
import shutil
import tempfile
from pathlib import Path

from spin.tests.testutil import spin

from torch.testing._internal.common_utils import run_tests, TestCase


PYTORCH_ROOT = Path(__file__).parent.parent.parent


class TestSpin(TestCase):
    def test_autotype(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            here = Path(__file__).parent
            untyped_file_name = "autotype_test_untyped.py"
            source = here / untyped_file_name
            dest = Path(tmp_dir) / untyped_file_name
            shutil.copy(source, dest)

            cwd = os.getcwd()
            os.chdir(PYTORCH_ROOT)
            spin("pyrefly", "infer", dest)
            # Post test code
            os.chdir(cwd)

            with open(dest) as f:
                retyped_contents = f.read()
            retyped_ast = ast.parse(retyped_contents)

            typed_file_name = "autotype_test_typed.py"
            with open(here / typed_file_name) as f:
                typed_contents = f.read()
            typed_ast = ast.parse(typed_contents)

            self.assertEqual(
                ast.dump(typed_ast),
                ast.dump(retyped_ast),
            )


if __name__ == "__main__":
    run_tests()
