# Owner(s): ["module: spin"]

import shutil
from pathlib import Path

from spin.tests.testutil import spin

from torch.testing._internal.common_utils import run_tests


def test_autotype(main_pkg, tmp_path):
    here = Path(__file__).parent

    untyped_file_name = "autotype_test_untyped.py"
    source = here / untyped_file_name
    dest = tmp_path / untyped_file_name
    shutil.copy(source, dest)
    spin("pyrefly", "infer", dest)
    with open(dest) as f:
        retyped_contents = f.read()

    typed_file_name = "autotype_test_typed.py"
    with open(here / typed_file_name) as f:
        typed_contents = f.read()

    assert typed_contents == retyped_contents


if __name__ == "__main__":
    run_tests()
