import pathlib

import pytest

from spin.tests.conftest import dir_switcher


PYTORCH_ROOT = pathlib.Path(__file__).parent.parent.parent

breakpoint()


@pytest.fixture()
def main_pkg():
    yield from dir_switcher(PYTORCH_ROOT)
