import os
import pathlib

import pytest


def dir_switcher(path):
    # Pre-test code
    cwd = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        # Post test code
        os.chdir(cwd)


PYTORCH_ROOT = pathlib.Path(__file__).parent.parent.parent


@pytest.fixture()
def main_pkg():
    yield from dir_switcher(PYTORCH_ROOT)
