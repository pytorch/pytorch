import importlib
import importlib.metadata
import os
import pathlib
import subprocess

import pytest

import numpy as np
import numpy._core.include
import numpy._core.lib.pkgconfig
from numpy.testing import IS_EDITABLE, IS_INSTALLED, IS_WASM, NUMPY_ROOT

INCLUDE_DIR = NUMPY_ROOT / '_core' / 'include'
PKG_CONFIG_DIR = NUMPY_ROOT / '_core' / 'lib' / 'pkgconfig'


@pytest.mark.skipif(not IS_INSTALLED, reason="`numpy-config` not expected to be installed")
@pytest.mark.skipif(IS_WASM, reason="wasm interpreter cannot start subprocess")
class TestNumpyConfig:
    def check_numpyconfig(self, arg):
        p = subprocess.run(['numpy-config', arg], capture_output=True, text=True)
        p.check_returncode()
        return p.stdout.strip()

    def test_configtool_version(self):
        stdout = self.check_numpyconfig('--version')
        assert stdout == np.__version__

    def test_configtool_cflags(self):
        stdout = self.check_numpyconfig('--cflags')
        assert f'-I{os.fspath(INCLUDE_DIR)}' in stdout

    def test_configtool_pkgconfigdir(self):
        stdout = self.check_numpyconfig('--pkgconfigdir')
        assert pathlib.Path(stdout) == PKG_CONFIG_DIR.resolve()


@pytest.mark.skipif(not IS_INSTALLED, reason="numpy must be installed to check its entrypoints")
def test_pkg_config_entrypoint():
    (entrypoint,) = importlib.metadata.entry_points(group='pkg_config', name='numpy')
    assert entrypoint.value == numpy._core.lib.pkgconfig.__name__


@pytest.mark.skipif(not IS_INSTALLED, reason="numpy.pc is only available when numpy is installed")
@pytest.mark.skipif(IS_EDITABLE, reason="editable installs don't have a numpy.pc")
def test_pkg_config_config_exists():
    assert PKG_CONFIG_DIR.joinpath('numpy.pc').is_file()
