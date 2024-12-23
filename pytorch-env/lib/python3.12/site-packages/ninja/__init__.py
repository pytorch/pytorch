# -*- coding: utf-8 -*-
import os
import platform
import subprocess
import sys

from ._version import version as __version__

__all__ = ["__version__", "DATA", "BIN_DIR", "ninja"]


def __dir__():
    return __all__


try:
    from .ninja_syntax import Writer, escape, expand
except ImportError:
    # Support importing `ninja_syntax` from the source tree
    if not os.path.exists(
            os.path.join(os.path.dirname(__file__), 'ninja_syntax.py')):
        sys.path.insert(0, os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../Ninja-src/misc')))
    from ninja_syntax import Writer, escape, expand  # noqa: F401

DATA = os.path.join(os.path.dirname(__file__), 'data')

# Support running tests from the source tree
if not os.path.exists(DATA):
    from skbuild.constants import CMAKE_INSTALL_DIR as SKBUILD_CMAKE_INSTALL_DIR
    from skbuild.constants import set_skbuild_plat_name

    if platform.system().lower() == "darwin":
        # Since building the project specifying --plat-name or CMAKE_OSX_* variables
        # leads to different SKBUILD_DIR, the code below attempt to guess the most
        # likely plat-name.
        _skbuild_dirs = os.listdir(os.path.join(os.path.dirname(__file__), '..', '..', '_skbuild'))
        if _skbuild_dirs:
            _likely_plat_name = '-'.join(_skbuild_dirs[0].split('-')[:3])
            set_skbuild_plat_name(_likely_plat_name)

    _data = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', SKBUILD_CMAKE_INSTALL_DIR(), 'src/ninja/data'))
    if os.path.exists(_data):
        DATA = _data

BIN_DIR = os.path.join(DATA, 'bin')


def _program(name, args):
    return subprocess.call([os.path.join(BIN_DIR, name)] + args, close_fds=False)


def ninja():
    raise SystemExit(_program('ninja', sys.argv[1:]))
