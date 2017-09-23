## @package get_python_cmake_flags
# Module scripts.get_python_cmake_flags
##############################################################################
# Use this script to find your preferred python installation.
##############################################################################
#
# You can use the following to build with your preferred version of python
# if your installation is not being properly detected by CMake.
#
#   mkdir -p build && cd build
#   cmake $(python ../scripts/get_python_libs.py) ..
#   make
#

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from distutils import sysconfig
import os
import sys
import platform

# Flags to print to stdout
flags = ''
inc = sysconfig.get_python_inc()
lib = sysconfig.get_config_var("LIBDIR")

# macOS specific
if sys.platform == "darwin":
    lib = os.path.dirname(lib) + '/Python'
    if os.path.isfile(lib):
        flags += '-DPYTHON_LIBRARY={lib} '.format(lib=lib)

if os.path.isfile(inc + '/Python.h'):
    flags += '-DPYTHON_INCLUDE_DIR={inc} '.format(inc=inc)

print(flags, end='')
