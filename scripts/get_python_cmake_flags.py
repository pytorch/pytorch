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
#   cmake $(python ../scripts/get_python_cmake_flags.py) ..
#   make
#


import sys
import sysconfig

flags = [
    f"-DPYTHON_EXECUTABLE:FILEPATH={sys.executable}",
    f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
]

print(" ".join(flags), end="")
