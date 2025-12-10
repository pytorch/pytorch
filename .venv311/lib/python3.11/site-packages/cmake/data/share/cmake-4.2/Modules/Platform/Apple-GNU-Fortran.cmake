# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Platform/Apple-GNU)
__apple_compiler_gnu(Fortran)
cmake_gnu_set_sysroot_flag(Fortran)
cmake_gnu_set_osx_deployment_target_flag(Fortran)

set(CMAKE_Fortran_OSX_COMPATIBILITY_VERSION_FLAG "-compatibility_version ")
set(CMAKE_Fortran_OSX_CURRENT_VERSION_FLAG "-current_version ")
