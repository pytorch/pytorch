# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# determine the compiler to use for ASM using AT&T syntax, e.g. GNU as

set(ASM_DIALECT "-ATT")
set(CMAKE_ASM${ASM_DIALECT}_COMPILER_LIST ${_CMAKE_TOOLCHAIN_PREFIX}gas ${_CMAKE_TOOLCHAIN_PREFIX}as)
include(CMakeDetermineASMCompiler)
set(ASM_DIALECT)
