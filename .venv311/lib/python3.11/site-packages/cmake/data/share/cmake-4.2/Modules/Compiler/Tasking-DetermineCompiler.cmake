# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.
set(_compiler_id_pp_test "defined(__TASKING__)")

set(_compiler_id_version_compute "
  # define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__VERSION__/1000)
  # define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__VERSION__ % 100)")

string(APPEND _compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_INTERNAL @MACRO_DEC@(__VERSION__)")
