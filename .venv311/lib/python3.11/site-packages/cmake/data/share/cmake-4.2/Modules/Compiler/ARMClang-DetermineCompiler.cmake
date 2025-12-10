# ARMClang Toolchain
set(_compiler_id_pp_test "defined(__clang__) && defined(__ARMCOMPILER_VERSION)")

set(_compiler_id_version_compute "
  # define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__ARMCOMPILER_VERSION/1000000)
  # define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__ARMCOMPILER_VERSION/10000 % 100)
  # define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__ARMCOMPILER_VERSION/100   % 100)")

string(APPEND _compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_INTERNAL @MACRO_DEC@(__ARMCOMPILER_VERSION)")
