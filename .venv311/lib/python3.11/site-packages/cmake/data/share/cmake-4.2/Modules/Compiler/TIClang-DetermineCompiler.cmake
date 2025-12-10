# TI Clang-based Toolchains
set(_compiler_id_pp_test "defined(__clang__) && defined(__ti__)")

set(_compiler_id_version_compute "
  # define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__ti_major__)
  # define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__ti_minor__)
  # define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__ti_patchlevel__)")

string(APPEND _compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_INTERNAL @MACRO_DEC@(__ti_version__)")
