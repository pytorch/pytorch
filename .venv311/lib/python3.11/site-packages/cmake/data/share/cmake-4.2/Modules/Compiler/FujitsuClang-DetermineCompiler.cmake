
set(_compiler_id_pp_test "defined(__CLANG_FUJITSU)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__FCC_major__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__FCC_minor__)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__FCC_patchlevel__)
# define @PREFIX@COMPILER_VERSION_INTERNAL_STR __clang_version__
")
