
set(_compiler_id_pp_test "defined(__NVCOMPILER)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__NVCOMPILER_MAJOR__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__NVCOMPILER_MINOR__)
# if defined(__NVCOMPILER_PATCHLEVEL__)
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__NVCOMPILER_PATCHLEVEL__)
# endif")
