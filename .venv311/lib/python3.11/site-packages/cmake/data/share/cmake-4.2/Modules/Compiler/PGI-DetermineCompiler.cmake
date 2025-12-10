
set(_compiler_id_pp_test "defined(__PGI)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__PGIC__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__PGIC_MINOR__)
# if defined(__PGIC_PATCHLEVEL__)
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__PGIC_PATCHLEVEL__)
# endif")
