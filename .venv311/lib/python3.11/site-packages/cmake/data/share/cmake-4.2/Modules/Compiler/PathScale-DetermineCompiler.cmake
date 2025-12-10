
set(_compiler_id_pp_test "defined(__PATHCC__)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__PATHCC__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__PATHCC_MINOR__)
# if defined(__PATHCC_PATCHLEVEL__)
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__PATHCC_PATCHLEVEL__)
# endif")
