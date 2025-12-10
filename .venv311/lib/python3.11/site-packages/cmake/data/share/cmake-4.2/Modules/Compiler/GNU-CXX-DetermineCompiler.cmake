
set(_compiler_id_pp_test "defined(__GNUC__) || defined(__GNUG__)")

set(_compiler_id_version_compute "
# if defined(__GNUC__)
#  define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__GNUC__)
# else
#  define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__GNUC_PATCHLEVEL__)
# endif")
