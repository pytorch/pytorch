
set(_compiler_id_pp_test "defined(__ORANGEC__)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__ORANGEC_MAJOR__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__ORANGEC_MINOR__)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__ORANGEC_PATCHLEVEL__)")
