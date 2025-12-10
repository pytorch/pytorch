
set(_compiler_id_pp_test "defined(_CRAYC)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(_RELEASE_MAJOR)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(_RELEASE_MINOR)")
