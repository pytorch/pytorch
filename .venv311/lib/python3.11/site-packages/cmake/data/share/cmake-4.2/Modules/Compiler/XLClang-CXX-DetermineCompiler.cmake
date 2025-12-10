set(_compiler_id_pp_test "defined(__ibmxl__) && defined(__clang__)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__ibmxl_version__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__ibmxl_release__)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__ibmxl_modification__)
# define @PREFIX@COMPILER_VERSION_TWEAK @MACRO_DEC@(__ibmxl_ptf_fix_level__)
")
