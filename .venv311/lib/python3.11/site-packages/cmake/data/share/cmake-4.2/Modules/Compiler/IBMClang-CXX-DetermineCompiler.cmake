set(_compiler_id_pp_test "defined(__open_xl__) && defined(__clang__)")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__open_xl_version__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__open_xl_release__)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__open_xl_modification__)
# define @PREFIX@COMPILER_VERSION_TWEAK @MACRO_DEC@(__open_xl_ptf_fix_level__)
# define @PREFIX@COMPILER_VERSION_INTERNAL_STR  __clang_version__
")
