
set(_compiler_id_pp_test "defined(__FUJITSU)")

set(_compiler_id_version_compute "
# if defined(__FCC_version__)
#   define @PREFIX@COMPILER_VERSION __FCC_version__
# elif defined(__FCC_major__)
#   define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__FCC_major__)
#   define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__FCC_minor__)
#   define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__FCC_patchlevel__)
# endif
# if defined(__fcc_version)
#   define @PREFIX@COMPILER_VERSION_INTERNAL @MACRO_DEC@(__fcc_version)
# elif defined(__FCC_VERSION)
#   define @PREFIX@COMPILER_VERSION_INTERNAL @MACRO_DEC@(__FCC_VERSION)
# endif
")
