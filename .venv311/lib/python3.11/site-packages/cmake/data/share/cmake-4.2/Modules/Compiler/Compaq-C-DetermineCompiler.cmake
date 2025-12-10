
set(_compiler_id_pp_test "defined(__DECC)")

set(_compiler_id_version_compute "
  /* __DECC_VER = VVRRTPPPP */
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__DECC_VER/10000000)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__DECC_VER/100000  % 100)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__DECC_VER         % 10000)")
