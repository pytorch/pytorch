
set(_compiler_id_pp_test "defined(__HP_cc)")

set(_compiler_id_version_compute "
  /* __HP_cc = VVRRPP */
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__HP_cc/10000)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__HP_cc/100 % 100)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__HP_cc     % 100)")
