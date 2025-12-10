
# sdcc, the small devices C compiler for embedded systems,
#   http://sdcc.sourceforge.net  */
set(_compiler_id_pp_test "defined(__SDCC_VERSION_MAJOR) || defined(SDCC)")

set(_compiler_id_version_compute "
# if defined(__SDCC_VERSION_MAJOR)
#  define COMPILER_VERSION_MAJOR @MACRO_DEC@(__SDCC_VERSION_MAJOR)
#  define COMPILER_VERSION_MINOR @MACRO_DEC@(__SDCC_VERSION_MINOR)
#  define COMPILER_VERSION_PATCH @MACRO_DEC@(__SDCC_VERSION_PATCH)
# else
  /* SDCC = VRP */
#  define COMPILER_VERSION_MAJOR @MACRO_DEC@(SDCC/100)
#  define COMPILER_VERSION_MINOR @MACRO_DEC@(SDCC/10 % 10)
#  define COMPILER_VERSION_PATCH @MACRO_DEC@(SDCC    % 10)
# endif")
