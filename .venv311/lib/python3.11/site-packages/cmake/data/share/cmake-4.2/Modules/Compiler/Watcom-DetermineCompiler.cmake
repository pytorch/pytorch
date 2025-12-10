
set(_compiler_id_pp_test "defined(__WATCOMC__) && __WATCOMC__ < 1200")

set(_compiler_id_version_compute "
   /* __WATCOMC__ = VVRR */
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__WATCOMC__ / 100)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@((__WATCOMC__ / 10) % 10)
# if (__WATCOMC__ % 10) > 0
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__WATCOMC__ % 10)
# endif")
