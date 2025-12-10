set(_compiler_id_pp_test "defined(__ghs__)")

set(_compiler_id_version_compute "
/* __GHS_VERSION_NUMBER = VVVVRP */
# ifdef __GHS_VERSION_NUMBER
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__GHS_VERSION_NUMBER / 100)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__GHS_VERSION_NUMBER / 10 % 10)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__GHS_VERSION_NUMBER      % 10)
# endif")
