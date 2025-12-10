
set(_compiler_id_pp_test "defined(__LCC__) && (defined(__GNUC__) || defined(__GNUG__) || defined(__MCST__))")

set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__LCC__ / 100)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__LCC__ % 100)
# if defined(__LCC_MINOR__)
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__LCC_MINOR__)
# endif
# if defined(__GNUC__) && defined(__GNUC_MINOR__)
#  define @PREFIX@SIMULATE_ID \"GNU\"
#  define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(__GNUC__)
#  define @PREFIX@SIMULATE_VERSION_MINOR @MACRO_DEC@(__GNUC_MINOR__)
#  if defined(__GNUC_PATCHLEVEL__)
#   define @PREFIX@SIMULATE_VERSION_PATCH @MACRO_DEC@(__GNUC_PATCHLEVEL__)
#  endif
# endif")
