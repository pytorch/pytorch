
set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__clang_major__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__clang_minor__)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__clang_patchlevel__)
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(_MSC_VER / 100)
#  define @PREFIX@SIMULATE_VERSION_MINOR @MACRO_DEC@(_MSC_VER % 100)
# endif")

set(_compiler_id_simulate "
# if defined(_MSC_VER)
#  define @PREFIX@SIMULATE_ID \"MSVC\"
# endif")
