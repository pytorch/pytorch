
set(_compiler_id_pp_test "defined(_MSC_VER)")

set(_compiler_id_version_compute "
  /* _MSC_VER = VVRR */
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(_MSC_VER / 100)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(_MSC_VER % 100)
# if defined(_MSC_FULL_VER)
#  if _MSC_VER >= 1400
    /* _MSC_FULL_VER = VVRRPPPPP */
#   define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(_MSC_FULL_VER % 100000)
#  else
    /* _MSC_FULL_VER = VVRRPPPP */
#   define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(_MSC_FULL_VER % 10000)
#  endif
# endif
# if defined(_MSC_BUILD)
#  define @PREFIX@COMPILER_VERSION_TWEAK @MACRO_DEC@(_MSC_BUILD)
# endif")
