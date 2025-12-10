
set(_compiler_id_pp_test "defined(__INTEL_COMPILER) || defined(__ICC)")

set(_compiler_id_version_compute "
  /* __INTEL_COMPILER = VRP prior to 2021, and then VVVV for 2021 and later,
     except that a few beta releases use the old format with V=2021.  */
# if __INTEL_COMPILER < 2021 || __INTEL_COMPILER == 202110 || __INTEL_COMPILER == 202111
#  define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__INTEL_COMPILER/100)
#  define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__INTEL_COMPILER/10 % 10)
#  if defined(__INTEL_COMPILER_UPDATE)
#   define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__INTEL_COMPILER_UPDATE)
#  else
#   define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__INTEL_COMPILER   % 10)
#  endif
# else
#  define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__INTEL_COMPILER)
#  define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__INTEL_COMPILER_UPDATE)
   /* The third version component from --version is an update index,
      but no macro is provided for it.  */
#  define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(0)
# endif
# if defined(__INTEL_COMPILER_BUILD_DATE)
   /* __INTEL_COMPILER_BUILD_DATE = YYYYMMDD */
#  define @PREFIX@COMPILER_VERSION_TWEAK @MACRO_DEC@(__INTEL_COMPILER_BUILD_DATE)
# endif
# if defined(_MSC_VER)
   /* _MSC_VER = VVRR */
#  define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(_MSC_VER / 100)
#  define @PREFIX@SIMULATE_VERSION_MINOR @MACRO_DEC@(_MSC_VER % 100)
# endif
# if defined(__GNUC__)
#  define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(__GNUC__)
# elif defined(__GNUG__)
#  define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(__GNUG__)
# endif
# if defined(__GNUC_MINOR__)
#  define @PREFIX@SIMULATE_VERSION_MINOR @MACRO_DEC@(__GNUC_MINOR__)
# endif
# if defined(__GNUC_PATCHLEVEL__)
#  define @PREFIX@SIMULATE_VERSION_PATCH @MACRO_DEC@(__GNUC_PATCHLEVEL__)
# endif")

set(_compiler_id_simulate "
# if defined(_MSC_VER)
#  define @PREFIX@SIMULATE_ID \"MSVC\"
# endif
# if defined(__GNUC__)
#  define @PREFIX@SIMULATE_ID \"GNU\"
# endif")
