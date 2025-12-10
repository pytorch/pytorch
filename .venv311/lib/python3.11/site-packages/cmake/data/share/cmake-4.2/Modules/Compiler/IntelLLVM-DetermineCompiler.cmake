
set(_compiler_id_pp_test "(defined(__clang__) && defined(__INTEL_CLANG_COMPILER)) || defined(__INTEL_LLVM_COMPILER)")

set(_compiler_id_version_compute "
/* __INTEL_LLVM_COMPILER = VVVVRP prior to 2021.2.0, VVVVRRPP for 2021.2.0 and
 * later.  Look for 6 digit vs. 8 digit version number to decide encoding.
 * VVVV is no smaller than the current year when a version is released.
 */
#if __INTEL_LLVM_COMPILER < 1000000L
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__INTEL_LLVM_COMPILER/100)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__INTEL_LLVM_COMPILER/10 % 10)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__INTEL_LLVM_COMPILER    % 10)
#else
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__INTEL_LLVM_COMPILER/10000)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__INTEL_LLVM_COMPILER/100 % 100)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__INTEL_LLVM_COMPILER     % 100)
#endif
#if defined(_MSC_VER)
  /* _MSC_VER = VVRR */
# define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(_MSC_VER / 100)
# define @PREFIX@SIMULATE_VERSION_MINOR @MACRO_DEC@(_MSC_VER % 100)
#endif
#if defined(__GNUC__)
# define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(__GNUC__)
#elif defined(__GNUG__)
# define @PREFIX@SIMULATE_VERSION_MAJOR @MACRO_DEC@(__GNUG__)
#endif
#if defined(__GNUC_MINOR__)
# define @PREFIX@SIMULATE_VERSION_MINOR @MACRO_DEC@(__GNUC_MINOR__)
#endif
#if defined(__GNUC_PATCHLEVEL__)
# define @PREFIX@SIMULATE_VERSION_PATCH @MACRO_DEC@(__GNUC_PATCHLEVEL__)
#endif")

set(_compiler_id_simulate "
#if defined(_MSC_VER)
# define @PREFIX@SIMULATE_ID \"MSVC\"
#endif
#if defined(__GNUC__)
# define @PREFIX@SIMULATE_ID \"GNU\"
#endif")
