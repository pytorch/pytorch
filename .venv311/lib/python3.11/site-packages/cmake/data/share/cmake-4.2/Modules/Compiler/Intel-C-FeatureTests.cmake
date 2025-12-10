# References:
#   - https://software.intel.com/en-us/articles/iso-iec-standards-language-conformance-for-intel-c-compiler
#   - https://software.intel.com/en-us/articles/c99-support-in-intel-c-compiler
#   - https://software.intel.com/en-us/articles/c11-support-in-intel-c-compiler

set(DETECT_C99 "defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L")
set(DETECT_C11 "defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L")

#static assert is only around in version 1500 update 2 and above
set(_cmake_feature_test_c_static_assert "(__INTEL_COMPILER > 1500 || (__INTEL_COMPILER == 1500 && __INTEL_COMPILER_UPDATE > 1) ) && (${DETECT_C11} || ${DETECT_C99} && !defined(_MSC_VER))")

set(_cmake_oldestSupported "__INTEL_COMPILER >= 1110")
set(Intel_C99 "${_cmake_oldestSupported} && ${DETECT_C99}")
set(_cmake_feature_test_c_restrict "${Intel_C99}")
set(_cmake_feature_test_c_variadic_macros "${Intel_C99}")
set(_cmake_feature_test_c_function_prototypes "${_cmake_oldestSupported}")
unset(Intel_C99)

unset(DETECT_C99)
unset(DETECT_C11)
