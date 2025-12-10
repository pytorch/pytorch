
set(_cmake_oldestSupported "((__RENESAS_VERSION__ >> 24 & 0xFF)) >= 1")


set(Renesas_C11 "${_cmake_oldestSupported} && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L")
set(Renesas_C99 "${_cmake_oldestSupported} && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L")

set(_cmake_feature_test_c_std_99 "${Renesas_C99}")
set(_cmake_feature_test_c_restrict "${Renesas_C99}")
set(_cmake_feature_test_c_variadic_macros "${Renesas_C99}")

set(Renesas_C90 "${_cmake_oldestSupported}")
set(_cmake_feature_test_c_std_90 "${Renesas_C90}")
set(_cmake_feature_test_c_function_prototypes "${Renesas_C90}")
