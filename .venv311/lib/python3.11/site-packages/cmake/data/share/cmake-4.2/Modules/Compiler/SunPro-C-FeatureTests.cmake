set(_cmake_oldestSupported "__SUNPRO_C >= 0x5130")

set(SunPro_C11 "${_cmake_oldestSupported} && defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L || __STDC_VERSION__ >= 199901L && defined(__C11FEATURES__))")
set(_cmake_feature_test_c_static_assert "${SunPro_C11}")
unset(SunPro_C11)

set(SunPro_C99 "${_cmake_oldestSupported} && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L")
set(_cmake_feature_test_c_restrict "${SunPro_C99}")
set(_cmake_feature_test_c_variadic_macros "${SunPro_C99}")
unset(SunPro_C99)

set(SunPro_C90 "${_cmake_oldestSupported}")
set(_cmake_feature_test_c_function_prototypes "${SunPro_C90}")
unset(SunPro_C90)
