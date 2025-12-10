
set(_cmake_oldestSupported "((__clang_major__ * 100) + __clang_minor__) >= 304")

set(Clang_C11 "${_cmake_oldestSupported} && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L")
set(_cmake_feature_test_c_static_assert "${Clang_C11}")
set(Clang_C99 "${_cmake_oldestSupported} && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L")
set(_cmake_feature_test_c_restrict "${Clang_C99}")
set(_cmake_feature_test_c_variadic_macros "${Clang_C99}")

set(Clang_C90 "${_cmake_oldestSupported}")
set(_cmake_feature_test_c_function_prototypes "${Clang_C90}")
