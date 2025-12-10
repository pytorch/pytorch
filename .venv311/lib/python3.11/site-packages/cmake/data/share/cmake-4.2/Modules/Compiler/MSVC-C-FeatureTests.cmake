set(_cmake_oldestSupported "_MSC_VER >= 1600")

set(_cmake_feature_test_c_restrict "_MSC_VER >= 1927")
set(_cmake_feature_test_c_static_assert "_MSC_VER >= 1928")

set(_cmake_feature_test_c_variadic_macros "${_cmake_oldestSupported}")
set(_cmake_feature_test_c_function_prototypes "${_cmake_oldestSupported}")
