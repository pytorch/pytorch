
# Reference: http://clang.llvm.org/cxx_status.html
# http://clang.llvm.org/docs/LanguageExtensions.html

set(_cmake_oldestSupported "((__clang_major__ * 100) + __clang_minor__) >= 301")

include("${CMAKE_CURRENT_LIST_DIR}/Clang-CXX-TestableFeatures.cmake")

set(Clang34_CXX14 "((__clang_major__ * 100) + __clang_minor__) >= 304 && __cplusplus > 201103L")
# http://llvm.org/bugs/show_bug.cgi?id=19242
set(_cmake_feature_test_cxx_attribute_deprecated "${Clang34_CXX14}")
# http://llvm.org/bugs/show_bug.cgi?id=19698
set(_cmake_feature_test_cxx_decltype_auto "${Clang34_CXX14}")
set(_cmake_feature_test_cxx_digit_separators "${Clang34_CXX14}")
# http://llvm.org/bugs/show_bug.cgi?id=19674
set(_cmake_feature_test_cxx_generic_lambdas "${Clang34_CXX14}")

set(Clang31_CXX11 "${_cmake_oldestSupported} && __cplusplus >= 201103L")
set(_cmake_feature_test_cxx_enum_forward_declarations "${Clang31_CXX11}")
set(_cmake_feature_test_cxx_sizeof_member "${Clang31_CXX11}")
# TODO: Should be supported by Clang 2.9
set(Clang29_CXX11 "${_cmake_oldestSupported} && __cplusplus >= 201103L")
set(_cmake_feature_test_cxx_extended_friend_declarations "${Clang29_CXX11}")
set(_cmake_feature_test_cxx_extern_templates "${Clang29_CXX11}")
set(_cmake_feature_test_cxx_func_identifier "${Clang29_CXX11}")
set(_cmake_feature_test_cxx_inline_namespaces "${Clang29_CXX11}")
set(_cmake_feature_test_cxx_long_long_type "${Clang29_CXX11}")
set(_cmake_feature_test_cxx_right_angle_brackets "${Clang29_CXX11}")
set(_cmake_feature_test_cxx_variadic_macros "${Clang29_CXX11}")

# TODO: Should be supported forever?
set(Clang_CXX98 "${_cmake_oldestSupported} && __cplusplus >= 199711L")
set(_cmake_feature_test_cxx_template_template_parameters "${Clang_CXX98}")
