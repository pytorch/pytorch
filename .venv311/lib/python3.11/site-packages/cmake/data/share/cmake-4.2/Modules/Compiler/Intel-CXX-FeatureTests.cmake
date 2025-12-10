# References:
#   - https://software.intel.com/en-us/articles/c0x-features-supported-by-intel-c-compiler
#   - https://software.intel.com/en-us/articles/c14-features-supported-by-intel-c-compiler
#   - http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0096r3.html

# Notes:
# [1] Some Intel versions define some feature macros under -std=gnu++98
#     that do not work in that mode (or work with warnings):
#     - __cpp_attributes 200809
#     - __cpp_init_captures 201304
#     - __cpp_lambdas 200907
#     - __cpp_rvalue_references 200610
#     - __cpp_variadic_templates 200704

set(_cmake_feature_test_cxx_variable_templates "__cpp_variable_templates >= 201304")

set(_cmake_oldestSupported "__INTEL_COMPILER >= 1210")
set(DETECT_CXX11 "((__cplusplus >= 201103L) || defined(__INTEL_CXX11_MODE__) || defined(__GXX_EXPERIMENTAL_CXX0X__))")
#ICC version 15 update 1 has a bug where __cplusplus is defined as 1 no matter
#if you are compiling as 98/11/14. So to properly detect C++14 with this version
#we look for the existence of __GXX_EXPERIMENTAL_CXX0X__ but not __INTEL_CXX11_MODE__
set(DETECT_BUGGY_ICC15 "((__INTEL_COMPILER == 1500) && (__INTEL_COMPILER_UPDATE == 1))")
set(DETECT_CXX14 "((__cplusplus >= 201300L) || ((__cplusplus == 201103L) && !defined(__INTEL_CXX11_MODE__)) || ((${DETECT_BUGGY_ICC15}) && defined(__GXX_EXPERIMENTAL_CXX0X__) && !defined(__INTEL_CXX11_MODE__) ) || (defined(__INTEL_CXX11_MODE__) && defined(__cpp_aggregate_nsdmi)) )")
unset(DETECT_BUGGY_ICC15)

set(Intel17_CXX14 "__INTEL_COMPILER >= 1700 && ${DETECT_CXX14}")
set(_cmake_feature_test_cxx_relaxed_constexpr "__cpp_constexpr >= 201304 || (${Intel17_CXX14} && !defined(_MSC_VER))")

set(Intel16_CXX14 "__INTEL_COMPILER >= 1600 && ${DETECT_CXX14}")
set(_cmake_feature_test_cxx_aggregate_default_initializers "${Intel16_CXX14}")
set(_cmake_feature_test_cxx_contextual_conversions "${Intel16_CXX14}")
set(_cmake_feature_test_cxx_generic_lambdas "__cpp_generic_lambdas >= 201304")
set(_cmake_feature_test_cxx_digit_separators "${Intel16_CXX14}")
unset(Intel16_CXX14)

set(Intel15 "__INTEL_COMPILER >= 1500")
set(Intel15_CXX14 "${Intel15} && ${DETECT_CXX14}")
set(_cmake_feature_test_cxx_decltype_auto "__cpp_decltype_auto >= 201304 || ${Intel15_CXX14}")
set(_cmake_feature_test_cxx_lambda_init_captures "(__cpp_init_captures >= 201304 || ${Intel15}) && ${DETECT_CXX14}") # [1]
set(_cmake_feature_test_cxx_attribute_deprecated "${Intel15_CXX14}")
set(_cmake_feature_test_cxx_return_type_deduction "__cpp_return_type_deduction >= 201304 || ${Intel15_CXX14}")
unset(Intel15_CXX14)
unset(Intel15)

set(Intel15_CXX11 "__INTEL_COMPILER >= 1500 && ${DETECT_CXX11}")
set(_cmake_feature_test_cxx_alignas "${Intel15_CXX11}")
set(_cmake_feature_test_cxx_alignof "${Intel15_CXX11}")
set(_cmake_feature_test_cxx_inheriting_constructors "${Intel15_CXX11}")
set(_cmake_feature_test_cxx_user_literals "__cpp_user_defined_literals >= 200809 || (${Intel15_CXX11} && (!defined(_MSC_VER) || __INTEL_COMPILER >= 1600))")
set(_cmake_feature_test_cxx_thread_local "${Intel15_CXX11}")
unset(Intel15_CXX11)

set(Intel14_CXX11 "${DETECT_CXX11} && (__INTEL_COMPILER > 1400 || (__INTEL_COMPILER == 1400 && __INTEL_COMPILER_UPDATE >= 2))")
# Documented as 12.0+ but in testing it only works on 14.0.2+
set(_cmake_feature_test_cxx_decltype_incomplete_return_types "${Intel14_CXX11} && !defined(_MSC_VER)")

set(Intel14_CXX11 "__INTEL_COMPILER >= 1400 && ${DETECT_CXX11}")
set(_cmake_feature_test_cxx_delegating_constructors "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_constexpr "__cpp_constexpr >= 200704 || ${Intel14_CXX11}")
set(_cmake_feature_test_cxx_sizeof_member "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_strong_enums "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_reference_qualified_functions "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_raw_string_literals "__cpp_raw_strings >= 200710 || ${Intel14_CXX11}")
set(_cmake_feature_test_cxx_unicode_literals "__cpp_unicode_literals >= 200710 || (${Intel14_CXX11} && (!defined(_MSC_VER) || __INTEL_COMPILER >= 1600))")
set(_cmake_feature_test_cxx_inline_namespaces "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_unrestricted_unions "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_nonstatic_member_init "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_enum_forward_declarations "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_override "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_final "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_noexcept "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_defaulted_move_initializers "${Intel14_CXX11}")
set(_cmake_feature_test_cxx_generalized_initializers "${Intel14_CXX11}")
unset(Intel14_CXX11)

set(Intel13_CXX11 "__INTEL_COMPILER >= 1300 && ${DETECT_CXX11}")
set(_cmake_feature_test_cxx_explicit_conversions "${Intel13_CXX11}")
set(_cmake_feature_test_cxx_range_for "${Intel13_CXX11}")
# Cannot find Intel documentation for N2640: cxx_uniform_initialization
set(_cmake_feature_test_cxx_uniform_initialization "${Intel13_CXX11}")
unset(Intel13_CXX11)

set(Intel121 "${_cmake_oldestSupported}")
set(Intel121_CXX11 "${Intel121} && ${DETECT_CXX11}")
set(_cmake_feature_test_cxx_variadic_templates "(__cpp_variadic_templates >= 200704 || ${Intel121}) && ${DETECT_CXX11}") # [1]
set(_cmake_feature_test_cxx_alias_templates "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_nullptr "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_trailing_return_types "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_attributes "(__cpp_attributes >= 200809 || ${Intel121}) && ${DETECT_CXX11}") # [1]
set(_cmake_feature_test_cxx_default_function_template_args "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_extended_friend_declarations "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_rvalue_references "(__cpp_rvalue_references >= 200610 || ${Intel121}) && ${DETECT_CXX11}") # [1]
set(_cmake_feature_test_cxx_decltype "__cpp_decltype >= 200707 || ${Intel121_CXX11}")
set(_cmake_feature_test_cxx_defaulted_functions "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_deleted_functions "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_local_type_template_args "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_lambdas "(__cpp_lambdas >= 200907 || ${Intel121}) && ${DETECT_CXX11}") # [1]
set(_cmake_feature_test_cxx_binary_literals "__cpp_binary_literals >= 201304 || ${Intel121}")
set(_cmake_feature_test_cxx_static_assert "(__cpp_static_assert >= 200410 || ${Intel121}) && ${DETECT_CXX11}")
set(_cmake_feature_test_cxx_right_angle_brackets "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_auto_type "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_extern_templates "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_variadic_macros "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_long_long_type "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_func_identifier "${Intel121_CXX11}")
set(_cmake_feature_test_cxx_template_template_parameters "${Intel121_CXX11}")
unset(Intel121_CXX11)
unset(Intel121)

unset(DETECT_CXX11)
unset(DETECT_CXX14)
