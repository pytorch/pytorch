
# Reference: http://gcc.gnu.org/projects/cxx0x.html
# http://gcc.gnu.org/projects/cxx1y.html

set(_cmake_oldestSupported "(__GNUC__ * 100 + __GNUC_MINOR__) >= 404")

set(GNU50_CXX14 "(__GNUC__ * 100 + __GNUC_MINOR__) >= 500 && __cplusplus >= 201402L")
set(_cmake_feature_test_cxx_variable_templates "${GNU50_CXX14}")
set(_cmake_feature_test_cxx_relaxed_constexpr "${GNU50_CXX14}")
set(_cmake_feature_test_cxx_aggregate_default_initializers "${GNU50_CXX14}")

# GNU 4.9 in c++14 mode sets __cplusplus to 201300L, so don't test for the
# correct value of it below.
# https://patchwork.ozlabs.org/patch/382470/
set(GNU49_CXX14 "(__GNUC__ * 100 + __GNUC_MINOR__) >= 409 && __cplusplus > 201103L")
set(_cmake_feature_test_cxx_contextual_conversions "${GNU49_CXX14}")
set(_cmake_feature_test_cxx_attribute_deprecated "${GNU49_CXX14}")
set(_cmake_feature_test_cxx_decltype_auto "${GNU49_CXX14}")
set(_cmake_feature_test_cxx_digit_separators "${GNU49_CXX14}")
set(_cmake_feature_test_cxx_generic_lambdas "${GNU49_CXX14}")
# GNU 4.3 supports binary literals as an extension, but may warn about
# use of extensions prior to GNU 4.9
# http://stackoverflow.com/questions/16334024/difference-between-gcc-binary-literals-and-c14-ones
set(_cmake_feature_test_cxx_binary_literals "${GNU49_CXX14}")
# The features below are documented as available in GNU 4.8 (by implementing an
# earlier draft of the standard paper), but that version of the compiler
# does not set __cplusplus to a value greater than 201103L until GNU 4.9:
# http://gcc.gnu.org/onlinedocs/gcc-4.8.2/cpp/Standard-Predefined-Macros.html#Standard-Predefined-Macros
# http://gcc.gnu.org/onlinedocs/gcc-4.9.0/cpp/Standard-Predefined-Macros.html#Standard-Predefined-Macros
# So, CMake only reports availability for it with GNU 4.9 or later.
set(_cmake_feature_test_cxx_return_type_deduction "${GNU49_CXX14}")
set(_cmake_feature_test_cxx_lambda_init_captures "${GNU49_CXX14}")

# Introduced in GCC 4.8.1
set(GNU481_CXX11 "((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) >= 40801) && __cplusplus >= 201103L")
set(_cmake_feature_test_cxx_decltype_incomplete_return_types "${GNU481_CXX11}")
set(_cmake_feature_test_cxx_reference_qualified_functions "${GNU481_CXX11}")
set(GNU48_CXX11 "(__GNUC__ * 100 + __GNUC_MINOR__) >= 408 && __cplusplus >= 201103L")
set(_cmake_feature_test_cxx_alignas "${GNU48_CXX11}")
# The alignof feature works with GNU 4.7 and -std=c++11, but it is documented
# as available with GNU 4.8, so treat that as true.
set(_cmake_feature_test_cxx_alignof "${GNU48_CXX11}")
set(_cmake_feature_test_cxx_attributes "${GNU48_CXX11}")
set(_cmake_feature_test_cxx_inheriting_constructors "${GNU48_CXX11}")
set(_cmake_feature_test_cxx_thread_local "${GNU48_CXX11}")
set(GNU47_CXX11 "(__GNUC__ * 100 + __GNUC_MINOR__) >= 407 && __cplusplus >= 201103L")
set(_cmake_feature_test_cxx_alias_templates "${GNU47_CXX11}")
set(_cmake_feature_test_cxx_delegating_constructors "${GNU47_CXX11}")
set(_cmake_feature_test_cxx_extended_friend_declarations "${GNU47_CXX11}")
set(_cmake_feature_test_cxx_final "${GNU47_CXX11}")
set(_cmake_feature_test_cxx_nonstatic_member_init "${GNU47_CXX11}")
set(_cmake_feature_test_cxx_override "${GNU47_CXX11}")
set(_cmake_feature_test_cxx_user_literals "${GNU47_CXX11}")
# NOTE: C++11 was ratified in September 2011. GNU 4.7 is the first minor
# release following that (March 2012), and the first minor release to
# support -std=c++11. Prior to that, support for C++11 features is technically
# experimental and possibly incomplete (see for example the note below about
# cxx_variadic_template_template_parameters)
# GNU does not define __cplusplus correctly before version 4.7.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=1773
# __GXX_EXPERIMENTAL_CXX0X__ is defined in prior versions, but may not be
# defined in the future.
set(GNU_CXX0X_DEFINED "(__cplusplus >= 201103L || (defined(__GXX_EXPERIMENTAL_CXX0X__) && __GXX_EXPERIMENTAL_CXX0X__))")
set(GNU46_CXX11 "(__GNUC__ * 100 + __GNUC_MINOR__) >= 406 && ${GNU_CXX0X_DEFINED}")
set(_cmake_feature_test_cxx_constexpr "${GNU46_CXX11}")
set(_cmake_feature_test_cxx_defaulted_move_initializers "${GNU46_CXX11}")
set(_cmake_feature_test_cxx_enum_forward_declarations "${GNU46_CXX11}")
set(_cmake_feature_test_cxx_noexcept "${GNU46_CXX11}")
set(_cmake_feature_test_cxx_nullptr "${GNU46_CXX11}")
set(_cmake_feature_test_cxx_range_for "${GNU46_CXX11}")
set(_cmake_feature_test_cxx_unrestricted_unions "${GNU46_CXX11}")
set(GNU45_CXX11 "(__GNUC__ * 100 + __GNUC_MINOR__) >= 405 && ${GNU_CXX0X_DEFINED}")
set(_cmake_feature_test_cxx_explicit_conversions "${GNU45_CXX11}")
set(_cmake_feature_test_cxx_lambdas "${GNU45_CXX11}")
set(_cmake_feature_test_cxx_local_type_template_args "${GNU45_CXX11}")
set(_cmake_feature_test_cxx_raw_string_literals "${GNU45_CXX11}")
set(GNU44_CXX11 "(__GNUC__ * 100 + __GNUC_MINOR__) >= 404 && ${GNU_CXX0X_DEFINED}")
set(_cmake_feature_test_cxx_auto_type "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_defaulted_functions "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_deleted_functions "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_generalized_initializers "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_inline_namespaces "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_sizeof_member "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_strong_enums "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_trailing_return_types "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_unicode_literals "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_uniform_initialization "${GNU44_CXX11}")
set(_cmake_feature_test_cxx_variadic_templates "${GNU44_CXX11}")
# TODO: If features are ever recorded for GNU 4.3, there should possibly
# be a new feature added like cxx_variadic_template_template_parameters,
# which is implemented by GNU 4.4, but not 4.3. cxx_variadic_templates is
# actually implemented by GNU 4.3, but variadic template template parameters
# 'completes' it, so that is the version we record as having the variadic
# templates capability in CMake. See
# http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf
# TODO: Should be supported by GNU 4.3
set(GNU43_CXX11 "${_cmake_oldestSupported} && ${GNU_CXX0X_DEFINED}")
set(_cmake_feature_test_cxx_decltype "${GNU43_CXX11}")
set(_cmake_feature_test_cxx_default_function_template_args "${GNU43_CXX11}")
set(_cmake_feature_test_cxx_long_long_type "${GNU43_CXX11}")
set(_cmake_feature_test_cxx_right_angle_brackets "${GNU43_CXX11}")
set(_cmake_feature_test_cxx_rvalue_references "${GNU43_CXX11}")
set(_cmake_feature_test_cxx_static_assert "${GNU43_CXX11}")
# TODO: Should be supported since GNU 3.4?
set(_cmake_feature_test_cxx_extern_templates "${_cmake_oldestSupported} && ${GNU_CXX0X_DEFINED}")
# TODO: Should be supported forever?
set(_cmake_feature_test_cxx_func_identifier "${_cmake_oldestSupported} && ${GNU_CXX0X_DEFINED}")
set(_cmake_feature_test_cxx_variadic_macros "${_cmake_oldestSupported} && ${GNU_CXX0X_DEFINED}")
set(_cmake_feature_test_cxx_template_template_parameters "${_cmake_oldestSupported} && __cplusplus")
