
set(testable_features
  cxx_alias_templates
  cxx_alignas
  cxx_attributes
  cxx_auto_type
  cxx_binary_literals
  cxx_constexpr
  cxx_contextual_conversions
  cxx_decltype
  cxx_default_function_template_args
  cxx_defaulted_functions
  cxx_delegating_constructors
  cxx_deleted_functions
  cxx_explicit_conversions
  cxx_generalized_initializers
  cxx_inheriting_constructors
  cxx_lambdas
  cxx_local_type_template_args
  cxx_noexcept
  cxx_nonstatic_member_init
  cxx_nullptr
  cxx_range_for
  cxx_raw_string_literals
  cxx_reference_qualified_functions
  cxx_relaxed_constexpr
  cxx_return_type_deduction
  cxx_rvalue_references
  cxx_static_assert
  cxx_strong_enums
  cxx_thread_local
  cxx_unicode_literals
  cxx_unrestricted_unions
  cxx_user_literals
  cxx_variable_templates
  cxx_variadic_templates
)
if(NOT "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
  list(APPEND testable_features cxx_decltype_incomplete_return_types)
endif()

foreach(feature ${testable_features})
  set(_cmake_feature_test_${feature} "${_cmake_oldestSupported} && __has_feature(${feature})")
endforeach()

unset(testable_features)

set(_cmake_feature_test_cxx_aggregate_default_initializers "${_cmake_oldestSupported} && __has_feature(cxx_aggregate_nsdmi)")

set(_cmake_feature_test_cxx_trailing_return_types "${_cmake_oldestSupported} && __has_feature(cxx_trailing_return)")
set(_cmake_feature_test_cxx_alignof "${_cmake_oldestSupported} && __has_feature(cxx_alignas)")
set(_cmake_feature_test_cxx_final "${_cmake_oldestSupported} && __has_feature(cxx_override_control)")
set(_cmake_feature_test_cxx_override "${_cmake_oldestSupported} && __has_feature(cxx_override_control)")
set(_cmake_feature_test_cxx_uniform_initialization "${_cmake_oldestSupported} && __has_feature(cxx_generalized_initializers)")
set(_cmake_feature_test_cxx_defaulted_move_initializers "${_cmake_oldestSupported} && __has_feature(cxx_defaulted_functions)")
set(_cmake_feature_test_cxx_lambda_init_captures "${_cmake_oldestSupported} && __has_feature(cxx_init_captures)")
