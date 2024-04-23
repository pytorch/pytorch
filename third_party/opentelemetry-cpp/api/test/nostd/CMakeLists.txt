# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

foreach(
  testname
  function_ref_test
  string_view_test
  unique_ptr_test
  utility_test
  span_test
  shared_ptr_test
  variant_test)
  add_executable(${testname} "${testname}.cc")
  target_link_libraries(${testname} ${GTEST_BOTH_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
  gtest_add_tests(
    TARGET ${testname}
    TEST_PREFIX nostd.
    TEST_LIST ${testname})
endforeach()
