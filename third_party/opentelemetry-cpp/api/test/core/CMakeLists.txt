# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_executable(timestamp_test timestamp_test.cc)
target_link_libraries(timestamp_test ${GTEST_BOTH_LIBRARIES}
                      ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
gtest_add_tests(
  TARGET timestamp_test
  TEST_PREFIX trace.
  TEST_LIST timestamp_test)

add_executable(version_test version_test.cc)
target_link_libraries(version_test ${GTEST_BOTH_LIBRARIES}
                      ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
gtest_add_tests(
  TARGET version_test
  TEST_PREFIX version.
  TEST_LIST version_test)
