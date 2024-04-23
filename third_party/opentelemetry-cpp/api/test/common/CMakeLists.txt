# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

foreach(testname kv_properties_test string_util_test)
  add_executable(${testname} "${testname}.cc")
  target_link_libraries(${testname} ${GTEST_BOTH_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
  gtest_add_tests(
    TARGET ${testname}
    TEST_PREFIX common.
    TEST_LIST ${testname})
endforeach()

if(WITH_BENCHMARK)
  add_executable(spinlock_benchmark spinlock_benchmark.cc)
  target_link_libraries(spinlock_benchmark benchmark::benchmark
                        ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
endif()
