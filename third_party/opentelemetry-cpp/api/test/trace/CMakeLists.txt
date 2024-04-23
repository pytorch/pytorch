# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(propagation)

foreach(
  testname
  key_value_iterable_view_test
  provider_test
  span_id_test
  trace_id_test
  trace_flags_test
  span_context_test
  scope_test
  noop_test
  trace_state_test
  tracer_test)
  add_executable(api_${testname} "${testname}.cc")
  target_link_libraries(api_${testname} ${GTEST_BOTH_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
  gtest_add_tests(
    TARGET api_${testname}
    TEST_PREFIX trace.
    TEST_LIST api_${testname})
endforeach()

if(WITH_BENCHMARK)
  add_executable(span_id_benchmark span_id_benchmark.cc)
  target_link_libraries(span_id_benchmark benchmark::benchmark
                        ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
  add_executable(span_benchmark span_benchmark.cc)
  target_link_libraries(span_benchmark benchmark::benchmark
                        ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
endif()
