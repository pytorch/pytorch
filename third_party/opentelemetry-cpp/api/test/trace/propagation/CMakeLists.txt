# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(detail)

foreach(testname http_text_format_test b3_propagation_test)
  add_executable(${testname} "${testname}.cc")
  target_link_libraries(${testname} ${GTEST_BOTH_LIBRARIES}
                        ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
  gtest_add_tests(
    TARGET ${testname}
    TEST_PREFIX trace.
    TEST_LIST ${testname})
endforeach()

if(NOT WITH_NO_DEPRECATED_CODE)
  foreach(testname jaeger_propagation_test)

    add_executable(${testname} "${testname}.cc")
    target_link_libraries(${testname} ${GTEST_BOTH_LIBRARIES}
                          ${CMAKE_THREAD_LIBS_INIT} opentelemetry_api)
    gtest_add_tests(
      TARGET ${testname}
      TEST_PREFIX trace.
      TEST_LIST ${testname})
  endforeach()
endif()
