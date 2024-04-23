# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

# Header only singletons are not available in windows yet.

if(NOT WIN32)

  add_library(component_a STATIC component_a.cc)
  target_link_libraries(component_a opentelemetry_api)

  add_library(component_b STATIC component_b.cc)
  target_link_libraries(component_b opentelemetry_api)

  add_library(component_c SHARED component_c.cc)
  set_target_properties(component_c PROPERTIES CXX_VISIBILITY_PRESET default)
  target_link_libraries(component_c opentelemetry_api)

  add_library(component_d SHARED component_d.cc)
  set_target_properties(component_d PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(component_d opentelemetry_api)

  add_library(component_e SHARED component_e.cc)
  set_target_properties(component_e PROPERTIES CXX_VISIBILITY_PRESET default)
  target_link_libraries(component_e opentelemetry_api)

  add_library(component_f SHARED component_f.cc)
  set_target_properties(component_f PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(component_f opentelemetry_api)

  add_library(component_g SHARED component_g.cc)
  set_target_properties(component_g PROPERTIES CXX_VISIBILITY_PRESET default)
  target_link_libraries(component_g opentelemetry_api)

  add_library(component_h SHARED component_h.cc)
  set_target_properties(component_h PROPERTIES CXX_VISIBILITY_PRESET hidden)
  target_link_libraries(component_h opentelemetry_api)

  add_executable(singleton_test singleton_test.cc)

  # Not linking with component_g and component_h on purpose
  target_link_libraries(
    singleton_test
    component_a
    component_b
    component_c
    component_d
    component_e
    component_f
    ${GTEST_BOTH_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_DL_LIBS}
    opentelemetry_api)

  gtest_add_tests(
    TARGET singleton_test
    TEST_PREFIX singleton.
    TEST_LIST singleton_test)

endif()
