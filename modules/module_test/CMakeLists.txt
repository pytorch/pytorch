if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  # If we are building the standalone module, we set the proper cmake variables.
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  find_package(Caffe2 REQUIRED)
  set(BUILD_TEST ON)
endif()

if (BUILD_TEST)
  add_library(
      caffe2_module_test_dynamic
      ${CMAKE_CURRENT_SOURCE_DIR}/module_test_dynamic.cc)

  target_link_libraries(caffe2_module_test_dynamic caffe2_library)
  install(TARGETS caffe2_module_test_dynamic DESTINATION lib)
endif()
