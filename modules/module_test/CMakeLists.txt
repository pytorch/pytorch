if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  # If we are building the standalone module, we set the proper cmake variables.
  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  find_package(Caffe2 REQUIRED)
  set(BUILD_TEST ON)
  option(BUILD_SHARED_LIBS "Build shared libs." ON)
endif()

if(BUILD_TEST)
  add_library(
      caffe2_module_test_dynamic
      ${CMAKE_CURRENT_SOURCE_DIR}/module_test_dynamic.cc)

  if(HAVE_SOVERSION)
    set_target_properties(caffe2_module_test_dynamic PROPERTIES
        VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
  endif()
  target_link_libraries(caffe2_module_test_dynamic torch_library)
  install(TARGETS caffe2_module_test_dynamic DESTINATION lib)
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:caffe2_module_test_dynamic> DESTINATION lib OPTIONAL)
  endif()
endif()
