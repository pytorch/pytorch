if(USE_DISTRIBUTED AND NOT WIN32)
  set(DIST_AUTOGRAD_TEST_DIR "${TORCH_ROOT}/test/cpp/dist_autograd")
  set(DIST_AUTOGRAD_TEST_SOURCES
      ${TORCH_ROOT}/test/cpp/common/main.cpp
      ${DIST_AUTOGRAD_TEST_DIR}/test_dist_autograd.cpp)

  add_executable(test_dist_autograd ${DIST_AUTOGRAD_TEST_SOURCES})
  target_include_directories(test_dist_autograd PRIVATE ${ATen_CPU_INCLUDE})
  target_link_libraries(test_dist_autograd PRIVATE torch gtest_main)

  if(USE_CUDA)
    target_compile_definitions(test_dist_autograd PRIVATE "USE_CUDA")
  endif()

  if(INSTALL_TEST)
    set_target_properties(
      test_dist_autograd
      PROPERTIES INSTALL_RPATH
                 "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
    install(TARGETS test_dist_autograd DESTINATION bin)
    # Install PDB files for MSVC builds
    if(MSVC AND BUILD_SHARED_LIBS)
      install(
        FILES $<TARGET_PDB_FILE:test_dist_autograd>
        DESTINATION bin
        OPTIONAL)
    endif()
  endif()
endif()
