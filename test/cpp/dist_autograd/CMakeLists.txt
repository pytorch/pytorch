if(USE_DISTRIBUTED AND NOT WIN32)
  set(DIST_AUTOGRAD_TEST_DIR "${TORCH_ROOT}/test/cpp/dist_autograd")
  set(DIST_AUTOGRAD_TEST_SOURCES
    ${TORCH_ROOT}/test/cpp/common/main.cpp
    ${DIST_AUTOGRAD_TEST_DIR}/test_dist_autograd.cpp
  )

  add_executable(test_dist_autograd ${DIST_AUTOGRAD_TEST_SOURCES})
  target_include_directories(test_dist_autograd PRIVATE ${ATen_CPU_INCLUDE})
  target_link_libraries(test_dist_autograd PRIVATE torch gtest)

  if(USE_CUDA)
    target_link_libraries(test_dist_autograd PRIVATE
      ${CUDA_LIBRARIES}
      ${CUDA_NVRTC_LIB}
      ${CUDA_CUDA_LIB}
      ${TORCH_CUDA_LIBRARIES})

    # Make sure the CUDA stubs folder doesn't end up in the RPATH of test_dist_autograd:
    set(DIST_AUTOGRAD_LIBS ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB} ${CUDA_CUDA_LIB} ${TORCH_CUDA_LIBRARIES})
    foreach(LIB IN LISTS DIST_AUTOGRAD_LIBS)
      message("LIB: ${LIB}")
      if(LIB MATCHES "stubs")
        message("Filtering ${LIB} from being set in caffe2_nvrtc's RPATH, because it appears to point to the CUDA stubs directory, which should not be RPATHed.")
      else()
        cmake_path(GET LIB PARENT_PATH LIB_PATH)
        message("LIBPATH: ${LIB_PATH}")
        list(APPEND DIST_AUTOGRAD_RPATH ${LIB_PATH})
      endif()
    endforeach()
    message("DIST_AUTOGRAD_RPATH: ${DIST_AUTOGRAD_RPATH}")
    set_target_properties(test_dist_autograd PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)
    set_target_properties(test_dist_autograd PROPERTIES INSTALL_RPATH "${DIST_AUTOGRAD_RPATH}")

    target_compile_definitions(test_dist_autograd PRIVATE "USE_CUDA")
  endif()

  if(INSTALL_TEST)
    install(TARGETS test_dist_autograd DESTINATION bin)
    # Install PDB files for MSVC builds
    if(MSVC AND BUILD_SHARED_LIBS)
      install(FILES $<TARGET_PDB_FILE:test_dist_autograd> DESTINATION bin OPTIONAL)
    endif()
  endif()
endif()
