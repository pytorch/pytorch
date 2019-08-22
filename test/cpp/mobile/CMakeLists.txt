add_executable(test_bc_export
  test_bc_export.cpp)

caffe2_interface_library(torch torch_whole)
target_link_libraries(test_bc_export PRIVATE torch_whole gtest)
target_include_directories(test_bc_export PRIVATE ${ATen_CPU_INCLUDE})
#target_compile_definitions(test_bc_export PRIVATE USE_GTEST)

if (USE_CUDA)
  target_link_libraries(test_bc_export PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_bc_export PRIVATE USE_CUDA)
endif()

if (INSTALL_TEST)
  install(TARGETS test_bc_export DESTINATION bin)
  # Install PDB files for MSVC builds
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_bc_export> DESTINATION bin OPTIONAL)
  endif()
endif()
