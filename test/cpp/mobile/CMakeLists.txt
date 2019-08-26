add_executable(test_bc_export
  test_bc_export.cpp)

if(APPLE)
  target_link_libraries(test_bc_export PRIVATE -Wl,-force_load torch)
elseif(MSVC)
  target_link_libraries(test_bc_export PRIVATE -WHOLEARCHIVE:torch)
else()
  target_link_libraries(test_bc_export PRIVATE -Wl,--whole-archive,torch -Wl,--no-whole-archive)
endif()

target_link_libraries(test_bc_export PRIVATE gtest)
target_include_directories(test_bc_export PRIVATE ${ATen_CPU_INCLUDE})

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
