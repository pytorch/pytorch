set(AOT_INDUCTOR_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_inference)

# Build custom TorchScript op for AOTInductor
add_library(aoti_custom_class SHARED aoti_custom_class.cpp)
set_target_properties(aoti_custom_class PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
if(USE_CUDA)
  target_compile_definitions(aoti_custom_class PRIVATE USE_CUDA)
elseif(USE_ROCM)
  target_compile_definitions(aoti_custom_class PRIVATE USE_ROCM)
endif()

# Link against LibTorch
target_link_libraries(aoti_custom_class torch)

# Build the cpp gtest binary containing the cpp-only tests.
set(INDUCTOR_TEST_SRCS
  ${AOT_INDUCTOR_TEST_ROOT}/test.cpp
)

add_executable(test_aoti_inference
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${INDUCTOR_TEST_SRCS}
)
add_dependencies(test_aoti_inference aoti_custom_class)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aoti_inference PRIVATE USE_GTEST)

target_link_libraries(test_aoti_inference PRIVATE
  torch
  gtest_main
  -Wl,--no-as-needed aoti_custom_class
)

if(USE_CUDA)
  target_include_directories(test_aoti_inference PRIVATE ${ATen_CUDA_INCLUDE})
  target_compile_definitions(test_aoti_inference PRIVATE USE_CUDA)
elseif(USE_ROCM)
    target_include_directories(test_aoti_inference PRIVATE ${ATen_HIP_INCLUDE})
    target_compile_definitions(test_aoti_inference PRIVATE USE_ROCM)
endif()
target_compile_definitions(test_aoti_inference PRIVATE
    CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_options_if_supported(test_aoti_inference -Wno-unused-variable)
target_compile_options_if_supported(test_aoti_inference -Wno-unused-but-set-variable)
target_compile_options_if_supported(test_aoti_inference -Wno-unused-function)

if(INSTALL_TEST)
  install(TARGETS test_aoti_inference DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aoti_inference> DESTINATION bin OPTIONAL)
  endif()
endif()
