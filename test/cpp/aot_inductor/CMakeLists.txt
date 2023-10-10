
set(AOT_INDUCTOR_TEST_ROOT ${TORCH_ROOT}/test/cpp/aot_inductor)

# Build the cpp gtest binary containing the cpp-only tests.
set(INDUCTOR_TEST_SRCS
  ${AOT_INDUCTOR_TEST_ROOT}/test.cpp
)

add_executable(test_aot_inductor
  ${TORCH_ROOT}/test/cpp/common/main.cpp
  ${INDUCTOR_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aot_inductor PRIVATE USE_GTEST)

# Define a custom command to generate the library
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libmodel.so
        COMMAND python ${AOT_INDUCTOR_TEST_ROOT}/test.py
        DEPENDS ${AOT_INDUCTOR_TEST_ROOT}/test.py
)
add_custom_target(model_so ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libmodel.so)
add_dependencies(test_aot_inductor model_so)

target_link_libraries(test_aot_inductor PRIVATE
  torch
  gtest
  ${CMAKE_CURRENT_BINARY_DIR}/libmodel.so
)

if(USE_CUDA)
  target_include_directories(test_aot_inductor PRIVATE ${ATen_CUDA_INCLUDE})

  target_compile_definitions(test_aot_inductor PRIVATE USE_CUDA CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR})
endif()

if(INSTALL_TEST)
  install(TARGETS test_aot_inductor DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aot_inductor> DESTINATION bin OPTIONAL)
  endif()
endif()
