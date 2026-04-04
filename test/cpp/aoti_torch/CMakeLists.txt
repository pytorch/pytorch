# Skip on windows
if(WIN32)
  return()
endif()

set(AOTI_TORCH_CHECK_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_torch)

# Build the cpp gtest binary containing the cpp-only tests.
set(AOTI_TORCH_CHECK_TEST_SRCS
  ${AOTI_TORCH_CHECK_TEST_ROOT}/main.cpp
  ${AOTI_TORCH_CHECK_TEST_ROOT}/test_stable_ivalue.cpp
)

add_executable(test_aoti_torch
  ${AOTI_TORCH_CHECK_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aoti_torch PRIVATE USE_GTEST)


target_link_libraries(test_aoti_torch PRIVATE gtest_main   torch)
target_include_directories(test_aoti_torch PRIVATE ${ATen_CPU_INCLUDE})


# Disable unused-variable warnings for variables that are only used to test compilation
target_compile_options_if_supported(test_aoti_torch -Wno-unused-variable)
target_compile_options_if_supported(test_aoti_torch -Wno-unused-but-set-variable)
# Add -Wno-dangling-pointer for GCC 13
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
  target_compile_options_if_supported(test_aoti_torch -Wno-dangling-pointer)
endif()

if(INSTALL_TEST)
  install(TARGETS test_aoti_torch DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_aoti_torch> DESTINATION bin OPTIONAL)
  endif()
endif()
