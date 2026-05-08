# Skip on windows
if(WIN32)
  return()
endif()

set(AOTI_TORCH_CHECK_TEST_ROOT ${TORCH_ROOT}/test/cpp/shim)

# Build the cpp gtest binary containing the cpp-only tests.
set(AOTI_TORCH_CHECK_TEST_SRCS
  ${AOTI_TORCH_CHECK_TEST_ROOT}/main.cpp
  ${AOTI_TORCH_CHECK_TEST_ROOT}/test_stable_ivalue.cpp
)

add_executable(test_shim
  ${AOTI_TORCH_CHECK_TEST_SRCS}
)

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_shim PRIVATE USE_GTEST)


target_link_libraries(test_shim PRIVATE gtest_main torch)
target_include_directories(test_shim PRIVATE ${ATen_CPU_INCLUDE})

if(INSTALL_TEST)
  install(TARGETS test_shim DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_shim> DESTINATION bin OPTIONAL)
  endif()
endif()
