set(AOTI_ABI_CHECK_TEST_ROOT ${TORCH_ROOT}/test/cpp/aoti_abi_check)

# Build the cpp gtest binary containing the cpp-only tests.
set(AOTI_ABI_CHECK_TEST_SRCS
    ${AOTI_ABI_CHECK_TEST_ROOT}/main.cpp
    ${AOTI_ABI_CHECK_TEST_ROOT}/test_cast.cpp
    ${AOTI_ABI_CHECK_TEST_ROOT}/test_dtype.cpp
    ${AOTI_ABI_CHECK_TEST_ROOT}/test_macros.cpp
    ${AOTI_ABI_CHECK_TEST_ROOT}/test_math.cpp
    ${AOTI_ABI_CHECK_TEST_ROOT}/test_rand.cpp
    ${AOTI_ABI_CHECK_TEST_ROOT}/test_vec.cpp)

add_executable(test_aoti_abi_check ${AOTI_ABI_CHECK_TEST_SRCS})

# TODO temporary until we can delete the old gtest polyfills.
target_compile_definitions(test_aoti_abi_check PRIVATE USE_GTEST)

# WARNING: DO NOT LINK torch!!! The purpose is to check if the used aten/c10
# headers are writtern in a header-only way
target_link_libraries(test_aoti_abi_check PRIVATE gtest_main)
target_include_directories(test_aoti_abi_check PRIVATE ${ATen_CPU_INCLUDE})

if(INSTALL_TEST)
  install(TARGETS test_aoti_abi_check DESTINATION bin)
  # Install PDB files for MSVC builds
  if(MSVC AND BUILD_SHARED_LIBS)
    install(
      FILES $<TARGET_PDB_FILE:test_aoti_abi_check>
      DESTINATION bin
      OPTIONAL)
  endif()
endif()
