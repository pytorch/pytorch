set(TEST_SYCL_ROOT ${PROJECT_SOURCE_DIR}/test/cpp/sycl)
set(TEST_SYCL_SYCL_SRCS ${TEST_SYCL_ROOT}/simple_kernel.cpp)
set(TEST_SYCL_CXX_SRCS ${TEST_SYCL_ROOT}/test_simple_kernel.cpp)
set(TEST_SYCL_MAIN_SRCS ${TEST_SYCL_ROOT}/main.cpp)

# test building executable
sycl_add_executable(
  test_sycl_build_standalone
  SYCL_SOURCES ${TEST_SYCL_SYCL_SRCS}
  CXX_SOURCES ${TEST_SYCL_MAIN_SRCS} ${TEST_SYCL_CXX_SRCS})
target_link_libraries(test_sycl_build_standalone gtest)

if(INSTALL_TEST)
  install(TARGETS test_sycl_build_standalone DESTINATION bin)
endif()

# test building archive static library
sycl_add_library(
  sycl_simple_kernel_test
  STATIC
  SYCL_SOURCES ${TEST_SYCL_SYCL_SRCS})

add_executable(
  test_sycl_build_archive
  ${TEST_SYCL_MAIN_SRCS}
  ${TEST_SYCL_CXX_SRCS})
target_link_libraries(test_sycl_build_archive gtest)
target_link_libraries(test_sycl_build_archive
  -Wl,--whole-archive
  -L$<TARGET_FILE_DIR:sycl_simple_kernel_test>
  -l:libsycl_simple_kernel_test.a
  -Wl,--no-whole-archive)
# ``SYCL_LIBRARIES`` is included in ``INTERFACE_LINK_LIBRARIES`` of
# sycl_simple_kernel_test. Since not using,
# ``
# target_link_libraries(
#   test_sycl_build_archive
#   sycl_simple_kernel_test)
# ``
# but explicit linkage options here for '--whole-archive' which is required
# by those objects wrapping SYCL device object modules in the archive library.
# ``SYCL_LIBRARIES`` is added as link library explicitly.
target_link_libraries(test_sycl_build_archive ${SYCL_LIBRARIES})

if(INSTALL_TEST)
  install(TARGETS test_sycl_build_archive DESTINATION bin)
endif()
