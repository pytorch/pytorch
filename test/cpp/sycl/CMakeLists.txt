set(TEST_SYCL_ROOT ${PROJECT_SOURCE_DIR}/test/cpp/sycl)
set(TEST_SYCL_SYCL_SRCS ${TEST_SYCL_ROOT}/simple_kernel.cpp)
set(TEST_SYCL_CXX_SRCS ${TEST_SYCL_ROOT}/test_simple_kernel.cpp)
set(TEST_SYCL_MAIN_SRCS ${TEST_SYCL_ROOT}/main.cpp)

# test_sycl_standalone
sycl_add_executable(
  test_sycl_standalone
  SYCL_SOURCES ${TEST_SYCL_SYCL_SRCS}
  CXX_SOURCES ${TEST_SYCL_MAIN_SRCS} ${TEST_SYCL_CXX_SRCS})
target_link_libraries(test_sycl_standalone gtest)

if(INSTALL_TEST)
  install(TARGETS test_sycl_standalone DESTINATION bin)
endif()

# test_sycl_shared
sycl_add_library(
  sycl_simple_kernel_test
  SHARED
  SYCL_SOURCES ${TEST_SYCL_SYCL_SRCS})

add_executable(
  test_sycl_shared
  ${TEST_SYCL_MAIN_SRCS}
  ${TEST_SYCL_CXX_SRCS})
target_link_libraries(test_sycl_shared gtest)
target_link_libraries(test_sycl_shared sycl_simple_kernel_test)
set_target_properties(test_sycl_shared PROPERTIES INSTALL_RPATH $<TARGET_FILE_DIR:sycl_simple_kernel_test>)

if(INSTALL_TEST)
  install(TARGETS test_sycl_shared DESTINATION bin)
  install(TARGETS sycl_simple_kernel_test DESTINATION lib)
endif()
