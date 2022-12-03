cmake_minimum_required(VERSION 3.1)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../..)
set(TEST_ROOT ${TORCH_ROOT}/test/edge)

add_executable(test_operator_registration
  ${TEST_ROOT}/test_operator_registration.cpp
  ${TEST_ROOT}/test_main.cpp
)

target_include_directories(test_operator_registration PRIVATE ${ATen_CPU_INCLUDE})

target_compile_definitions(test_operator_registration PRIVATE USE_GTEST)

set(TEST_DEPENDENCIES torch gtest)

target_link_libraries(test_operator_registration PRIVATE
  ${TEST_DEPENDENCIES}
)

if(INSTALL_TEST)
  install(TARGETS test_operator_registration DESTINATION bin)
endif()
