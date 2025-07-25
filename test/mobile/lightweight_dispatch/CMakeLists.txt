cmake_minimum_required(VERSION 3.15)

set(TORCH_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../..)
set(TEST_ROOT ${TORCH_ROOT}/test/mobile/lightweight_dispatch)

add_executable(test_codegen_unboxing ${TEST_ROOT}/test_lightweight_dispatch.cpp
                                     ${TEST_ROOT}/test_codegen_unboxing.cpp)

target_include_directories(test_codegen_unboxing PRIVATE ${ATen_CPU_INCLUDE})

target_compile_definitions(test_codegen_unboxing PRIVATE USE_GTEST)

set(TEST_UNBOXING_DEPENDENCIES torch gtest_main)

target_link_libraries(test_codegen_unboxing
                      PRIVATE ${TEST_UNBOXING_DEPENDENCIES})

if(INSTALL_TEST)
  install(TARGETS test_codegen_unboxing DESTINATION bin)
endif()
