set(DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(interpreter)

set(INTERPRETER_TEST_SOURCES
  ${DEPLOY_DIR}/test_deploy.cpp
)
add_executable(test_deploy ${INTERPRETER_TEST_SOURCES})
target_include_directories(test_deploy PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy PUBLIC gtest dl)
# no-as-needed to ensure shm and torch are included to satisfy runtime dlopen
# dependencies for libinterpreter, regardless of whether they are used in interpreter_test
target_link_libraries(test_deploy PUBLIC "-Wl,--no-as-needed" shm torch protobuf::libprotobuf-lite)


add_executable(deploy_benchmark ${DEPLOY_DIR}/example/benchmark.cpp)
target_include_directories(deploy_benchmark PRIVATE ${PYTORCH_ROOT}/torch)
# no-as-needed to ensure shm and torch are included to satisfy runtime dlopen
# dependencies for libinterpreter, regardless of whether they are used in interpreter_test
target_link_libraries(deploy_benchmark PUBLIC "-Wl,--no-as-needed" shm torch protobuf::libprotobuf-lite)
