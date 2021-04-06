set(DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(interpreter)

add_custom_command(
  OUTPUT libtorch_deployinterpreter.o
  COMMAND cp $<TARGET_FILE:torch_deployinterpreter> .
  COMMAND ld -r -b binary -o libtorch_deployinterpreter.o libtorch_deployinterpreter.so
  COMMAND rm libtorch_deployinterpreter.so
  DEPENDS torch_deployinterpreter
  VERBATIM
)

add_library(torch_deploy libtorch_deployinterpreter.o ${DEPLOY_DIR}/deploy.cpp)
target_link_libraries(torch_deploy PUBLIC "-Wl,--no-as-needed" shm torch protobuf::libprotobuf-lite)


set(INTERPRETER_TEST_SOURCES
  ${DEPLOY_DIR}/test_deploy.cpp
)
add_executable(test_deploy ${INTERPRETER_TEST_SOURCES})
target_include_directories(test_deploy PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy PUBLIC gtest dl torch_deploy)

add_executable(deploy_benchmark ${DEPLOY_DIR}/example/benchmark.cpp)
target_include_directories(deploy_benchmark PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(deploy_benchmark PUBLIC torch_deploy)

