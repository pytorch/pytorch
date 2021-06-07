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

if(USE_DISTRIBUTED)
  # Wrap linked dependencies as interface libraries.
  #
  # This ensures that the linker will not throw away any symbols that we may
  # need from `libtorch_deployinterpreter`. See the comment above the
  # definition of `caffe2_interface_library` for more info.
  #
  # For those coming from buck, this is the equivalent of setting
  # `link_whole=True` on the target library.
  caffe2_interface_library(c10d c10d_library)
  caffe2_interface_library(tensorpipe tensorpipe_library)

  target_link_libraries(torch_deploy PRIVATE c10d_library)
  if(USE_TENSORPIPE)
    target_link_libraries(torch_deploy PRIVATE tensorpipe_library)
  endif()
endif()

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
