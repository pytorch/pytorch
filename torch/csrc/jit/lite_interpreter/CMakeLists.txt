add_subdirectory(${PROJECT_SOURCE_DIR}/aten/src/ATen/core ATen_core)

set(LITE_EXECUTOR_SRCS
    ${PROJECT_SOURCE_DIR}/torch/csrc/jit/lite_interpreter/import_bytecode.cpp
    ${PROJECT_SOURCE_DIR}/torch/csrc/jit/lite_interpreter/bytecode.cpp
    ${PROJECT_SOURCE_DIR}/aten/src/ATen/core/ivalue.cpp
    ${PROJECT_SOURCE_DIR}/torch/csrc/jit/pickler.cpp
    ${PROJECT_SOURCE_DIR}/torch/csrc/jit/instruction.cpp
    ${PROJECT_SOURCE_DIR}/torch/csrc/jit/lite_interpreter/register_ins_ops.cpp
    )

add_library(lite_interpreter ${LITE_EXECUTOR_SRCS})

if(APPLE)
  target_link_libraries(lite_interpreter PRIVATE -Wl,-force_load torch)
elseif(MSVC)
  target_link_libraries(lite_interpreter PRIVATE -WHOLEARCHIVE:torch)
else()
  target_link_libraries(lite_interpreter PRIVATE -Wl,--whole-archive,torch -Wl,--no-whole-archive)
endif()

target_include_directories(lite_interpreter
    PUBLIC
    ${PROJECT_SOURCE_DIR}/caffe2/serialize/
    ${PROJECT_BINARY_DIR}/aten/src/
    )
