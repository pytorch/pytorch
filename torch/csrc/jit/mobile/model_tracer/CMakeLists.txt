set(
  MODEL_TRACER_DIR
  "${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer")
set(MODEL_TRACER_DIR
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/tracer.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/TensorUtils.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/MobileModelRunner.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.cpp
)

add_library(backend_with_compiler_runtime SHARED
        ${TORCH_ROOT}/test/cpp/jit/test_backend_compiler_lib.cpp
        ${TORCH_ROOT}/torch/csrc/jit/backends/backend_interface.cpp
        )
target_link_libraries(backend_with_compiler_runtime PRIVATE torch)

add_executable(
  model_tracer
  ${MODEL_TRACER_DIR})

target_link_libraries(model_tracer PRIVATE torch gtest)

install(TARGETS model_tracer DESTINATION bin)

# if(INSTALL_TEST)
#   install(TARGETS model_tracer DESTINATION bin)
#   # Install PDB files for MSVC builds
#   if(MSVC AND BUILD_SHARED_LIBS)
#     install(
#       FILES $<TARGET_PDB_FILE:test_lite_interpreter_runtime>
#       DESTINATION bin OPTIONAL)
#   endif()
# endif()
