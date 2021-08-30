set(
  MODEL_TRACER_DIR
  "${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer")
set(MODEL_TRACER_DIR
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/tracer.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/TracerRunner.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/TensorUtils.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/MobileModelRunner.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.cpp
  ${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer/KernelDTypeTracer.cpp
)

add_executable(
  model_tracer
  ${MODEL_TRACER_DIR})

target_link_libraries(model_tracer PRIVATE torch)

install(TARGETS model_tracer DESTINATION bin)
