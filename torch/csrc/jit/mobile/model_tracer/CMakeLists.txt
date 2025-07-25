set(MODEL_TRACER_DIR "${TORCH_ROOT}/torch/csrc/jit/mobile/model_tracer")

list(APPEND MODEL_TRACER_SOURCES "")

append_filelist("torch_mobile_tracer_sources" MODEL_TRACER_SOURCES)

add_executable(model_tracer ${MODEL_TRACER_SOURCES})

target_link_libraries(model_tracer PRIVATE torch)

install(TARGETS model_tracer DESTINATION bin)
