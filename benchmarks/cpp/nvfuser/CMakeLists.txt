if(USE_CUDA)
  add_executable(nvfuser_bench
    batch_norm_channels_first.cpp
    batch_norm_channels_first_backward.cpp
    batch_norm_channels_last.cpp
    batch_norm_channels_last_backward.cpp
    bert.cpp
    broadcast.cpp
    gelu_backward.cpp
    heuristic_lookup.cpp
    shape_inference.cpp
    instance_norm.cpp
    layer_norm.cpp
    layer_norm_backward.cpp
    rms_norm.cpp
    rms_norm_backward.cpp
    lstm_cell.cpp
    reduction.cpp
    softmax.cpp
    softmax_backward.cpp
    scale_bias_relu.cpp
    transpose.cpp
    matmul.cpp
    timm.cpp
    utils.cpp
    main.cpp)

  target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
  if(NOT MSVC)
    target_compile_options_if_supported(nvfuser_bench -Werror)
    target_compile_options_if_supported(nvfuser_bench -Wno-unused-variable)
    target_compile_options_if_supported(nvfuser_bench -Wno-deprecated-copy)
  endif()

endif()
