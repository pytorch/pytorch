if(USE_CUDA)
  add_executable(nvfuser_bench
    batch_norm.cpp
    batch_norm_backward.cpp
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
    utils.cpp
    main.cpp)

  target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
  if(NOT MSVC)
    target_compile_options(nvfuser_bench PRIVATE -Wno-unused-variable)
  endif()

endif()
