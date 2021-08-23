if(USE_CUDA)
  add_executable(nvfuser_bench
    batch_norm.cpp
    bert.cpp
    broadcast.cpp
    gelu_backward.cpp
    heuristic_lookup.cpp
    instance_norm.cpp
    layer_norm.cpp
    lstm_cell.cpp
    reduction.cpp
    softmax.cpp
    scale_bias_relu.cpp
    utils.cpp
    main.cpp)

  target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
endif()
