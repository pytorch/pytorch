
add_executable(nvfuser_bench
  batch_norm.cpp
  bert.cpp
  gelu_backward.cpp
  layer_norm.cpp
  lstm_cell.cpp
  reduction.cpp
  softmax.cpp
  scale_bias_relu.cpp
  main.cpp)

target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
