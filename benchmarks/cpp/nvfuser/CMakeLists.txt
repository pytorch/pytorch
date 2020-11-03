add_executable(nvfuser_bench lstm_cell.cpp main.cpp)
target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
