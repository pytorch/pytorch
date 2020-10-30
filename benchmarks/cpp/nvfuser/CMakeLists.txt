add_executable(nvfuser_bench end_to_end.cpp main.cpp)
target_link_libraries(nvfuser_bench PRIVATE torch_library benchmark)
