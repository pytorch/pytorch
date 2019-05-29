cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(t)

find_package(Torch REQUIRED)

# Define our library target
add_library(t SHARED test_libtorch.cpp)
# Enable C++11
target_compile_features(t PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(t "${TORCH_LIBRARIES}")
