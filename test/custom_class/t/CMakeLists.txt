cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(t)

find_package(Torch REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/pybind11/ ./pybind11)

# Define our library target
pybind11_add_module(t SHARED test_libtorch.cpp)

# Link against LibTorch
target_link_libraries(t PRIVATE "${TORCH_LIBRARIES}")

