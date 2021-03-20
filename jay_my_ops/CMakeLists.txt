
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(onnx_ops)

find_package(Torch REQUIRED)

# Define our library target
add_library(onnx_ops SHARED onnx_ops.cpp)
# Enable C++14
target_compile_features(onnx_ops PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(onnx_ops "${TORCH_LIBRARIES}")
