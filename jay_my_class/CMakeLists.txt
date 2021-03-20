cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)

find_package(Torch REQUIRED)

# Define our library target
add_library(custom_class SHARED class.cpp onnx_class.cpp)
set(CMAKE_CXX_STANDARD 14)
set(PYTHON_INCLUDE_DIR "/home/jay/anaconda3/envs/dev/include/python3.8")
# Link against LibTorch
target_link_libraries(custom_class "${TORCH_LIBRARIES}")