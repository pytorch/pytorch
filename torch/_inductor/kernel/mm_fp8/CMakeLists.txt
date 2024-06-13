cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(_fp8_mm_cuda)

# Define library target
add_library(_fp8_mm_cuda SHARED mm_fp8.cpp)

# Enable C++14
target_compile_features(_fp8_mm_cuda PRIVATE cxx_std_14)

# Link against LibTorch
target_link_libraries(_fp8_mm_cuda "${TORCH_LIBRARIES}")

# Link against c10
target_link_libraries(_fp8_mm_cuda libc10)
