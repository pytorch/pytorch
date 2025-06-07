cmake_minimum_required(VERSION 3.5)

# Project name and language
project(pytorch_vision_jni CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Print build commands
set(CMAKE_VERBOSE_MAKEFILE ON)

# Define source directory
set(pytorch_vision_cpp_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)

# Find all sources
file(GLOB pytorch_vision_SOURCES
  ${pytorch_vision_cpp_DIR}/pytorch_vision_jni.cpp
)

# Define JNI library
add_library(pytorch_vision_jni SHARED
    ${pytorch_vision_SOURCES}
)

# Enable exceptions
target_compile_options(pytorch_vision_jni PRIVATE -fexceptions)

# Set ABI-specific build subdir (important for Android NDK)
set(BUILD_SUBDIR ${ANDROID_ABI})

# Link native dependencies (update as needed for your project)
find_library(log-lib log)

# Add any other required libraries like torch, torchvision, etc.
# Example for linking against PyTorch Android libs (edit paths as needed):
# target_link_libraries(pytorch_vision_jni
#   "${TORCH_LIB_DIR}/libtorch_android.so"
#   "${TORCHVISION_LIB_DIR}/libtorchvision_android.so"
#   ${log-lib}
# )

target_link_libraries(pytorch_vision_jni
  ${log-lib}
)
