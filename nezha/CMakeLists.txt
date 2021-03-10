
cmake_minimum_required(VERSION 3.18)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
# find_package(OpenCV REQUIRED)
project(ort_ops)

find_package(Torch REQUIRED)
add_library(ort_ops SHARED core_op.cpp)
target_compile_features(ort_ops PRIVATE cxx_range_for)
target_link_libraries(ort_ops PRIVATE "${TORCH_LIBRARIES}")


# //No help, variable specified on the command line.
# CMAKE_CXX_COMPILER=/bin/g++-9

# //No help, variable specified on the command line.
# CMAKE_C_COMPILER=/bin/gcc-9

# cmake_minimum_required(VERSION 3.1 FATAL_ERROR)


# if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
#   set(LINUX TRUE)
# else()
#   set(LINUX FALSE)
# endif()

# project(nezha_module)

# find_package(Torch REQUIRED)

# # Define our library target
# add_library(nezha_module SHARED core_func.cpp)

# set(CMAKE_CXX_STANDARD 14)

# # Link against LibTorch
# target_link_libraries(nezha_module "${TORCH_LIBRARIES}")


# # # Define our library target
# # add_library(warp_perspective SHARED op.cpp)
# # # Enable C++14
# # target_compile_features(warp_perspective PRIVATE cxx_std_14)
# # # Link against LibTorch
# # target_link_libraries(warp_perspective "${TORCH_LIBRARIES}")
# # # Link against OpenCV
# # target_link_libraries(warp_perspective opencv_core opencv_imgproc)