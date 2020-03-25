cmake_minimum_required(VERSION 3.1)

project(custom_build_project)

set(CMAKE_CXX_STANDARD 14)

# Find torch library
find_package(Torch REQUIRED)

# Main executable
add_executable(Predictor predictor.cpp)
target_include_directories(Predictor PUBLIC ${TORCH_INCLUDE_DIRS})

find_package(Threads REQUIRED)

target_link_libraries(Predictor
  -Wl,-s
  -Wl,--gc-sections
  -Wl,--whole-archive
  ${TORCH_LIBRARIES}
  -Wl,--no-whole-archive
  Threads::Threads
)
