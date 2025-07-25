cmake_minimum_required(VERSION 3.15)

project(custom_build_project)

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING
          "The C++ standard whose features are requested to build this target.")

# Find torch library
find_package(Torch REQUIRED)

# Main executable
add_executable(Predictor predictor.cpp)
target_include_directories(Predictor PUBLIC ${TORCH_INCLUDE_DIRS})

find_package(Threads REQUIRED)

target_link_libraries(
  Predictor
  -Wl,-s
  -Wl,--gc-sections
  -Wl,--whole-archive
  ${TORCH_LIBRARIES}
  -Wl,--no-whole-archive
  Threads::Threads
  ${CMAKE_DL_LIBS})
