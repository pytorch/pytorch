cmake_minimum_required(VERSION 3.1)

project(op_deps_project)

set(CMAKE_CXX_STANDARD 14)

# Find torch library
find_package(Torch REQUIRED)

# Small test op library
add_library(OpLib
  quantized_ops.cpp
  simple_ops.cpp
  utils.cpp
)
target_include_directories(OpLib PUBLIC ${TORCH_INCLUDE_DIRS})

# Main executable
add_executable(main main.cc)

find_package(Threads REQUIRED)

target_link_libraries(main
  -Wl,--gc-sections
  -Wl,--whole-archive
  OpLib
  ${TORCH_LIBRARIES}
  -Wl,--no-whole-archive
  Threads::Threads
)

# Only install core libraries for downstream analysis job.
install(TARGETS OpLib DESTINATION lib)
install(FILES ${C10_LIBRARY} DESTINATION lib)
