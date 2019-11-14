cmake_minimum_required(VERSION 3.1)

project(op_deps_project)

# Find torch library
find_package(Torch REQUIRED)

# Small test op library
add_library(SimpleOps
  simple_ops.cpp
  utils.cpp
)
target_include_directories(SimpleOps PUBLIC ${TORCH_INCLUDE_DIRS})

# Main executable
add_executable(main main.cc)

find_package(Threads REQUIRED)

target_link_libraries(main
  -Wl,--gc-sections
  -Wl,--whole-archive
  SimpleOps
  ${TORCH_LIBRARIES}
  -Wl,--no-whole-archive
  Threads::Threads
)

# Only install core libraries for downstream analysis job.
install(TARGETS SimpleOps DESTINATION lib)
install(FILES ${C10_LIBRARY} DESTINATION lib)
