cmake_minimum_required(VERSION 3.1)

project(op_deps_project)

# Find torch library
add_library(torch UNKNOWN IMPORTED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARIES})

# Small test op library
add_library(SimpleOps
  simple_ops.cpp
  utils.cpp
)

# Main executable
add_executable(main main.cc)

# Need manually link all these static libraries:
find_library(torch torch PATHS "${TORCH_LIBRARIES}")
find_library(c10 c10 PATHS "${TORCH_LIBRARIES}")
find_library(nnpack nnpack PATHS "${TORCH_LIBRARIES}")
find_library(pytorch_qnnpack pytorch_qnnpack PATHS "${TORCH_LIBRARIES}")
find_library(eigen_blas eigen_blas PATHS "${TORCH_LIBRARIES}")
find_library(cpuinfo cpuinfo PATHS "${TORCH_LIBRARIES}")
find_library(clog clog PATHS "${TORCH_LIBRARIES}")

find_package(Threads REQUIRED)

target_link_libraries(main
  -Wl,--gc-sections
  -Wl,--whole-archive
  SimpleOps
  ${torch}
  -Wl,--no-whole-archive
  ${c10}
  ${nnpack}
  ${pytorch_qnnpack}
  ${eigen_blas}
  ${cpuinfo}
  ${clog}
  Threads::Threads
)

# Only install core libraries for downstream analysis job.
install(TARGETS SimpleOps DESTINATION lib)
install(FILES ${c10} DESTINATION lib)
