# Builds libtorch_native_aot_cuda.so: the AOT native-embed kernel library
# that installs kernels on the at::native DispatchStubs (see ATen/NativeAotStubs.h) when
# loaded. Standalone project, NOT part of the main torch build: its inputs
# (generated aot_*.cpp + exported kernel .o under ARTIFACTS_DIR) are
# produced by tools/native_aot/export.py + gen_aot_lib.py, which need
# the DSL wheel and can run after or independently of the torch build.
#
# Configure (see build_aot_lib.sh for the driver):
#   cmake -S tools/native_aot -B build/native_aot/cmake
#         -DTORCH_ROOT=<checkout>/torch
#         -DARTIFACTS_DIR=<checkout>/build/native_aot
#         -DDSL_RUNTIME_STATIC=<...>/libcuda_dialect_runtime_static.a
cmake_minimum_required(VERSION 3.18)
project(torch_native_aot LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TORCH_ROOT)
  message(FATAL_ERROR "Pass -DTORCH_ROOT=<path to torch package dir>")
endif()
if(NOT ARTIFACTS_DIR)
  message(FATAL_ERROR "Pass -DARTIFACTS_DIR=<path to exported artifacts>")
endif()
if(NOT DSL_RUNTIME_STATIC)
  message(FATAL_ERROR "Pass -DDSL_RUNTIME_STATIC=<libcuda_dialect_runtime_static.a>")
endif()

find_package(CUDAToolkit REQUIRED)

# Plain GLOB, not GLOB_RECURSE: artifacts live exactly one level down
# (<op>/<file>), and recursing would pick up this project's own compiled
# objects under ${ARTIFACTS_DIR}/cmake/ on rebuilds (duplicate symbols).
file(GLOB aot_sources "${ARTIFACTS_DIR}/*/aot_*.cpp")
file(GLOB kernel_objects "${ARTIFACTS_DIR}/*/*.o")
if(NOT aot_sources)
  message(FATAL_ERROR "No aot_*.cpp under ${ARTIFACTS_DIR}; run export.py + gen_aot_lib.py first")
endif()
message(STATUS "aot sources: ${aot_sources}")
list(LENGTH kernel_objects n_kernels)
message(STATUS "kernel objects: ${n_kernels}")

add_library(torch_native_aot_cuda SHARED ${aot_sources} ${kernel_objects})
set_source_files_properties(${kernel_objects} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

# Per-op include dirs so each aot_*.cpp finds its kernels' exported headers.
file(GLOB op_dirs LIST_DIRECTORIES true "${ARTIFACTS_DIR}/*")
foreach(dir ${op_dirs})
  if(IS_DIRECTORY ${dir})
    target_include_directories(torch_native_aot_cuda PRIVATE ${dir})
  endif()
endforeach()

target_include_directories(torch_native_aot_cuda PRIVATE
  "${TORCH_ROOT}/include"
  "${TORCH_ROOT}/include/torch/csrc/api/include"
)
target_compile_definitions(torch_native_aot_cuda PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_link_directories(torch_native_aot_cuda PRIVATE "${TORCH_ROOT}/lib")
target_link_libraries(torch_native_aot_cuda PRIVATE
  "${DSL_RUNTIME_STATIC}"
  torch torch_cpu torch_cuda c10 c10_cuda
  CUDA::cudart CUDA::cuda_driver
)
set_target_properties(torch_native_aot_cuda PROPERTIES
  BUILD_RPATH "${TORCH_ROOT}/lib;$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  LIBRARY_OUTPUT_DIRECTORY "${ARTIFACTS_DIR}"
)

# Install next to the other torch libs so torch/_native auto-loads it.
install(TARGETS torch_native_aot_cuda LIBRARY DESTINATION "${TORCH_ROOT}/lib")
