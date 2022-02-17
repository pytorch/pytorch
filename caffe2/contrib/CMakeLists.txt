add_subdirectory(aten)
add_subdirectory(nccl)
add_subdirectory(opencl)
add_subdirectory(prof)
add_subdirectory(shm_mutex)
add_subdirectory(fakelowp)
if(USE_TENSORRT)
add_subdirectory(tensorrt)
endif()

# Only build Gloo Caffe2 ops on Linux, as it hardcodes
# the Linux-specific `gloo::transport::tcp` namespace.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
add_subdirectory(gloo)
endif()

# Pass the src lists back to the parent

# CPU source, include, deps, test sources, binary sources
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_INCLUDE ${Caffe2_CPU_INCLUDE} PARENT_SCOPE)
set(Caffe2_DEPENDENCY_LIBS ${Caffe2_DEPENDENCY_LIBS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS} PARENT_SCOPE)

# GPU source, include, deps, test sources, binary sources
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_INCLUDE ${Caffe2_GPU_INCLUDE} PARENT_SCOPE)
set(Caffe2_CUDA_DEPENDENCY_LIBS ${Caffe2_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS} PARENT_SCOPE)

# HIP sources, include, test sources
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_INCLUDE ${Caffe2_HIP_INCLUDE} PARENT_SCOPE)
set(Caffe2_HIP_DEPENDENCY_LIBS ${Caffe2_HIP_DEPENDENCY_LIBS} PARENT_SCOPE)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} PARENT_SCOPE)
