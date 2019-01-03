add_subdirectory(aten)
add_subdirectory(gloo)
add_subdirectory(nccl)
add_subdirectory(opencl)
add_subdirectory(prof)
add_subdirectory(shm_mutex)
if (USE_TENSORRT)
  add_subdirectory(tensorrt)
endif()

# Pass the src lists back to the parent

# CPU source, include, deps, test sources, binary sources
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)

# GPU source, include, deps, test sources, binary sources
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
