add_subdirectory(aten)
add_subdirectory(gloo)
add_subdirectory(nccl)
add_subdirectory(prof)
add_subdirectory(shm_mutex)
add_subdirectory(script)
# Finally pass the src lists back to the parent

# CPU source, test sources, binary sources
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS} PARENT_SCOPE)

# GPU source, test sources, binary sources
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS} PARENT_SCOPE)
