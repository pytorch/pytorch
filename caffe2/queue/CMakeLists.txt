# ---[ GPU files
# ------[ general GPU
file(GLOB tmp *_gpu.cc)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
# ------[ CUDA sources
file(GLOB tmp *.cu)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
# exclude test files
file(GLOB tmp *_test.cc)
exclude(Caffe2_GPU_SRCS "${Caffe2_GPU_SRCS}" ${tmp})

# ---[ HIP files.
file(GLOB_RECURSE tmp *_hip.cc)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${tmp})
# exclude test files
file(GLOB tmp *_test.cc)
exclude(Caffe2_HIP_SRCS "${Caffe2_HIP_SRCS}" ${tmp})

# ---[ CPU files.
file(GLOB tmp *.cc)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
# exclude test, gpu and hip files
file(GLOB tmp *_test.cc)
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_GPU_SRCS} ${Caffe2_HIP_SRCS})

# ---[ GPU test files
file(GLOB tmp *_gpu_test.cc)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${tmp})

# ---[ HI test files
file(GLOB_RECURSE tmp *_hip_test.cc)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} ${tmp})

# ---[ CPU test files
file(GLOB tmp *_test.cc)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})
exclude(Caffe2_CPU_TEST_SRCS "${Caffe2_CPU_TEST_SRCS}" ${Caffe2_GPU_TEST_SRCS} ${Caffe2_HIP_TEST_SRCS})

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} PARENT_SCOPE)

