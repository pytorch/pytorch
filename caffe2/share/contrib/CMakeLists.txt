if(USE_NNPACK AND BUILD_CAFFE2_OPS)
  add_subdirectory(nnpack)
endif()
if(USE_ZSTD)
  add_subdirectory(zstd)
endif()
if(NOT MSVC)
  add_subdirectory(depthwise)
endif()

# CPU source, test sources, binary sources
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS} PARENT_SCOPE)

# GPU source, test sources, binary sources
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS} PARENT_SCOPE)
