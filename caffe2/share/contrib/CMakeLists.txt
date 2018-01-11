set(Caffe2_CPU_OBSERVER_SRCS)

if (USE_NNPACK)
  add_subdirectory(nnpack)
endif()
if (BUILD_OBSERVERS)
add_subdirectory(observers)
endif()
if (USE_ZSTD)
  add_subdirectory(zstd)
endif()

if (BUILD_OBSERVERS)
  add_library(Caffe2_CPU_OBSERVER STATIC ${Caffe2_CPU_OBSERVER_SRCS})
endif()

# CPU source, test sources, binary sources
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS} PARENT_SCOPE)

# GPU source, test sources, binary sources
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS} PARENT_SCOPE)
