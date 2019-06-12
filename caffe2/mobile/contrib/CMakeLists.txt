add_subdirectory(ios)
# [FIX later or remove] opengl code will be broken because of tensor refactoring, remove this from CI to unblock
if(USE_MOBILE_OPENGL AND (ANDROID OR IOS))
  # add_subdirectory(opengl)
endif()
if (USE_ACL)
  # add_subdirectory(arm-compute)
endif()
# Finally pass the src lists back to the parent

if (USE_NNAPI)
  add_subdirectory(nnapi)
endif()

# CPU source, test sources, binary sources
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS} PARENT_SCOPE)

# GPU source, test sources, binary sources
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS} PARENT_SCOPE)
