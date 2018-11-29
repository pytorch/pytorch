if(USE_OBSERVERS)
  message(STATUS "Include Observer library")
  set(Caffe2_CONTRIB_OBSERVERS_CPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/time_observer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/runcnt_observer.cc"
  )

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_CONTRIB_OBSERVERS_CPU_SRC})
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)

  # ---[ GPU test files
  file(GLOB tmp *_gpu_test.cc)
  set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${tmp})

  # ---[ HIP test files
  file(GLOB tmp hip/*_test.cc)
  set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} ${tmp})

  # ---[ CPU test files
  file(GLOB tmp *_test.cc)
  set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})
  exclude(Caffe2_CPU_TEST_SRCS "${Caffe2_CPU_TEST_SRCS}" ${Caffe2_GPU_TEST_SRCS})
  exclude(Caffe2_CPU_TEST_SRCS "${Caffe2_CPU_TEST_SRCS}" ${Caffe2_HIP_TEST_SRCS})

  # ---[ Send the lists to the parent scope.
  set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
  set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
  set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} PARENT_SCOPE)
endif()
