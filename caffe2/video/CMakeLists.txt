if(USE_OPENCV AND OpenCV_FOUND AND USE_FFMPEG AND FFMPEG_FOUND)
        message(STATUS "Including video processing operators")
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

  # ---[ HIP files
  # ------[ general HIP
  file(GLOB tmp hip/*.cc)
  set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${tmp})
  # ------[ HIP sources
  file(GLOB tmp hip/*.hip)
  set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${tmp})
  # exclude test files
  file(GLOB tmp hip/*_test.cc)
  exclude(Caffe2_HIP_SRCS "${Caffe2_HIP_SRCS}" ${tmp})

  # ---[ CPU files.
  file(GLOB tmp *.cc)
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
  # exclude test files and gpu files
  file(GLOB tmp *_test.cc)
  exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})
  exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_GPU_SRCS})
  exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_HIP_SRCS})

  # ---[ GPU test files
  file(GLOB tmp *_gpu_test.cc)
  set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${tmp})

  # ---[ HIP test files
  file(GLOB tmp hip/*_test.cc)
  set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} ${tmp})

  # ---[ CPU test files
  file(GLOB tmp *_test.cc)
  set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})
  exclude(Caffe2_CPU_TEST_SRCS "${Caffe2_CPU_TEST_SRCS}"
    ${Caffe2_GPU_TEST_SRCS})
  exclude(Caffe2_CPU_TEST_SRCS "${Caffe2_CPU_TEST_SRCS}"
    ${Caffe2_GPU_TEST_SRCS})

  # ---[ Send the lists to the parent scope.
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
  set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} PARENT_SCOPE)
  set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
  set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
  set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} PARENT_SCOPE)
else()
        message(STATUS "Excluding video processing operators due to no opencv")
endif()
