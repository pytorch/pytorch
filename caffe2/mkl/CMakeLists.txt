message(STATUS "caffe2/mkl/CMakeLists.txt")
message(STATUS "caffe2/mkl/CMakeLists.txt USE_MKL is ${USE_MKL}")
if(USE_MKL AND USE_MKLML)
  message(STATUS "Including MKL operators")

  # ---[ CPU files.
  file(GLOB_RECURSE tmp *.cc)
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
  # exclude test files and gpu files
  file(GLOB_RECURSE tmp *_test.cc)
  exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})

  # ---[ CPU test files - currently none but just to be safe
  file(GLOB_RECURSE tmp *_test.cc)
  set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})

  # ---[ Send the lists to the parent scope.
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
  set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
elseif(USE_MKLML)
  message(STATUS "caffe2/mkl/CMakeLists.txt wouldn't included MKLML if not for USE_MKL")
else()
  message(STATUS "Excluding mkl operators as we are not using mkl")
endif()
