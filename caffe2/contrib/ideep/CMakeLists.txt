if(USE_MKL AND MKL_USE_IDEEP)
  message(STATUS "Including IDEEP operators")

  # ---[ Enable AVX2
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

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
else()
  message(STATUS "Excluding ideep operators as we are not using ideep")
endif()
