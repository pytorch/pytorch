if(USE_METAL)
  file(GLOB_RECURSE tmp *.mm *.cc)
  # exclude test files
  file(GLOB_RECURSE test_files *_test.cc)
  exclude(tmp "${tmp}" ${test_files})
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp} PARENT_SCOPE)
endif()
