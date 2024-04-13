if(USE_SNPE AND ANDROID)
  file(GLOB_RECURSE tmp *.cc)
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp} PARENT_SCOPE)
endif()
