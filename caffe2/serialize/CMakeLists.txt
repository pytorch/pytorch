file(GLOB tmp *_test.cc)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})

set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
