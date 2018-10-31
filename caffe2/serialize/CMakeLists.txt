set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS}
  serialize/intermediate_model.cc)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS}
  serialize/intermediate_model_test.cc)

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
