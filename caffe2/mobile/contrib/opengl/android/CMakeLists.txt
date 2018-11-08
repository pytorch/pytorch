file(GLOB_RECURSE tmp *.cc *.c)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp} PARENT_SCOPE)
