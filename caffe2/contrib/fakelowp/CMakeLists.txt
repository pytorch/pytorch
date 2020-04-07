message(STATUS "Including fakelowp operators")

# ---[ CPU files.
file(GLOB_RECURSE tmp *.cc)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})

file(GLOB_RECURSE tmp *_test.cc)
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})
