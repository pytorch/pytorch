set(Caffe2_CPU_OBSERVER_SRCS)

add_subdirectory(observers)

add_library(Caffe2_CPU_OBSERVER STATIC ${Caffe2_CPU_OBSERVER_SRCS})
