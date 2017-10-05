set(Caffe2_CPU_OBSERVER_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/net_observer_reporter_print.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/observer_config.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/perf_observer.cc"
  )

set(Caffe2_CPU_OBSERVER_SRCS ${Caffe2_CPU_OBSERVER_SRCS} PARENT_SCOPE)
