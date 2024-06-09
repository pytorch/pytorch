set(Caffe2_EMULATOR_CPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/data_filler.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_filler.cc"
)
set(Caffe2_EMULATOR_CPU_TEST_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/data_filler_test.cc")

# Common files that are always going to be included.
list(APPEND Caffe2_CPU_SRCS ${Caffe2_EMULATOR_CPU_SRC})
list(APPEND Caffe2_CPU_TEST_SRCS ${Caffe2_EMULATOR_CPU_TEST_SRC})

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
