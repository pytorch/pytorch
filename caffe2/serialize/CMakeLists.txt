file(GLOB tmp *_test.cc)

set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})
list(APPEND Caffe2_CPU_SRCS
  ${PROJECT_SOURCE_DIR}/third_party/miniz-3.0.2/miniz.c
  ${CMAKE_CURRENT_SOURCE_DIR}/inline_container.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/istream_adapter.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/file_adapter.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/crc.cc)
list(APPEND Caffe2_CPU_INCLUDE ${PROJECT_SOURCE_DIR}/third_party/miniz-3.0.2)

set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_INCLUDE ${Caffe2_CPU_INCLUDE} PARENT_SCOPE)
