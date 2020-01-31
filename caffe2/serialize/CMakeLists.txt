file(GLOB tmp *_test.cc)

set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})

CHECK_CXX_COMPILER_FLAG(-mpclmul COMPILER_HAS_M_PCLMUL)
CHECK_CXX_COMPILER_FLAG(-msse4.2 COMPILER_HAS_M_SSE42)
if (COMPILER_HAS_M_PCLMUL AND COMPILER_HAS_M_SSE42)
  set(SSE42_FILES
      ${PROJECT_SOURCE_DIR}/third_party/miniz-2.0.8/CrcHw.cpp)
  set_source_files_properties(
    ${SSE42_FILES}
    PROPERTIES COMPILE_FLAGS "-mpclmul -msse4.2 ")
  #target_compile_options(torch_cpu PUBLIC "-DUSE_EXTERNAL_MZCRC=1")
endif()

list(APPEND Caffe2_CPU_SRCS
  ${PROJECT_SOURCE_DIR}/third_party/miniz-2.0.8/miniz.c
  ${SSE42_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/inline_container.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/istream_adapter.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/file_adapter.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/read_adapter_interface.cc)
list(APPEND Caffe2_CPU_INCLUDE ${PROJECT_SOURCE_DIR}/third_party/miniz-2.0.8)

if (COMPILER_HAS_M_PCLMUL AND COMPILER_HAS_M_SSE42)
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/third_party/miniz-2.0.8/miniz.c
    APPEND PROPERTIES COMPILE_FLAGS "-DUSE_EXTERNAL_MZCRC ")
endif()


set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_INCLUDE ${Caffe2_CPU_INCLUDE} PARENT_SCOPE)
