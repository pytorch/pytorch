if(USE_NNPACK)
  message(STATUS "Include NNPACK operators")
  set(Caffe2_CONTRIB_NNPACK_CPU_SRC
      "${CMAKE_CURRENT_SOURCE_DIR}/nnpack_ops.cc"
  )

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_CONTRIB_NNPACK_CPU_SRC} PARENT_SCOPE)
endif()
