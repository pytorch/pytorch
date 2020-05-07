# TODO: figure out conflict between contrib/nnpack/nnpack_ops.cc and mobile_nnpack.cc
if(IOS)
  # Basic ios srcs.
  set(Caffe2_CONTRIB_IOS_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/ios_caffe.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/ios_caffe_predictor.cc"
    # "${CMAKE_CURRENT_SOURCE_DIR}/mobile_nnpack.cc"
    )
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_CONTRIB_IOS_SRC})

  if(USE_METAL)
    # metal/mpscnn files
    add_subdirectory(mpscnn)
  endif()
endif()

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
