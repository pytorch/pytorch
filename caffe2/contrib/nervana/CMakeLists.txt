if(USE_NERVANA_GPU)
  message(STATUS "Include Nervana operators")
  set(Caffe2_CONTRIB_NCCL_GPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nervana_c_api.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/nervana_fc_op_gpu.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/nervana_init_gpu.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/nervana_math_gpu.cc"
  )

  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${Caffe2_CONTRIB_NCCL_GPU_SRC})
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
endif()
