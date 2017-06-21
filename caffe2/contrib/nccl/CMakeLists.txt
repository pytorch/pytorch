if(USE_NCCL)
    message(STATUS "Include NCCL operators")
    set(Caffe2_CONTRIB_NCCL_GPU_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_nccl_gpu.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda_nccl_op_gpu.cc"
    )

    set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${Caffe2_CONTRIB_NCCL_GPU_SRC})
    set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
else()
  message(STATUS "NCCL operators skipped due to no CUDA support")
endif()
