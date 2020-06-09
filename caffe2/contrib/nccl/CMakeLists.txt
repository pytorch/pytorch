if(USE_NCCL)
    if(USE_CUDA)
      message(STATUS "Include NCCL operators")
      set(Caffe2_CONTRIB_NCCL_GPU_SRC
          "${CMAKE_CURRENT_SOURCE_DIR}/cuda_nccl_gpu.cc"
          "${CMAKE_CURRENT_SOURCE_DIR}/cuda_nccl_op_gpu.cc"
      )

      set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${Caffe2_CONTRIB_NCCL_GPU_SRC})
      set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
    endif(USE_CUDA)

    if(USE_ROCM)
      message(STATUS "Include AMD RCCL operators")
      set(Caffe2_CONTRIB_NCCL_HIP_SRC
          "${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_nccl_gpu.cc"
          "${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_nccl_op_gpu.cc"
      )

      set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${Caffe2_CONTRIB_NCCL_HIP_SRC})
      set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} PARENT_SCOPE)
    endif(USE_ROCM)
else()
  message(STATUS "NCCL operators skipped due to no CUDA support")
endif()
