if(USE_GLOO)
  set(Caffe2_CONTRIB_GLOO_CPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/allgather_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/barrier_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/common.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_world_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/context.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter_ops.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/store_handler.cc"
    )
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_CONTRIB_GLOO_CPU_SRC} PARENT_SCOPE)

  if(USE_CUDA)
    set(Caffe2_CONTRIB_GLOO_GPU_SRC
      "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_ops_gpu.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_ops_gpu.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/common_world_ops_gpu.cc"
      )
    set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${Caffe2_CONTRIB_GLOO_GPU_SRC} PARENT_SCOPE)
  endif(USE_CUDA)

  if(USE_ROCM)
    set(Caffe2_CONTRIB_GLOO_HIP_SRC
      "${CMAKE_CURRENT_SOURCE_DIR}/hip/allreduce_ops_gpu.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/hip/broadcast_ops_gpu.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/hip/common_world_ops_gpu.cc"
      )
    set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${Caffe2_CONTRIB_GLOO_HIP_SRC} PARENT_SCOPE)
    set(Caffe2_HIP_INCLUDE ${GLOO_HIP_INCLUDE} ${Caffe2_HIP_INCLUDE} PARENT_SCOPE)
  endif(USE_ROCM)
endif()
