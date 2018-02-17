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

  set(Caffe2_CONTRIB_GLOO_GPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_ops_gpu.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/broadcast_ops_gpu.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/common_world_ops_gpu.cc"
    )

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_CONTRIB_GLOO_CPU_SRC} PARENT_SCOPE)
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${Caffe2_CONTRIB_GLOO_GPU_SRC} PARENT_SCOPE)
endif()
