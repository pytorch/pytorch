set(Caffe2_STORE_COMMON_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/file_store_handler.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/file_store_handler_op.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/store_handler.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/store_ops.cc"
)

set(Caffe2_STORE_COMMON_GPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/file_store_handler_op_gpu.cc"
)

set(Caffe2_STORE_REDIS_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/redis_store_handler.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/redis_store_handler_op.cc"
)

set(Caffe2_STORE_REDIS_GPU_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/redis_store_handler_op_gpu.cc"
)

# Common files that are always going to be included.
list(APPEND Caffe2_CPU_SRCS ${Caffe2_STORE_COMMON_SRC})
list(APPEND Caffe2_GPU_SRCS ${Caffe2_STORE_COMMON_GPU_SRC})

if(USE_REDIS)
  list(APPEND Caffe2_CPU_SRCS ${Caffe2_STORE_REDIS_SRC})
  list(APPEND Caffe2_GPU_SRCS ${Caffe2_STORE_REDIS_GPU_SRC})
endif()

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
