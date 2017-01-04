set(Caffe2_DB_COMMON_CPU_SRC
    "/create_db_op.cc"
    "/protodb.cc"
)
set(Caffe2_DB_COMMON_GPU_SRC
    "/create_db_op_gpu.cc"
)
set(Caffe2_DB_LMDB_SRC
    "/lmdb.cc"
)
set(Caffe2_DB_LEVELDB_SRC
    "/leveldb.cc"
)
set(Caffe2_DB_ROCKSDB_SRC
    "/rocksdb.cc"
)
set(Caffe2_DB_ZMQDB_SRC
    "/zmqdb.cc"
)

prepend(Caffe2_DB_COMMON_CPU_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DB_COMMON_CPU_SRC})
prepend(Caffe2_DB_COMMON_GPU_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DB_COMMON_GPU_SRC})
prepend(Caffe2_DB_LMDB_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DB_LMDB_SRC})
prepend(Caffe2_DB_LEVELDB_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DB_LEVELDB_SRC})
prepend(Caffe2_DB_ROCKSDB_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DB_ROCKSDB_SRC})
prepend(Caffe2_DB_ZMQDB_SRC ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DB_ZMQDB_SRC})

# ----[ Assemble all sources depending on options
set(Caffe2_DB_SRC_TMP ${Caffe2_DB_COMMON_CPU_SRC} )

if(HAVE_CUDA)
  set(Caffe2_DB_GPU_SRC ${Caffe2_DB_COMMON_GPU_SRC} PARENT_SCOPE)
endif()

if(USE_LMDB)
  list(APPEND Caffe2_DB_SRC_TMP ${Caffe2_DB_LMDB_SRC})
endif()

if(USE_LEVELDB)
  list(APPEND Caffe2_DB_SRC_TMP ${Caffe2_DB_LEVELDB_SRC})
endif()

if(USE_ROCKSDB)
  list(APPEND Caffe2_DB_SRC_TMP ${Caffe2_DB_ROCKSDB_SRC})
endif()

if(USE_ZMQDB)
  list(APPEND Caffe2_DB_SRC_TMP ${Caffe2_DB_ZMQDB_SRC})
endif()

# ---[ Set final list of db sources
set(Caffe2_DB_SRC ${Caffe2_DB_SRC_TMP} PARENT_SCOPE)
