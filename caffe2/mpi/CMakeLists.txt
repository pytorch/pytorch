if(USE_MPI AND MPI_CXX_FOUND)
    set(Caffe2_MPI_CPU_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/mpi_common.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/mpi_ops.cc"
        # TODO: properly compile this together with python.
        # "${CMAKE_CURRENT_SOURCE_DIR}/mpi_python.cc"
    )
    set(Caffe2_MPI_GPU_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/mpi_ops_gpu.cc"
    )
    set(Caffe2_MPI_CPU_TEST_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/mpi_test.cc"
    )
    set(Caffe2_MPI_GPU_TEST_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/mpi_gpu_test.cc"
    )

    # pass up to calling script
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_MPI_CPU_SRC})
    set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
    set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${Caffe2_MPI_GPU_SRC})
    set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
    set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${Caffe2_MPI_CPU_TEST_SRC})
    set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
    set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${Caffe2_MPI_GPU_TEST_SRC})
    set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
else()
    message(STATUS "MPI operators skipped due to no MPI support")
endif()
