if(USE_CUDA)
    set(Caffe2_CUDA_RTC_GPU_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/elemenntwise_rtc_gpu.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/pool_op_rtc_gpu.cc"
    )

    set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${Caffe2_CUDA_RTC_GPU_SRC})
    set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
else()
    message(STATUS "CUDA RTC operators skipped due to no CUDA support")
endif()
