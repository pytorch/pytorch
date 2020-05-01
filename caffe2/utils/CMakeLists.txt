if(INTERN_BUILD_MOBILE AND NOT BUILD_CAFFE2_MOBILE)
  list(APPEND Caffe2_CPU_SRCS
    utils/string_utils.cc
    utils/threadpool/pthreadpool-cpp.cc
    utils/threadpool/ThreadPool.cc
  )
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
  return()
endif()

list(APPEND Caffe2_CPU_SRCS
  utils/bench_utils.cc
  utils/cpuid.cc
  utils/math/broadcast.cc
  utils/math/elementwise.cc
  utils/math/reduce.cc
  utils/math/transpose.cc
  utils/math/utils.cc
  utils/math_cpu.cc
  utils/murmur_hash3.cc
  utils/proto_convert.cc
  utils/proto_utils.cc
  utils/proto_wrap.cc
  utils/threadpool/pthreadpool-cpp.cc
  utils/threadpool/ThreadPool.cc
  utils/signal_handler.cc
  utils/smart_tensor_printer.cc
  utils/string_utils.cc)

if(USE_INTERNAL_PTHREADPOOL_IMPL)
  list(APPEND Caffe2_CPU_SRCS
    utils/threadpool/pthreadpool.cc
    utils/threadpool/pthreadpool_impl.cc)
endif()

set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS}
        utils/math/broadcast.cu
        utils/math/elementwise.cu
        utils/math/reduce.cu
        utils/math/transpose.cu
        utils/math_gpu.cu
        )

set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS}
        utils/math/hip/broadcast.hip
        utils/math/hip/elementwise.hip
        utils/math/hip/reduce.hip
        utils/math/hip/transpose.hip
        utils/hip/math_gpu.hip
        )

set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS}
        utils/fixed_divisor_test.cc
        utils/math_test.cc
        utils/fatal_signal_asan_no_sig_test.cc
        utils/simple_queue_test.cc
        utils/proto_utils_test.cc
        utils/cpuid_test.cc
        utils/smart_tensor_printer_test.cc
        utils/cast_test.cc
        )

set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS}
        utils/math_gpu_test.cc
        )

set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS}
        utils/hip/math_gpu_test.cc
        utils/hip/math_blas_gpu_test.cc
        )

# TODO Once all source files are defined inside the local c10_utils_xxx targets,
# it should be the job of the parent CMakeLists.txt to decide what to do with the target (i.e. link it to caffe2)
# instead of us locally adding it to Caffe2_xxx variables.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} PARENT_SCOPE)
