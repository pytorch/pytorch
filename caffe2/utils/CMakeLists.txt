project(c10_utils CXX C)

set(OLD_LIB_SOURCES_CPU
        utils/proto_utils.cc
        utils/murmur_hash3.cc
        utils/smart_tensor_printer.cc
        utils/signal_handler.cc
        utils/string_utils.cc
        utils/threadpool/ThreadPool.cc
        utils/cpuid.cc
        utils/bench_utils.cc
        utils/math_cpu.cc
        utils/math_utils.cc
        )

# ---[ threadpool/pthreadpool* is a local modification of the NNPACK
# pthreadpool with a very similar interface. Neither NNPACK, nor this
# thread pool supports Windows.
if (NOT MSVC)
  list(APPEND OLD_LIB_SOURCES_CPU
          utils/threadpool/pthreadpool.cc
          utils/threadpool/pthreadpool_impl.cc
          )
endif()

set(OLD_LIB_SOURCES_GPU
        utils/math_gpu.cu
        )

set(OLD_TEST_SOURCES_CPU
        utils/fixed_divisor_test.cc
        utils/math_test.cc
        utils/fatal_signal_asan_no_sig_test.cc
        utils/simple_queue_test.cc
        utils/proto_utils_test.cc
        utils/cpuid_test.cc
        utils/smart_tensor_printer_test.cc
        utils/cast_test.cc
        )

set(OLD_TEST_SOURCES_GPU
        utils/math_gpu_test.cc
        )

# TODO Remove the OLD_xxx variables above and add them to the variables for the local library target below instead

set(LIB_SOURCES_CPU
        dummy.cpp
        )

set(LIB_SOURCES_GPU
        dummy.cpp
        )

set(TEST_SOURCES_CPU
        dummy.cpp
        )

set(LIB_SOURCES_GPU
        dummy.cpp
        )


add_library(${PROJECT_NAME}_cpu OBJECT ${LIB_SOURCES_CPU})
target_enable_style_warnings("${PROJECT_NAME}_cpu")

add_library(${PROJECT_NAME}_gpu OBJECT ${LIB_SOURCES_GPU})
target_enable_style_warnings("${PROJECT_NAME}_gpu")

if(BUILD_TEST)
  add_executable(${PROJECT_NAME}_cpu_test ${TEST_SOURCES_CPU} $<TARGET_OBJECTS:${PROJECT_NAME}>)
  add_test(NAME ${PROJECT_NAME}_cpu_test COMMAND $<TARGET_FILE:${PROJECT_NAME}_cpu_test>)
  #target_enable_style_warnings(${PROJECT_NAME}_cpu_test)
  target_link_libraries(${PROJECT_NAME}_cpu_test gtest_main)
  if(INSTALL_TEST)
    install(TARGETS ${PROJECT_NAME}_cpu_test DESTINATION test)
  endif()

  add_executable(${PROJECT_NAME}_gpu_test ${TEST_SOURCES_GPU} $<TARGET_OBJECTS:${PROJECT_NAME}>)
  add_test(NAME ${PROJECT_NAME}_gpu_test COMMAND $<TARGET_FILE:${PROJECT_NAME}_gpu_test>)
  #target_enable_style_warnings(${PROJECT_NAME}_gpu_test)
  target_link_libraries(${PROJECT_NAME}_gpu_test gtest_main)
  if(INSTALL_TEST)
    install(TARGETS ${PROJECT_NAME}_gpu_test DESTINATION test)
  endif()
endif()


# TODO Once all source files are defined inside the local c10_utils_xxx targets,
# it should be the job of the parent CMakeLists.txt to decide what to do with the target (i.e. link it to caffe2)
# instead of us locally adding it to Caffe2_xxx variables.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${OLD_LIB_SOURCES_CPU} $<TARGET_OBJECTS:c10_utils_cpu> PARENT_SCOPE)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${OLD_LIB_SOURCES_GPU} $<TARGET_OBJECTS:c10_utils_gpu> PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${OLD_TEST_SOURCES_CPU} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${OLD_TEST_SOURCES_GPU} PARENT_SCOPE)
