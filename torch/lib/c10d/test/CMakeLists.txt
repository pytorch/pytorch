if(USE_C10D_NCCL)
  if (MSVC)
    set(LIBSHM_SUBDIR libshm_windows)
  else()
    set(LIBSHM_SUBDIR libshm)
  endif()

  set(LIBSHM_SRCDIR ${TORCH_SRC_DIR}/lib/${LIBSHM_SUBDIR})

  add_library(nccl_utils SHARED ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
  target_link_libraries(nccl_utils __caffe2_nccl shm)
  target_include_directories(nccl_utils PUBLIC
      ${CMAKE_BINARY_DIR}/caffe2/aten/src ${CMAKE_BINARY_DIR}/nccl/include
      ${LIBSHM_SRCDIR})
endif()

if(USE_CUDA)
  cuda_add_library(c10d_cuda_test CUDATest.cu)
  target_link_libraries(c10d_cuda_test c10d)
  add_dependencies(c10d_cuda_test c10d)
endif()

function(c10d_add_test test_src)
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${test_name} pthread ${ARGN})
  target_compile_options(${test_name} PRIVATE -Wno-error)
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endfunction()

c10d_add_test(FileStoreTest.cpp c10d)
c10d_add_test(TCPStoreTest.cpp c10d)

if(USE_CUDA)
  if(USE_C10D_GLOO)
    c10d_add_test(ProcessGroupGlooTest.cpp c10d c10d_cuda_test)
    c10d_add_test(ProcessGroupGlooAsyncTest.cpp c10d c10d_cuda_test)
  endif()
  if(USE_C10D_NCCL)
    c10d_add_test(ProcessGroupNCCLTest.cpp c10d c10d_cuda_test)
    c10d_add_test(ProcessGroupNCCLErrorsTest.cpp nccl_utils c10d c10d_cuda_test
        gtest_main)
  endif()
else()
  if(USE_C10D_GLOO)
    c10d_add_test(ProcessGroupGlooTest.cpp c10d c10d)
  endif()
endif()

if(USE_C10D_MPI)
  add_definitions(-DMPIEXEC=${MPIEXEC})
  c10d_add_test(ProcessGroupMPITest.cpp c10d)
endif()
