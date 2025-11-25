if(USE_CUDA)
  add_library(c10d_cuda_test CUDATest.cu)
  target_include_directories(c10d_cuda_test PRIVATE $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
  target_link_libraries(c10d_cuda_test torch_cuda)
  add_dependencies(c10d_cuda_test torch_cuda)
elseif(USE_ROCM)
  set_source_files_properties(hip/HIPTest.hip PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_library(c10d_hip_test hip/HIPTest.hip)
  target_include_directories(c10d_hip_test PRIVATE $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
  target_link_libraries(c10d_hip_test torch_hip)
  add_dependencies(c10d_hip_test torch_hip)
endif()

function(c10d_add_test test_src)
  set(prefix ARG)
  set(noValues)
  set(singleValues INSTALL_TEST)
  set(multiValues LINK_LIBRARIES)

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix} "${noValues}" "${singleValues}" "${multiValues}" ${ARGN})

  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_include_directories(${test_name} PRIVATE
      $<BUILD_INTERFACE:${TORCH_SRC_DIR}/test/cpp/c10d/hip>
      $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>
      $<TARGET_PROPERTY:fmt::fmt-header-only,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_link_libraries(${test_name} PRIVATE
    fmt::fmt-header-only
    ${ARG_LINK_LIBRARIES}
  )
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)

  if(ARG_INSTALL_TEST)
    set_target_properties(${test_name} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
    install(TARGETS ${test_name} DESTINATION bin)
  endif()
endfunction()

if (USE_ROCM)
  c10d_add_test(hip/BackoffTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST OFF)
  c10d_add_test(hip/FileStoreTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST ${INSTALL_TEST})
  c10d_add_test(hip/TCPStoreTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST ${INSTALL_TEST})
  if(NOT WIN32)
    c10d_add_test(hip/HashStoreTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST ${INSTALL_TEST})
  endif()

else()
  c10d_add_test(BackoffTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST OFF)
  c10d_add_test(FileStoreTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST ${INSTALL_TEST})
  c10d_add_test(TCPStoreTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST ${INSTALL_TEST})
  if(NOT WIN32)
    c10d_add_test(HashStoreTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST ${INSTALL_TEST})
  endif()
endif()

if(USE_CUDA)
  if(USE_GLOO AND USE_C10D_GLOO)
    c10d_add_test(ProcessGroupGlooTest.cpp LINK_LIBRARIES torch_cpu c10d_cuda_test gtest_main INSTALL_TEST ${INSTALL_TEST})
    c10d_add_test(ProcessGroupGlooAsyncTest.cpp LINK_LIBRARIES torch_cpu c10d_cuda_test gtest_main INSTALL_TEST ${INSTALL_TEST})
  endif()
  if(USE_NCCL AND USE_C10D_NCCL)
    # NCCL is a private dependency of libtorch, but the tests include some
    # private headers of libtorch, which in turn include NCCL. As a hacky
    # alternative to making NCCL a public dependency of libtorch, we make it
    # a private dependency of the tests as well.
    c10d_add_test(
      ProcessGroupNCCLTest.cpp
      LINK_LIBRARIES torch_cpu c10d_cuda_test gtest_main __caffe2_nccl INSTALL_TEST ${INSTALL_TEST})
    c10d_add_test(
      ProcessGroupNCCLErrorsTest.cpp
      LINK_LIBRARIES torch_cpu c10d_cuda_test gtest_main __caffe2_nccl INSTALL_TEST ${INSTALL_TEST})
    if(INSTALL_TEST)
      install(TARGETS c10d_cuda_test DESTINATION lib)
    endif()
  endif()
  if(USE_UCC AND USE_C10D_UCC)
    # UCC is a private dependency of libtorch, but the tests include some
    # private headers of libtorch, which in turn include UCC. As a hacky
    # alternative to making UCC a public dependency of libtorch, we make it
    # a private dependency of the tests as well.
    c10d_add_test(
      ProcessGroupUCCTest.cpp
      LINK_LIBRARIES torch_cpu c10d_cuda_test gtest_main __caffe2_ucc INSTALL_TEST ${INSTALL_TEST})
    if(INSTALL_TEST)
      install(TARGETS c10d_cuda_test DESTINATION lib)
    endif()
  endif()
elseif(USE_ROCM)
  if(USE_GLOO AND USE_C10D_GLOO)
    c10d_add_test(hip/ProcessGroupGlooTest.cpp LINK_LIBRARIES torch_cpu c10d_hip_test gtest_main INSTALL_TEST ${INSTALL_TEST})
    c10d_add_test(hip/ProcessGroupGlooAsyncTest.cpp LINK_LIBRARIES torch_cpu c10d_hip_test gtest_main INSTALL_TEST ${INSTALL_TEST})
  endif()
  if(USE_NCCL AND USE_C10D_NCCL)
    c10d_add_test(
      hip/ProcessGroupNCCLTest.cpp
      LINK_LIBRARIES torch_cpu c10d_hip_test gtest_main __caffe2_nccl INSTALL_TEST ${INSTALL_TEST})
    c10d_add_test(
      hip/ProcessGroupNCCLErrorsTest.cpp
      LINK_LIBRARIES torch_cpu c10d_hip_test gtest_main __caffe2_nccl INSTALL_TEST ${INSTALL_TEST})
    if(INSTALL_TEST)
      install(TARGETS c10d_hip_test DESTINATION lib)
    endif()
  endif()
  if(USE_UCC AND USE_C10D_UCC)
    c10d_add_test(
      hip/ProcessGroupUCCTest.cpp
      LINK_LIBRARIES torch_cpu c10d_hip_test gtest_main __caffe2_ucc INSTALL_TEST ${INSTALL_TEST})
    if(INSTALL_TEST)
      install(TARGETS c10d_hip_test DESTINATION lib)
    endif()
  endif()
else()
  if(USE_GLOO AND USE_C10D_GLOO)
    c10d_add_test(ProcessGroupGlooTest.cpp LINK_LIBRARIES torch_cpu gtest_main INSTALL_TEST OFF)
  endif()
endif()

if(USE_MPI AND USE_C10D_MPI)
  add_definitions(-DMPIEXEC=${MPIEXEC})
  # MPI is a private dependency of libtorch, but the tests include some
  # private headers of libtorch, which in turn include MPI. As a hacky
  # alternative to making MPI a public dependency of libtorch, we make it
  # a private dependency of the tests as well.
  c10d_add_test(ProcessGroupMPITest.cpp LINK_LIBRARIES torch_cpu MPI::MPI_CXX INSTALL_TEST ${INSTALL_TEST})
endif()

if(LINUX AND USE_GLOO AND USE_C10D_GLOO)
  add_executable(example_allreduce example/allreduce.cpp)
  target_include_directories(example_allreduce PRIVATE $<BUILD_INTERFACE:${TORCH_SRC_DIR}/csrc/distributed>)
  target_link_libraries(example_allreduce torch_cpu)
  if(USE_CUDA)
    target_link_libraries(example_allreduce torch_cuda)
  elseif(USE_ROCM)
    target_link_libraries(example_allreduce torch_hip)
  endif()
endif()
