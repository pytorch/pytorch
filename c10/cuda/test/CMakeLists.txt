# ---[ Test binaries.

set(C10_CUDA_ALL_TEST_FILES
    impl/CUDAAssertionsTest_1_var_test.cu
    impl/CUDAAssertionsTest_catches_stream.cu
    impl/CUDAAssertionsTest_catches_thread_and_block_and_device.cu
    impl/CUDAAssertionsTest_from_2_processes.cu
    impl/CUDAAssertionsTest_multiple_writes_from_blocks_and_threads.cu
    impl/CUDAAssertionsTest_multiple_writes_from_multiple_blocks.cu
    impl/CUDAAssertionsTest_multiple_writes_from_same_block.cu
    impl/CUDATest.cpp
)
if(BUILD_TEST)
  foreach(test_src ${C10_CUDA_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "c10_cuda_${test_file_name}")
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} c10_cuda gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()
