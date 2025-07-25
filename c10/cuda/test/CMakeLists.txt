# ---[ Test binaries.

set(C10_CUDA_ALL_TEST_FILES
    impl/CUDAAssertionsTest_1_var_test.cu
    impl/CUDAAssertionsTest_catches_stream.cu
    impl/CUDAAssertionsTest_catches_thread_and_block_and_device.cu
    impl/CUDAAssertionsTest_from_2_processes.cu
    impl/CUDAAssertionsTest_multiple_writes_from_blocks_and_threads.cu
    impl/CUDAAssertionsTest_multiple_writes_from_multiple_blocks.cu
    impl/CUDAAssertionsTest_multiple_writes_from_same_block.cu
    impl/CUDATest.cpp)
if(BUILD_TEST)
  foreach(test_src ${C10_CUDA_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "c10_cuda_${test_file_name}")
    if(WIN32 AND test_src MATCHES "^.*\.hip$")
      set_source_files_properties(${test_src}
                                  PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
      hip_add_executable(${test_name} "${test_src}")
      list(JOIN PYTORCH_ROCM_ARCH " " ROCM_PROPERTY_ARCH_LIST)
      set_target_properties(
        ${test_name} PROPERTIES LINKER_LANGUAGE CXX HIP_ARCHITECTURES
                                                    ${ROCM_PROPERTY_ARCH_LIST})
    else()
      add_executable(${test_name} "${test_src}")
    endif()
    if(test_src MATCHES "^.*\.hip$")
      set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CXX)
    endif()
    target_link_libraries(${test_name} ${C10_CUDA_LIB} ${C10_LIB} gmock gtest
                          gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      set_target_properties(
        ${test_name}
        PROPERTIES INSTALL_RPATH
                   "${CMAKE_INSTALL_RPATH}:${_rpath_portable_origin}/../lib")
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()
