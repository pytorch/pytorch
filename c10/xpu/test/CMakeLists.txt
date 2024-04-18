# ---[ Test binaries.

set(C10_XPU_ALL_TEST_FILES
    impl/XPUCachingAllocatorTest.cpp
    impl/XPUDeviceTest.cpp
    impl/XPUGuardTest.cpp
    impl/XPUStreamTest.cpp
)
if(BUILD_TEST)
  foreach(test_src ${C10_XPU_ALL_TEST_FILES})
    get_filename_component(test_file_name ${test_src} NAME_WE)
    set(test_name "c10_xpu_${test_file_name}")
    add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} c10_xpu gtest_main)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    if(INSTALL_TEST)
      install(TARGETS ${test_name} DESTINATION test)
    endif()
  endforeach()
endif()
