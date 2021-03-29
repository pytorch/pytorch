message("CRASHHANDLER")
if(LINUX)
    set(TORCH_API_TEST_DIR "${TORCH_ROOT}/test/cpp/api")
    set(TORCH_API_TEST_SOURCES
    ${TORCH_ROOT}/test/cpp/crash_handler/segfault.cpp
    #   ${TORCH_API_TEST_DIR}/segfault.cpp
    )


    add_executable(test_crash_handler ${TORCH_API_TEST_SOURCES})
    target_include_directories(test_crash_handler PRIVATE)
    target_link_libraries(test_crash_handler PRIVATE torch_cpu)
    target_link_libraries(test_crash_handler PRIVATE torch)

    if(INSTALL_TEST)
        install(TARGETS test_crash_handler DESTINATION bin)
    endif()
endif()



