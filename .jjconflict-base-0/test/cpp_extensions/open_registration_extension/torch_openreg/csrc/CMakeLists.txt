set(LIBRARY_NAME torch_openreg)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_cpu_library openreg)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(kineto_LIBRARY)
    if(NOT TARGET kineto)
        add_library(kineto STATIC IMPORTED)
        set_target_properties(kineto PROPERTIES
            IMPORTED_LOCATION "${kineto_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES
                "${PYTORCH_INSTALL_DIR}/include/kineto;${PYTORCH_INSTALL_DIR}/include")
    endif()
    target_link_libraries(${LIBRARY_NAME} PRIVATE kineto)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_KINETO)
    message(STATUS "torch_openreg: Kineto found at ${kineto_LIBRARY}, USE_KINETO enabled")
else()
    message(STATUS "torch_openreg: Kineto not found, profiler compiled as no-op")
endif()

if(USE_TEST AND kineto_LIBRARY)
    set(PROFILER_TEST_TARGET openreg_profiler_tests)
    add_executable(${PROFILER_TEST_TARGET}
        ${PROJECT_SOURCE_DIR}/tests/cpp/test_openreg_activity_profiler.cpp)
    target_link_libraries(${PROFILER_TEST_TARGET}
        PRIVATE torch_openreg torch_cpu_library kineto openreg gtest gtest_main)
    target_compile_definitions(${PROFILER_TEST_TARGET} PRIVATE USE_KINETO KINETO_NAMESPACE=libkineto)
    add_test(NAME ${PROFILER_TEST_TARGET} COMMAND ${PROFILER_TEST_TARGET})
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)