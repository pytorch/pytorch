set(LIBRARY_NAME torch_openreg)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_link_libraries(${LIBRARY_NAME} PRIVATE torch_cpu_library openreg)
target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_library(KINETO_LIBRARY kineto
    PATHS "${PYTORCH_INSTALL_DIR}/lib" "${PYTORCH_INSTALL_DIR}/lib64"
    NO_DEFAULT_PATH)
if(KINETO_LIBRARY)
    set(KINETO_INCLUDE_DIR ${PYTORCH_INSTALL_DIR}/include/kineto)
    target_include_directories(${LIBRARY_NAME} PRIVATE ${KINETO_INCLUDE_DIR})
    target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_KINETO)
    message(STATUS "torch_openreg: Kineto found at ${KINETO_LIBRARY}, USE_KINETO enabled")

    if(USE_TEST)
        set(PROFILER_TEST_TARGET openreg_profiler_tests)
        add_executable(${PROFILER_TEST_TARGET}
            ${PROJECT_SOURCE_DIR}/tests/cpp/test_openreg_activity_profiler.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/profiler/OpenRegActivityProfiler.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/profiler/OpenRegActivityProfilerSession.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/profiler/OpenRegTracer.cpp)
        target_link_libraries(${PROFILER_TEST_TARGET}
            PRIVATE torch_cpu_library gtest gtest_main)
        target_include_directories(${PROFILER_TEST_TARGET} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${KINETO_INCLUDE_DIR})
        target_compile_definitions(${PROFILER_TEST_TARGET} PRIVATE USE_KINETO KINETO_NAMESPACE=libkineto)
        add_test(NAME ${PROFILER_TEST_TARGET} COMMAND ${PROFILER_TEST_TARGET})
    endif()
else()
    message(STATUS "torch_openreg: Kineto not found, profiler/OpenRegActivityProfiler.cpp compiled as no-op")
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

endif()