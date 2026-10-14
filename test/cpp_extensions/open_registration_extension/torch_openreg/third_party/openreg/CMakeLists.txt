cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(TORCH_OPENREG CXX C)

option(USE_TEST "Build and run unit tests" ON)

set(LIBRARY_NAME openreg)
set(LIBRARY_TEST ortests)

file(GLOB_RECURSE SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp"
)

add_library(${LIBRARY_NAME} SHARED ${SOURCE_FILES})

target_include_directories(${LIBRARY_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(UNIX AND NOT APPLE)
    # shm_open/shm_unlink (used for IPC event support) live in librt on
    # glibc < 2.34; harmless to link on newer glibc where it's a no-op stub.
    target_link_libraries(${LIBRARY_NAME} PUBLIC rt)
endif()

install(TARGETS ${LIBRARY_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(USE_TEST)
    enable_testing()

    include(${CMAKE_CURRENT_LIST_DIR}/cmake/GTestTargets.cmake)

    file(GLOB_RECURSE TEST_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    )

    add_executable(${LIBRARY_TEST} ${TEST_FILES})
    target_link_libraries(${LIBRARY_TEST}
        PRIVATE
        ${LIBRARY_NAME}
        gtest
        gtest_main
    )

    add_test(NAME alltests COMMAND ${LIBRARY_TEST})
endif()
