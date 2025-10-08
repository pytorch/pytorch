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
    add_custom_command(TARGET ${LIBRARY_TEST}
                POST_BUILD
                COMMAND ${CMAKE_CTEST_COMMAND} -C Release --output-on-failure --verbose)
endif()
