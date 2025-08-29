find_package(GTest QUIET)

if(GTest_FOUND)
    message(STATUS "Found System GTest: ${GTest_INCLUDE_DIRS}")
else()
    set(GTest_REL_PATH "../../../../../../../third_party/googletest")
    get_filename_component(GTest_DIR "${CMAKE_CURRENT_LIST_DIR}/${GTest_REL_PATH}" ABSOLUTE)

    if(EXISTS "${GTest_DIR}/CMakeLists.txt")
        message(STATUS "Using GTest at ${GTest_DIR}")
        set(BUILD_GMOCK OFF CACHE BOOL "Disable GMock build")
        set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest install")
        add_subdirectory(${GTest_DIR} "${CMAKE_BINARY_DIR}/googletest_build")

        set(GTest_INCLUDE_DIRS "${GTest_DIR}/include")
        set(GTest_LIBRARIES GTest gtest_main)
        set(GTest_FOUND TRUE)
    else()
        message(FATAL_ERROR "GTest Not Found")
    endif()
endif()

if(GTest_FOUND)
    add_library(GTest::GTest SHARED IMPORTED)
    set_target_properties(GTest::GTest PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GTest_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${GTest_LIBRARIES}"
    )
endif()
