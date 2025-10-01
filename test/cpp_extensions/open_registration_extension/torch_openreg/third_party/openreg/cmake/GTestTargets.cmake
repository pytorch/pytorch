set(GTest_REL_PATH "../../../../../../../third_party/googletest")
get_filename_component(GTest_DIR "${CMAKE_CURRENT_LIST_DIR}/${GTest_REL_PATH}" ABSOLUTE)

if(EXISTS "${GTest_DIR}/CMakeLists.txt")
    message(STATUS "Found GTest: ${GTest_DIR}")

    set(BUILD_GMOCK OFF CACHE BOOL "Disable GMock build")
    set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest install")
    add_subdirectory(${GTest_DIR} "${CMAKE_BINARY_DIR}/gtest")
else()
    message(FATAL_ERROR "GTest Not Found")
endif()
