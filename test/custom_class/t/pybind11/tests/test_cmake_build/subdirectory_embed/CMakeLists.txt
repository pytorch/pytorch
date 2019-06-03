cmake_minimum_required(VERSION 3.0)
project(test_subdirectory_embed CXX)

set(PYBIND11_INSTALL ON CACHE BOOL "")
set(PYBIND11_EXPORT_NAME test_export)

add_subdirectory(${PYBIND11_PROJECT_DIR} pybind11)

# Test basic target functionality
add_executable(test_cmake_build ../embed.cpp)
target_link_libraries(test_cmake_build PRIVATE pybind11::embed)

add_custom_target(check $<TARGET_FILE:test_cmake_build> ${PROJECT_SOURCE_DIR}/../test.py)

# Test custom export group -- PYBIND11_EXPORT_NAME
add_library(test_embed_lib ../embed.cpp)
target_link_libraries(test_embed_lib PRIVATE pybind11::embed)

install(TARGETS test_embed_lib
        EXPORT  test_export
        ARCHIVE DESTINATION bin
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib)
install(EXPORT      test_export
        DESTINATION lib/cmake/test_export/test_export-Targets.cmake)
