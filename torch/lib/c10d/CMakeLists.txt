cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_MODULE_PATH})

# Relies on CMAKE_INSTALL_PREFIX to be set to ../tmp_install.
# It then finds $PREFIX/share/cmake/ATen/ATenConfig.cmake,
# which defines ATEN_INCLUDE_DIR and ATEN_LIBRARIES.
find_package(ATen REQUIRED)
if(NOT ATen_FOUND)
  message(FATAL "ATen not found")
endif()

add_library(store Utils.cpp Store.cpp FileStore.cpp TCPStore.cpp)
target_compile_options(store PUBLIC "-std=c++11")

add_library(process_group ProcessGroup.cpp)
target_compile_options(process_group PUBLIC "-std=c++11")
target_include_directories(process_group PUBLIC ${ATEN_INCLUDE_DIR})
target_link_libraries(process_group PUBLIC ${ATEN_LIBRARIES})

enable_testing()
add_subdirectory(test)
