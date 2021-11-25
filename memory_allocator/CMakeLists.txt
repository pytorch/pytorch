cmake_minimum_required(VERSION 3.19)
project(pytorch_memory_allocator)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(pytorch_memory_allocator main_static_resnet.cpp)
#add_executable(pytorch_memory_allocator main_static_resnet.cpp)
set_property(TARGET pytorch_memory_allocator PROPERTY CXX_STANDARD 14)

target_include_directories(pytorch_memory_allocator PRIVATE ${ATen_CPU_INCLUDE})
target_link_libraries(pytorch_memory_allocator PRIVATE torch)

# find jemalloc
find_package(PkgConfig REQUIRED)
pkg_check_modules(JEMALLOC REQUIRED jemalloc)
pkg_search_module(JEMALLOC REQUIRED jemalloc)
include_directories(${JEMALLOC_INCLUDE_DIRS})

# make all connections for jemallc
target_link_directories(pytorch_memory_allocator PRIVATE ${JEMALLOC_LIBRARY_DIRS})
target_link_libraries(pytorch_memory_allocator PRIVATE ${JEMALLOC_LIBRARIES})
