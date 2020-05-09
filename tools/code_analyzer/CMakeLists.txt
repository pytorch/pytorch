cmake_minimum_required(VERSION 3.1)

project(code_analyzer)
set(CMAKE_CXX_STANDARD 14)

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# Main executable
add_executable(analyzer
  analyzer.cpp
  op_deps_pass.cpp
)

set_target_properties(analyzer PROPERTIES
    COMPILE_FLAGS "-fno-rtti -O3")

llvm_map_components_to_libnames(llvm_libs core irreader support)

target_link_libraries(analyzer ${llvm_libs})
