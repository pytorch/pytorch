cmake_minimum_required(VERSION 3.1)

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_library(OpDependencyPass MODULE
  op_dependency.cpp
)

set_target_properties(OpDependencyPass PROPERTIES
    COMPILE_FLAGS "-std=c++11 -fno-rtti -O3")

if(APPLE)
  set_target_properties(OpDependencyPass PROPERTIES
      LINK_FLAGS "-undefined dynamic_lookup"
  )
endif(APPLE)
