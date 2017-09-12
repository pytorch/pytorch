cmake_minimum_required(VERSION 3.0)
project(ATen)

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

if(CMAKE_VERSION VERSION_LESS "3.1")
  set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

set(src
  BatchDataset.cc
  ConcatDataset.cc
  Dataset.cc
  MergeDataset.cc
  ResampleDataset.cc
  ShuffleDataset.cc
  TensorDataset.cc
  TransformDataset.cc
)

add_library(xtdata ${TH_LINK_STYLE} ${src})
target_link_libraries(xtdata ATen)

include_directories(.)
# add_executable(test-data test/basic.cc)
# target_link_libraries(test-data xtdata)
