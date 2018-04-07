cmake_minimum_required(VERSION 3.0)
project(THCRTS)

if (NOT SUBPROJECT)
  include(../../cmake/Common.cmake)
  include(LoadCUDA)
  include(LoadMAGMA)
  add_subdirectory(../THRTS THRTS)
endif()

set(SRC
  THCRTSGeneral.cpp
  THCStream.cpp
  THCCachingAllocator.cpp
  THCCachingHostAllocator.cpp
  THCAllocator.cpp
  THCThreadLocal.cpp
)

cuda_add_library(${PROJECT_NAME}_static STATIC ${SRC})
target_link_libraries(${PROJECT_NAME}_static PUBLIC THRTS ${CUDA_LIBRARIES})
if(USE_MAGMA)
  target_link_libraries(${PROJECT_NAME}_static PUBLIC ${MAGMA_LIBRARIES})
endif()
target_include_directories(${PROJECT_NAME}_static PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${ATEN_INSTALL_INCLUDE_SUBDIR}>
  )
caffe2_interface_library(${PROJECT_NAME}_static ${PROJECT_NAME})

install(FILES
  THCRTSGeneral.h
  THCCachingAllocator.h
  THCCachingHostAllocator.h
  THCAllocator.h
  THCStream.h
  THCThreadLocal.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")
