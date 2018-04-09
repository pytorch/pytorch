cmake_minimum_required(VERSION 3.0)
project(THRTS)

if (NOT SUBPROJECT)
  include(../../cmake/Common.cmake)
  include(LoadOpenMP)
  include(LoadBLAS)
  include(LoadLAPACK)
endif()

configure_file(THGeneral.h.in THGeneral.h)

if(NOT MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  set_source_files_properties(THAtomic.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
  set_source_files_properties(THAllocator.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
endif()

set(SRC
  THGeneral.c
  THAtomic.c
  THAllocator.c
)

add_library(${PROJECT_NAME}_static STATIC ${SRC})
target_include_directories(${PROJECT_NAME}_static PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${ATEN_INSTALL_INCLUDE_SUBDIR}>
  )

if(NEED_LIBRT)
  target_link_libraries(${PROJECT_NAME}_static PUBLIC rt)
endif()
if(BLAS_FOUND)
  target_link_libraries(${PROJECT_NAME}_static PUBLIC ${BLAS_LIBRARIES})
endif()
if(LAPACK_FOUND)
  target_link_libraries(${PROJECT_NAME}_static PUBLIC ${LAPACK_LIBRARIES})
endif()

caffe2_interface_library(${PROJECT_NAME}_static ${PROJECT_NAME})

install(FILES
  THAllocator.h
  ${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h
  THAtomic.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH")
