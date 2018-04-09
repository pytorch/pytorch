cmake_minimum_required(VERSION 3.0)
project(TH)

if (NOT SUBPROJECT)
  include(../../cmake/Common.cmake)
  include(LoadOpenMP)
  include(LoadSSE)
  include(LoadLAPACK)
  include(LoadARM)
  include(LoadBLAS)
  add_subdirectory(../avx_mathfun avx_mathfun)
  add_subdirectory(../THRTS THRTS)
endif()

set(SRC
  THHalf.c
  THSize.c
  THStorage.c
  THTensor.cpp
  THBlas.c
  THLapack.cpp
  THLogAdd.c
  THRandom.cpp
  THFile.c
  THDiskFile.c
  THMemoryFile.c
  THVector.cpp
)

# IF ANY SIMD FOUND
if(C_AVX2_FOUND OR C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)
  list(APPEND SRC generic/simd/convolve.c)
endif()

# IF SSE4 FOUND
if(C_SSE4_1_FOUND OR C_SSE4_2_FOUND)
  list(APPEND SRC generic/simd/convolve5x5_sse.c)

  if(MSVC)
    set_source_files_properties(generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/fp:fast")
  else()
    set_source_files_properties(generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math")
  endif(MSVC)
endif()

# IF AVX FOUND
if(C_AVX_FOUND)
  list(APPEND SRC vector/AVX.c)
  list(APPEND SRC generic/simd/convolve5x5_avx.c)

  if(MSVC)
    set_source_files_properties(generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/fp:fast ${C_AVX_FLAGS}")
    set_source_files_properties(vector/AVX.c PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/arch:AVX ${C_AVX_FLAGS}")
  else()
    set_source_files_properties(generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math ${C_AVX_FLAGS}")
    set_source_files_properties(vector/AVX.c PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX_FLAGS}")
  endif()
endif()

if(C_AVX2_FOUND)
  list(APPEND SRC vector/AVX2.cpp)

  if(MSVC)
    set_source_files_properties(vector/AVX2.cpp PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/arch:AVX2 ${C_AVX2_FLAGS}")
  else()
    set_source_files_properties(vector/AVX2.cpp PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX2_FLAGS}")
  endif()
endif()

if(NOT MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  set_source_files_properties(THAtomic.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
  set_source_files_properties(THAllocator.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
endif()

add_library(${PROJECT_NAME}_static STATIC ${SRC})
target_link_libraries(${PROJECT_NAME}_static PUBLIC avx_mathfun THRTS)
if(BLAS_FOUND)
  target_link_libraries(${PROJECT_NAME}_static PUBLIC ${BLAS_LIBRARIES})
endif()
if (USE_LAPACK)
  target_link_libraries(${PROJECT_NAME}_static PUBLIC ${LAPACK_LIBRARIES})
endif()
target_include_directories(${PROJECT_NAME}_static PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${ATEN_INSTALL_INCLUDE_SUBDIR}>
  )
caffe2_interface_library(${PROJECT_NAME}_static ${PROJECT_NAME})

install(FILES
  TH.h
  THMath.h
  THBlas.h
  THDiskFile.h
  THFile.h
  THFilePrivate.h
  THGenerateAllTypes.h
  THGenerateDoubleType.h
  THGenerateFloatType.h
  THGenerateHalfType.h
  THGenerateLongType.h
  THGenerateIntType.h
  THGenerateShortType.h
  THGenerateCharType.h
  THGenerateByteType.h
  THGenerateFloatTypes.h
  THGenerateIntTypes.h
  THLapack.h
  THLogAdd.h
  THMemoryFile.h
  THRandom.h
  THSize.h
  THStorage.h
  THTensor.h
  THTensorApply.h
  THTensorDimApply.h
  THTensorMacros.h
  THVector.h
  THHalf.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH")

install(FILES
  vector/AVX.h
  vector/AVX2.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH/vector")

install(FILES
  generic/THBlas.c
  generic/THBlas.h
  generic/THLapack.cpp
  generic/THLapack.h
  generic/THStorage.c
  generic/THStorage.h
  generic/THStorageCopy.c
  generic/THStorageCopy.h
  generic/THTensor.cpp
  generic/THTensor.h
  generic/THTensorConv.cpp
  generic/THTensorConv.h
  generic/THTensorCopy.c
  generic/THTensorCopy.h
  generic/THTensorLapack.c
  generic/THTensorLapack.h
  generic/THTensorMath.c
  generic/THTensorMath.h
  generic/THTensorRandom.cpp
  generic/THTensorRandom.h
  generic/THVectorDispatch.cpp
  generic/THVector.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH/generic")
