set(extra_src)

# IF ANY SIMD FOUND
IF(C_AVX2_FOUND OR C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/generic/simd/convolve.c)
ENDIF(C_AVX2_FOUND OR C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)

# IF SSE4 FOUND
IF(C_SSE4_1_FOUND OR C_SSE4_2_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/generic/simd/convolve5x5_sse.c)
ENDIF(C_SSE4_1_FOUND OR C_SSE4_2_FOUND)

# IF AVX FOUND
IF(C_AVX_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX.c)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/generic/simd/convolve5x5_avx.c)
ENDIF(C_AVX_FOUND)

IF(C_AVX2_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX2.cpp)
ENDIF(C_AVX2_FOUND)

SET(hdr
  THGeneral.h THHalf.h THAllocator.h THSize.h THStorage.h THTensor.h THTensorApply.h THBlas.h THMath.h
  THLapack.h THLogAdd.h THRandom.h THVector.h THAtomic.h )

set(ATen_CPU_SRCS ${ATen_CPU_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THGeneral.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THHalf.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THAllocator.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THSize.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THStorage.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THBlas.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THLapack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THLogAdd.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THRandom.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THFile.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THDiskFile.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THMemoryFile.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THAtomic.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THVector.cpp
  ${extra_src}
PARENT_SCOPE)
######################################################


set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
PARENT_SCOPE)

CONFIGURE_FILE(THGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h")


INSTALL(FILES
  TH.h
  THAllocator.h
  THMath.h
  THBlas.h
  THDiskFile.h
  THFile.h
  THFilePrivate.h
  ${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h
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
  THAtomic.h
  THHalf.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH")

INSTALL(FILES
  vector/AVX.h
  vector/AVX2.h
  ../ATen/native/cpu/avx_mathfun.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH/vector")

INSTALL(FILES
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
