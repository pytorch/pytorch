set(extra_src)

# IF ANY SIMD FOUND
IF(C_AVX2_FOUND OR C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/generic/simd/convolve.cpp)
ENDIF(C_AVX2_FOUND OR C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)

# IF SSE4 FOUND
IF(C_SSE4_1_FOUND OR C_SSE4_2_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/generic/simd/convolve5x5_sse.cpp)
ENDIF(C_SSE4_1_FOUND OR C_SSE4_2_FOUND)

# IF AVX FOUND
IF(C_AVX_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX.cpp)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/generic/simd/convolve5x5_avx.cpp)
ENDIF(C_AVX_FOUND)

IF(C_AVX2_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX2.cpp)
ENDIF(C_AVX2_FOUND)

SET(hdr
  THGeneral.h THHalf.h THAllocator.h THSize.h THStorage.h THTensor.h THTensorApply.h THBlas.h THMath.h
  THLapack.h THLogAdd.h THRandom.h THVector.h )

set(ATen_TH_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/THGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THHalf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THSize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THStorage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THBlas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THLapack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THLogAdd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THRandom.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THFile.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THDiskFile.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THMemoryFile.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THVector.cpp
  ${extra_src}
  )
# Remember that PARENT_SCOPE variables are not in the current scope
set(ATen_TH_SRCS ${ATen_TH_SRCS} PARENT_SCOPE)
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} ${ATen_TH_SRCS} PARENT_SCOPE)
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
  THHalf.h
  THTensor.hpp
  THStorage.hpp
  THGenerator.hpp
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH")

INSTALL(FILES
  vector/AVX.h
  vector/AVX2.h
  ../ATen/native/cpu/avx_mathfun.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH/vector")

INSTALL(FILES
  generic/THBlas.cpp
  generic/THBlas.h
  generic/THLapack.cpp
  generic/THLapack.h
  generic/THStorage.cpp
  generic/THStorage.h
  generic/THStorageCopy.cpp
  generic/THStorageCopy.h
  generic/THTensor.cpp
  generic/THTensor.h
  generic/THTensorConv.cpp
  generic/THTensorConv.h
  generic/THTensorCopy.cpp
  generic/THTensorCopy.h
  generic/THTensorLapack.cpp
  generic/THTensorLapack.h
  generic/THTensorMath.cpp
  generic/THTensorMath.h
  generic/THTensorRandom.cpp
  generic/THTensorRandom.h
  generic/THVectorDispatch.cpp
  generic/THVector.h
  # See Note [TH abstraction violation]
  generic/THStorage.hpp
  generic/THTensor.hpp
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH/generic")
