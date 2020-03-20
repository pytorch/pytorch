set(extra_src)

# IF AVX FOUND
IF(C_AVX_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX.cpp)
ENDIF(C_AVX_FOUND)

IF(C_AVX2_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX2.cpp)
ENDIF(C_AVX2_FOUND)

SET(hdr
  THGeneral.h THHalf.h THAllocator.h THStorage.h THStorageFunctions.h THTensor.h THTensorApply.h THBlas.h
  THLapack.h THVector.h )

set(ATen_TH_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/THGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THStorageFunctions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensorRandom.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensorFill.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensorMath.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensorMoreMath.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensorEvenMoreMath.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensorLapack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THBlas.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THLapack.cpp
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
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
PARENT_SCOPE)

set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
PARENT_SCOPE)

CONFIGURE_FILE(THGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h")


INSTALL(FILES
  TH.h
  THAllocator.h
  THBlas.h
  THDiskFile.h
  THFile.h
  THFilePrivate.h
  ${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h
  THGenerateAllTypes.h
  THGenerateBFloat16Type.h
  THGenerateBoolType.h
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
  THGenerateQUInt8Type.h
  THGenerateQInt8Type.h
  THGenerateQInt32Type.h
  THGenerateQTypes.h
  THLapack.h
  THMemoryFile.h
  THStorage.h
  THStorageFunctions.h
  THTensor.h
  THTensorApply.h
  THTensorDimApply.h
  THVector.h
  THHalf.h
  THTensor.hpp
  THStorageFunctions.hpp
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
  generic/THTensor.hpp
  generic/THTensorFill.cpp
  generic/THTensorFill.h
  generic/THTensorLapack.cpp
  generic/THTensorLapack.h
  generic/THTensorMath.cpp
  generic/THTensorMath.h
  generic/THTensorRandom.cpp
  generic/THTensorRandom.h
  generic/THVectorDispatch.cpp
  generic/THVector.h
  # See Note [TH abstraction violation]
  generic/THTensorFastGetSet.hpp
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH/generic")
