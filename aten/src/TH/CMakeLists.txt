set(Aten_TH_AVX_extra_src)

set(hdr
  THGeneral.h THHalf.h THStorage.h THStorageFunctions.h THTensor.h)

set(ATen_TH_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/THGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THStorageFunctions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THTensor.cpp
  )
# Remember that PARENT_SCOPE variables are not in the current scope
set(ATen_TH_SRCS ${ATen_TH_SRCS} PARENT_SCOPE)
set(ATen_CPU_SRCS ${ATen_CPU_SRCS} ${ATen_TH_SRCS} PARENT_SCOPE)

# Aten_TH_AVX_extra_src is used in aten/src/ATen/CMakeLists.txt
# when built with BUILD_LITE_INTERPRETER=1
set(Aten_TH_AVX_extra_src ${Aten_TH_AVX_extra_src} PARENT_SCOPE)
######################################################


set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
PARENT_SCOPE)

set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
PARENT_SCOPE)

configure_file(THGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h")


install(FILES
  TH.h
  ${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h
  THGenerateAllTypes.h
  THGenerateBFloat16Type.h
  THGenerateBoolType.h
  THGenerateDoubleType.h
  THGenerateFloatType.h
  THGenerateHalfType.h
  THGenerateComplexFloatType.h
  THGenerateComplexDoubleType.h
  THGenerateLongType.h
  THGenerateIntType.h
  THGenerateShortType.h
  THGenerateCharType.h
  THGenerateByteType.h
  THGenerateFloatTypes.h
  THGenerateComplexTypes.h
  THGenerateIntTypes.h
  THGenerateQUInt8Type.h
  THGenerateQUInt4x2Type.h
  THGenerateQInt8Type.h
  THGenerateQInt32Type.h
  THGenerateQTypes.h
  THStorage.h
  THStorageFunctions.h
  THTensor.h
  THHalf.h
  THTensor.hpp
  THStorageFunctions.hpp
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH")

install(FILES
  generic/THStorage.cpp
  generic/THStorage.h
  generic/THStorageCopy.cpp
  generic/THStorageCopy.h
  generic/THTensor.cpp
  generic/THTensor.h
  generic/THTensor.hpp
  # See Note [TH abstraction violation]
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/TH/generic")
