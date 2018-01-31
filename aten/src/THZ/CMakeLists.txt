set(extra_src)

# IF AVX FOUND
IF(C_AVX_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX.c)
ENDIF(C_AVX_FOUND)

IF(C_AVX2_FOUND)
  LIST(APPEND extra_src ${CMAKE_CURRENT_SOURCE_DIR}/vector/AVX2.c)
ENDIF(C_AVX2_FOUND)

SET(hdr
    THZGeneral.h THZStorage.h THZBlas.h THZLapack.h THZMath.h
    THZVector.h THZTensor.h)

set(ATen_CPU_SRCS ${ATen_CPU_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THZStorage.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THZTensor.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THZBlas.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THZLapack.c
  ${CMAKE_CURRENT_SOURCE_DIR}/THZVector.c
  ${extra_src}
PARENT_SCOPE)
######################################################


set(ATen_CPU_INCLUDE ${ATen_CPU_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
PARENT_SCOPE)


INSTALL(FILES
  THZ.h
  THZMath.h
  THZGenerateAllTypes.h
  THZGenerateZDoubleType.h
  THZGenerateZFloatType.h
  THZGeneral.h
  THZStorage.h
  THZTensor.h
  THZBlas.h
  THZLapack.h
  THZVector.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THZ")

INSTALL(FILES
  vector/AVX.h
  vector/AVX2.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THZ/vector")

INSTALL(FILES
  generic/THZBlas.c
  generic/THZBlas.h
  generic/THZLapack.c
  generic/THZLapack.h
  generic/THZStorage.c
  generic/THZStorage.h
  generic/THZStorageCopy.c
  generic/THZStorageCopy.h
  generic/THZTensor.c
  generic/THZTensor.h
  generic/THZTensorConv.c
  generic/THZTensorConv.h
  generic/THZTensorCopy.c
  generic/THZTensorCopy.h
  generic/THZTensorLapack.c
  generic/THZTensorLapack.h
  generic/THZTensorMath.c
  generic/THZTensorMath.h
  generic/THZTensorRandom.c
  generic/THZTensorRandom.h
  generic/THZVectorDispatch.c
  generic/THZVector.h
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THZ/generic")
