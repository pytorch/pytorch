set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  ${TH_INCLUDE_PATH}/TH
  "${CMAKE_CURRENT_BINARY_DIR}"
PARENT_SCOPE)


set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSTensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSTensor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSparse.cu
  PARENT_SCOPE
)

INSTALL(FILES
          THCS.h
          THCSTensor.h
          THCSGenerateAllTypes.h
          THCSGenerateByteType.h
          THCSGenerateCharType.h
          THCSGenerateShortType.h
          THCSGenerateIntType.h
          THCSGenerateLongType.h
          THCSGenerateHalfType.h
          THCSGenerateFloatType.h
          THCSGenerateFloatTypes.h
          THCSGenerateDoubleType.h
          THCSparse.h
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCS")

INSTALL(FILES
          generic/THCSTensor.cpp
          generic/THCSTensor.cu
          generic/THCSTensor.h
          generic/THCSTensorMath.h
          generic/THCSTensorMath.cu
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THCS/generic")
