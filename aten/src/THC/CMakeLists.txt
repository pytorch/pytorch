set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

configure_file(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THCGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THCSleep.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorage.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathMagma.cpp
  PARENT_SCOPE)

install(FILES
          THC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
          THCGeneral.hpp
          THCSleep.h
          THCStorage.h
          THCStorageCopy.h
          THCTensor.h
          THCTensorMathReduce.cuh
          THCAtomics.cuh
          THCGenerateAllTypes.h
          THCGenerateBFloat16Type.h
          THCGenerateBoolType.h
          THCGenerateByteType.h
          THCGenerateCharType.h
          THCGenerateShortType.h
          THCGenerateIntType.h
          THCGenerateLongType.h
          THCGenerateHalfType.h
          THCGenerateFloatType.h
          THCGenerateFloatTypes.h
          THCGenerateDoubleType.h
          THCGenerateComplexFloatType.h
          THCGenerateComplexTypes.h
          THCGenerateComplexDoubleType.h
          THCNumerics.cuh
          # See Note [TH abstraction violation]
          THCTensor.hpp
          THCStorage.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

install(FILES
          generic/THCStorage.cpp
          generic/THCStorage.cu
          generic/THCStorage.h
          generic/THCTensor.cpp
          generic/THCTensor.h
          generic/THCStorageCopy.cpp
          generic/THCStorageCopy.cu
          generic/THCStorageCopy.h
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
