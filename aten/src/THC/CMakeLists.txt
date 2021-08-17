set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

configure_file(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THCCachingHostAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THCReduceApplyUtils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSleep.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorage.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorCopy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMath.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathMagma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathPairwise.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathReduce.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorScatterGather.cu
  PARENT_SCOPE)

install(FILES
          THC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
          THCGeneral.hpp
          THCSleep.h
          THCStorage.h
          THCStorageCopy.h
          THCTensor.h
          THCTensorCopy.h
          THCTensorCopy.hpp
          THCTensorMath.h
          THCApply.cuh
          THCReduceApplyUtils.cuh
          THCTensorMathReduce.cuh
          THCAsmUtils.cuh
          THCAtomics.cuh
          THCScanUtils.cuh
          THCAllocator.h
          THCCachingHostAllocator.h
          THCDeviceUtils.cuh
          THCDeviceTensor.cuh
          THCDeviceTensor-inl.cuh
          THCDeviceTensorUtils.cuh
          THCDeviceTensorUtils-inl.cuh
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
          THCIntegerDivider.cuh
          THCNumerics.cuh
          THCTensorInfo.cuh
          THCTensorTypeUtils.cuh
          THCTensorMathMagma.h
          THCThrustAllocator.cuh
          # See Note [TH abstraction violation]
          THCTensor.hpp
          THCStorage.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

install(FILES
          generic/THCStorage.cpp
          generic/THCStorage.cu
          generic/THCStorage.h
          generic/THCTensor.cpp
          generic/THCTensor.cu
          generic/THCTensor.h
          generic/THCTensor.hpp
          generic/THCStorageCopy.cpp
          generic/THCStorageCopy.cu
          generic/THCStorageCopy.h
          generic/THCTensorCopy.cu
          generic/THCTensorCopy.h
          generic/THCTensorMath.h
          generic/THCTensorMath.cu
          generic/THCTensorMathMagma.h
          generic/THCTensorMathMagma.cpp
          generic/THCTensorMathPairwise.h
          generic/THCTensorMathPairwise.cu
          generic/THCTensorScatterGather.h
          generic/THCTensorScatterGather.cu
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
