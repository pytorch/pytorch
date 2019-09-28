set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

CONFIGURE_FILE(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

set(extra_src)
# loop over all types
foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THC_FILE TensorSort TensorMathCompareT TensorMathPointwise TensorMathCompare TensorMathReduce TensorMasked)
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
         FILE(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu"
           "#include <THC/THC${THC_FILE}.cuh>\n#include <THC/THCTensor.hpp>\n\n#include <THC/generic/THC${THC_FILE}.cu>\n#include <THC/THCGenerate${THC_TYPE}Type.h>\n")
      endif()
      LIST(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
   endforeach()
endforeach()

foreach(THC_FILE TensorMathCompareT TensorMathCompare TensorMathReduce TensorMasked TensorMathPointwise)
   if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}Bool.cu")
      FILE(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}Bool.cu"
        "#include <THC/THC${THC_FILE}.cuh>\n#include <THC/THCTensor.hpp>\n\n#include <THC/generic/THC${THC_FILE}.cu>\n#include <THC/THCGenerateBoolType.h>\n")
   endif()
   LIST(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}Bool.cu")
endforeach()

set(ATen_CUDA_SRCS ${ATen_CUDA_SRCS}
  ${CMAKE_CURRENT_SOURCE_DIR}/THCCachingHostAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCGeneral.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/THCReduceApplyUtils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCBlas.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSleep.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorage.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCStorageCopy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorCopy.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMath.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathBlas.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathMagma.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathPairwise.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathReduce.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathScan.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorIndex.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorRandom.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorScatterGather.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorTopK.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorSort.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSortUtils.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMode.cu
  ${extra_src}
  PARENT_SCOPE)

INSTALL(FILES
          THC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
          THCGeneral.hpp
          THCBlas.h
          THCSleep.h
          THCStorage.h
          THCStorageCopy.h
          THCTensor.h
          THCTensorCopy.h
          THCTensorCopy.hpp
          THCTensorRandom.h
          THCTensorMath.h
          THCApply.cuh
          THCReduce.cuh
          THCReduceAll.cuh
          THCReduceApplyUtils.cuh
          THCTensorMathReduce.cuh
          THCAsmUtils.cuh
          THCAtomics.cuh
          THCScanUtils.cuh
          THCSortUtils.cuh
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
          THCIntegerDivider.cuh
          THCNumerics.cuh
          THCTensorSort.cuh
          THCTensorInfo.cuh
          THCTensorMathPointwise.cuh
          THCTensorTypeUtils.cuh
          THCTensorRandom.cuh
          THCTensorMathMagma.cuh
          THCThrustAllocator.cuh
          THCTensorMode.cuh
          THCTensorTopK.cuh
          # See Note [TH abstraction violation]
          THCTensor.hpp
          THCStorage.hpp
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

INSTALL(FILES
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
          generic/THCTensorMasked.h
          generic/THCTensorMasked.cu
          generic/THCTensorMath.h
          generic/THCTensorMath.cu
          generic/THCTensorMathBlas.cu
          generic/THCTensorMathBlas.h
          generic/THCTensorMathCompare.h
          generic/THCTensorMathCompare.cu
          generic/THCTensorMathCompareT.h
          generic/THCTensorMathCompareT.cu
          generic/THCTensorMathMagma.h
          generic/THCTensorMathMagma.cu
          generic/THCTensorMathPairwise.h
          generic/THCTensorMathPairwise.cu
          generic/THCTensorMathPointwise.h
          generic/THCTensorMathPointwise.cu
          generic/THCTensorMathReduce.h
          generic/THCTensorMathReduce.cu
          generic/THCTensorMathScan.h
          generic/THCTensorMathScan.cu
          generic/THCTensorScatterGather.h
          generic/THCTensorScatterGather.cu
          generic/THCTensorIndex.h
          generic/THCTensorIndex.cu
          generic/THCTensorSort.h
          generic/THCTensorSort.cu
          generic/THCTensorRandom.h
          generic/THCTensorRandom.cu
          generic/THCTensorMode.h
          generic/THCTensorMode.cu
          generic/THCTensorTopK.h
          generic/THCTensorTopK.cu
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
