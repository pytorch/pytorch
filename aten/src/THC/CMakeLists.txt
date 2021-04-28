set(ATen_CUDA_INCLUDE ${ATen_CUDA_INCLUDE}
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
PARENT_SCOPE)

configure_file(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

set(extra_src)
# loop over all types
foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THC_FILE TensorMathPointwise TensorMathReduce)
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
         file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu"
           "#include <THC/THC${THC_FILE}.cuh>\n#include <THC/THCTensor.hpp>\n\n#include <THC/generic/THC${THC_FILE}.cu>\n#include <THC/THCGenerate${THC_TYPE}Type.h>\n")
      endif()
      list(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
   endforeach()
endforeach()

foreach(THC_FILE TensorMathPointwise TensorMathReduce)
   if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}Bool.cu")
      file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}Bool.cu"
        "#include <THC/THC${THC_FILE}.cuh>\n#include <THC/THCTensor.hpp>\n\n#include <THC/generic/THC${THC_FILE}.cu>\n#include <THC/THCGenerateBoolType.h>\n")
   endif()
   list(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}Bool.cu")
endforeach()

foreach(THC_FILE TensorMathReduce)
   if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}BFloat16.cu")
      file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}BFloat16.cu"
        "#include <THC/THC${THC_FILE}.cuh>\n#include <THC/THCTensor.hpp>\n\n#include <THC/generic/THC${THC_FILE}.cu>\n#include <THC/THCGenerateBFloat16Type.h>\n")
   endif()
   list(APPEND extra_src "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}BFloat16.cu")
endforeach()

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
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathMagma.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathPairwise.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorMathReduce.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCTensorScatterGather.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/THCSortUtils.cu
  ${extra_src}
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
          THCReduce.cuh
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
          THCGenerateComplexFloatType.h
          THCGenerateComplexTypes.h
          THCGenerateComplexDoubleType.h
          THCIntegerDivider.cuh
          THCNumerics.cuh
          THCTensorInfo.cuh
          THCTensorMathPointwise.cuh
          THCTensorTypeUtils.cuh
          THCTensorMathMagma.cuh
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
          generic/THCTensorMathMagma.cu
          generic/THCTensorMathPairwise.h
          generic/THCTensorMathPairwise.cu
          generic/THCTensorMathPointwise.h
          generic/THCTensorMathPointwise.cu
          generic/THCTensorMathReduce.h
          generic/THCTensorMathReduce.cu
          generic/THCTensorScatterGather.h
          generic/THCTensorScatterGather.cu
          DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
