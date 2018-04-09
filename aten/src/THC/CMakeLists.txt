cmake_minimum_required(VERSION 3.0)
project(THC)

if (NOT SUBPROJECT)
  include(../../cmake/Common.cmake)
  include(LoadCUDA)
  include(LoadMAGMA)
  add_subdirectory(../THRTS THRTS)
  add_subdirectory(../TH TH)
  add_subdirectory(../THCRTS THCRTS)
endif()

configure_file(THCGeneral.h.in THCGeneral.h)

set(SRC
  THCGeneral.cpp
  THCStorageCopy.cpp
  THCTensor.cpp
  THCTensorCopy.cpp
  THCTensorRandom.cpp
  THCStorage.cpp

  THCReduceApplyUtils.cu
  THCBlas.cu
  THCSleep.cu
  THCStorage.cu
  THCStorageCopy.cu
  THCTensor.cu
  THCTensorCopy.cu
  THCTensorMath.cu
  THCTensorMathBlas.cu
  THCTensorMathMagma.cu
  THCTensorMathPairwise.cu
  THCTensorMathReduce.cu
  THCTensorMathScan.cu
  THCTensorIndex.cu
  THCTensorConv.cu
  THCTensorRandom.cu
  THCTensorScatterGather.cu
  THCTensorTopK.cu
  THCTensorSort.cu
  THCTensorTypeUtils.cu
  THCSortUtils.cu
  THCTensorMode.cu
  THCHalf.cu
)

foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THC_FILE TensorSort TensorMathCompareT TensorMathPointwise TensorMathCompare TensorMathReduce TensorMasked)
      # TODO: This should not be placed in the source directory
      # TODO: This will not recompile correctly if the contents change
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
         file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu"
              "#include \"../THC${THC_FILE}.cuh\"\n#include \"../generic/THC${THC_FILE}.cu\"\n#include \"../THCGenerate${THC_TYPE}Type.h\"\n")
      endif()
      list(APPEND SRC "generated/THC${THC_FILE}${THC_TYPE}.cu")
   endforeach()
endforeach()

cuda_add_library(${PROJECT_NAME}_static STATIC ${SRC})
target_link_libraries(${PROJECT_NAME}_static PUBLIC THCRTS TH ${CUDA_LIBRARIES})
if(USE_MAGMA)
  target_link_libraries(${PROJECT_NAME}_static PUBLIC ${MAGMA_LIBRARIES})
endif()
target_include_directories(${PROJECT_NAME}_static PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${ATEN_INSTALL_INCLUDE_SUBDIR}>
  )
caffe2_interface_library(${PROJECT_NAME}_static ${PROJECT_NAME})

install(FILES
  THC.h
  ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
  THCBlas.h
  THCSleep.h
  THCStorage.h
  THCStorageCopy.h
  THCTensor.h
  THCTensorCopy.h
  THCTensorRandom.h
  THCTensorMath.h
  THCTensorConv.h
  THCApply.cuh
  THCReduce.cuh
  THCReduceAll.cuh
  THCReduceApplyUtils.cuh
  THCTensorMathReduce.cuh
  THCAsmUtils.cuh
  THCAtomics.cuh
  THCScanUtils.cuh
  THCSortUtils.cuh
  THCDeviceUtils.cuh
  THCDeviceTensor.cuh
  THCDeviceTensor-inl.cuh
  THCDeviceTensorUtils.cuh
  THCDeviceTensorUtils-inl.cuh
  THCGenerateAllTypes.h
  THCGenerateByteType.h
  THCGenerateCharType.h
  THCGenerateShortType.h
  THCGenerateIntType.h
  THCGenerateLongType.h
  THCGenerateHalfType.h
  THCGenerateFloatType.h
  THCGenerateFloatTypes.h
  THCGenerateDoubleType.h
  THCHalf.h
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
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC")

install(FILES
  generic/THCStorage.c
  generic/THCStorage.cu
  generic/THCStorage.h
  generic/THCTensor.c
  generic/THCTensor.cu
  generic/THCTensor.h
  generic/THCStorageCopy.c
  generic/THCStorageCopy.cu
  generic/THCStorageCopy.h
  generic/THCTensorCopy.c
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
  generic/THCDeviceTensorUtils.cu
  generic/THCTensorRandom.h
  generic/THCTensorRandom.cu
  generic/THCTensorMode.h
  generic/THCTensorMode.cu
  generic/THCTensorTopK.h
  generic/THCTensorTopK.cu
  DESTINATION "${ATEN_INSTALL_INCLUDE_SUBDIR}/THC/generic")
