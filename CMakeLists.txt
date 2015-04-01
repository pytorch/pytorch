SET(src
    THCGeneral.c THCStorage.c THCStorageCopy.c THCTensor.c THCTensorCopy.c)

SET(src-cuda
  THCReduceApplyUtils.cu
  THCBlas.cu
  THCStorage.cu
  THCStorageCopy.cu
  THCTensor.cu
  THCTensorCopy.cu
  THCTensorMath2.cu
  THCTensorMathBlas.cu
  THCTensorMathCompare.cu
  THCTensorMathCompareT.cu
  THCTensorMath.cu
  THCTensorMathPairwise.cu
  THCTensorMathPointwise.cu
  THCTensorMathScan.cu
  THCTensorMathTransformReduce.cu
  THCTensorMasked.cu
  THCTensorIndex.cu
  THCTensorConv.cu
  THCTensorRandom.cu
  THCApply.cu
  THCTensorSort.cu
  )

CUDA_ADD_LIBRARY(THC SHARED ${src} ${src-cuda})
CUDA_ADD_CUBLAS_TO_TARGET(THC)
TARGET_LINK_LIBRARIES(THC TH ${CUDA_curand_LIBRARY})

INSTALL(TARGETS THC
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          THC.h
          THCGeneral.h
          THCBlas.h
          THCStorage.h
	  THCStorageCopy.h
          THCTensor.h
	  THCTensorCopy.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorConv.h
          THCTensorSort.h
          THCApply.cuh
          THCReduce.cuh
          THCReduceApplyUtils.cuh
          DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/THC")
