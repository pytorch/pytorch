
SET(src THCGeneral.c THCStorage.c THCTensor.c THCTensorCopy.c)

SET(src-cuda THC.cu)

CUDA_ADD_LIBRARY(THC SHARED ${src-cuda} ${src})
CUDA_ADD_CUBLAS_TO_TARGET(THC)
TARGET_LINK_LIBRARIES(THC TH ${CUDA_curand_LIBRARY})

INSTALL(TARGETS THC
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          THC.h
          THCGeneral.h
          THCGeneral.h
          THCStorage.h
          THCTensor.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorConv.h
          DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/THC")
