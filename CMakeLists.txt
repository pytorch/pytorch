SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_MODULE_PATH})

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch)
ENDIF()
IF(NOT CUDA_FOUND)
  FIND_PACKAGE(CUDA 5.5 REQUIRED)
ENDIF()
IF(NOT MAGMA_FOUND)
  FIND_PACKAGE(MAGMA)
ENDIF()

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")

IF(MAGMA_FOUND)
  INCLUDE_DIRECTORIES(${MAGMA_INCLUDE_DIR})
  SET(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
  INCLUDE(CheckPrototypeDefinition)
  check_prototype_definition(magma_get_sgeqrf_nb
   "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
   "0"
   "magma.h"
    MAGMA_V2)
  IF (MAGMA_V2)
    add_definitions(-DMAGMA_V2)
  ENDIF (MAGMA_V2)

  SET(USE_MAGMA 1)
  MESSAGE(STATUS "Compiling with MAGMA support")
  MESSAGE(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
  MESSAGE(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
  MESSAGE(STATUS "MAGMA V2 check: ${MAGMA_V2}")
ELSE(MAGMA_FOUND)
  MESSAGE(STATUS "MAGMA not found. Compiling without MAGMA support")
ENDIF(MAGMA_FOUND)

IF(APPLE)
  IF(${CUDA_VERSION} LESS 6.0)
    # work around for mac os x bug:
    # http://stackoverflow.com/questions/16286588/cuda-5-0-cmake-and-make-failing-on-osx-10-8-3
    if (NOT DEFINED CUDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
      set(CUDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by NVCC")
      message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
    endif()

    # bug on Apple
    LINK_DIRECTORIES("/usr/local/cuda/lib/")
  ELSEIF(${CUDA_VERSION} LESS 7.0)
    SET(CUDA_HOST_COMPILER clang)
    LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -stdlib=libstdc++ -Xlinker -stdlib=libstdc++")
    IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    ENDIF()
  ENDIF()
  # CUDA 7 supports clang and libc++ so no need to change anything
ENDIF(APPLE)

# Detect CUDA architecture and get best NVCC flags
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FindCudaArch.cmake)
SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

IF(NOT THC_INSTALL_BIN_SUBDIR
    OR NOT THC_INSTALL_LIB_SUBDIR
    OR NOT THC_INSTALL_INCLUDE_SUBDIR
    OR NOT THC_INSTALL_CMAKE_SUBDIR)

  INCLUDE_DIRECTORIES(${TH_INCLUDE_PATH} ${TH_INCLUDE_PATH}/TH)
  LINK_DIRECTORIES(${TH_LIB_PATH})

  SET(THC_INSTALL_BIN_SUBDIR "bin" CACHE PATH "THC install binary subdirectory")
  SET(THC_INSTALL_LIB_SUBDIR "lib" CACHE PATH "THC install library subdirectory")
  SET(THC_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "THC install include subdirectory")
  SET(THC_INSTALL_CMAKE_SUBDIR "share/cmake/THC" CACHE PATH "THC install cmake subdirectory")
ELSE()
  SET(THC_INSTALL_BIN_SUBDIR ${Torch_INSTALL_BIN_SUBDIR})
  SET(THC_INSTALL_LIB_SUBDIR ${Torch_INSTALL_LIB_SUBDIR})
  SET(THC_INSTALL_INCLUDE_SUBDIR ${Torch_INSTALL_INCLUDE_SUBDIR})
  SET(THC_INSTALL_CMAKE_SUBDIR ${Torch_INSTALL_CMAKE_SUBDIR})
ENDIF()

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

SET(CMAKE_C_FLAGS "-std=c99")
SET(src
    THCGeneral.c THCAllocator.c THCStorage.c THCStorageCopy.c THCTensor.c THCTensorCopy.c)

SET(src-cuda
  THCReduceApplyUtils.cu
  THCBlas.cu
  THCStorage.cu
  THCStorageCopy.cu
  THCTensor.cu
  THCTensorCopy.cu
  THCTensorMath.cu
  THCTensorMath2.cu
  THCTensorMathBlas.cu
  THCTensorMathCompare.cu
  THCTensorMathCompareT.cu
  THCTensorMathMagma.cu
  THCTensorMathPairwise.cu
  THCTensorMathPointwise.cu
  THCTensorMathScan.cu
  THCTensorMathTransformReduce.cu
  THCTensorMasked.cu
  THCTensorIndex.cu
  THCTensorConv.cu
  THCTensorRandom.cu
  THCTensorScatterGather.cu
  THCApply.cu
  THCTensorSort.cu
  THCTensorTopK.cu
  )

CUDA_ADD_LIBRARY(THC SHARED ${src} ${src-cuda})
CUDA_ADD_CUBLAS_TO_TARGET(THC)
TARGET_LINK_LIBRARIES(THC TH ${CUDA_curand_LIBRARY})

IF(USE_MAGMA)
  TARGET_LINK_LIBRARIES(THC ${MAGMA_LIBRARIES})
ENDIF(USE_MAGMA)

INSTALL(TARGETS THC
          RUNTIME DESTINATION "${THC_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${THC_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${THC_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          THC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
          THCBlas.h
          THCStorage.h
          THCStorageCopy.h
          THCTensor.h
          THCTensorCopy.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorConv.h
          THCTensorSort.h
          THCTensorTopK.h
          THCApply.cuh
          THCReduce.cuh
          THCReduceAll.cuh
          THCReduceApplyUtils.cuh
          THCAsmUtils.cuh
          THCScanUtils.cuh
          THCSortUtils.cuh
          THCAllocator.h
          THCDeviceUtils.cuh
          THCDeviceTensor.cuh
          THCDeviceTensor-inl.cuh
          THCDeviceTensorUtils.cuh
          THCDeviceTensorUtils-inl.cuh
          THCGenerateAllTypes.h
          DESTINATION "${THC_INSTALL_INCLUDE_SUBDIR}/THC")

INSTALL(FILES
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
          DESTINATION "${THC_INSTALL_INCLUDE_SUBDIR}/THC/generic")
