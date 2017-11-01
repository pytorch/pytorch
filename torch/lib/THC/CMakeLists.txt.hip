CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

# HIP_PATH
IF(NOT DEFINED $ENV{HIP_PATH})
  SET(HIP_PATH /opt/rocm/hip)
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

# HCC_PATH
IF(NOT DEFINED $ENV{HCC_PATH})
  SET(HCC_PATH /opt/rocm/hcc)
ELSE()
  SET(HCC_PATH $ENV{HCC_PATH})
ENDIF()

# HIPBLAS_PATH
IF(NOT DEFINED $ENV{HIPBLAS_PATH})
  SET(HIPBLAS_PATH /opt/rocm/hipblas)
ELSE()
  SET(HIPBLAS_PATH $ENV{HIPBLAS_PATH})
ENDIF()

# HIPRNG_PATH
IF(NOT DEFINED $ENV{HIPRNG_PATH})
  SET(HIPRNG_PATH /opt/rocm/hcrng)
ELSE()
  SET(HIPRNG_PATH $ENV{HIPRNG_PATH})
ENDIF()

# HIPSPARSE_PATH
IF(NOT DEFINED $ENV{HIPSPARSE_PATH})
  SET(HIPSPARSE_PATH /opt/rocm/hcsparse)
ELSE()
  SET(HIPSPARSE_PATH $ENV{HIPSPARSE_PATH})
ENDIF()

SET(THRUST_PATH "/root/Thrust")

# load HIP cmake module and load platform id
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")
FIND_PACKAGE(HIP 1.0 REQUIRED)
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE PLATFORM)
EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_CXX_FLAGS)
SET(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

SET(HIP_CXX_FLAGS "-D__HIP_PLATFORM_HCC__ -I/opt/rocm/hip/include -I/opt/rocm/hcc/include" ${HIP_CXX_FLAGS})

SET(HIP_HIPCC_FLAGS "-DGENERIC_GRID_LAUNCH=1 ${CMAKE_CXX_FLAGS}")
SET(HIP_HIPCC_FLAGS "-DGENERIC_GRID_LAUNCH=1 ${HIP_HIPCC_FLAGS}")
SET(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
SET(CMAKE_C_COMPILER ${HIP_HIPCC_EXECUTABLE})
SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration ${CMAKE_C_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS  "-std=c++11 ${CMAKE_CXX_FLAGS} ${HIP_CXX_FLAGS}")

IF (${PLATFORM} MATCHES "hcc")
  SET(CMAKE_MODULE_LINKER_FLAGS "-Wl,-Bsymbolic")
ENDIF()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

FIND_PACKAGE(HIP 1.0 REQUIRED)

MESSAGE(STATUS "ROCM TRUE:")
MESSAGE(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})

OPTION(NDEBUG "disable asserts (WARNING: this may result in invalid memory accesses)")
IF(NOT NDEBUG)
  MESSAGE(STATUS "Removing -DNDEBUG from compile flags")
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS "" ${CMAKE_C_FLAGS})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_DEBUG "" ${CMAKE_C_FLAGS_DEBUG})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "" ${CMAKE_C_FLAGS_RELEASE})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS "" ${CMAKE_CXX_FLAGS})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_DEBUG "" ${CMAKE_CXX_FLAGS_DEBUG})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "" ${CMAKE_CXX_FLAGS_RELEASE})
ENDIF()

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch)
ENDIF()

IF(NOT TH_LIBRARIES)
  SET(TH_LIBRARIES "TH")
ENDIF(NOT TH_LIBRARIES)
MESSAGE(STATUS "TH_LIBRARIES: ${TH_LIBRARIES}")

IF ("$ENV{STATIC_TH}" STREQUAL "YES")
LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
ENDIF()

IF ($ENV{TH_BINARY_BUILD})
  MESSAGE(STATUS "TH_BINARY_BUILD detected. Statically linking libstdc++")
  SET(CMAKE_CXX_FLAGS "-static-libstdc++ ${CMAKE_CXX_FLAGS}")
ENDIF()

INCLUDE_DIRECTORIES(${HIPBLAS_PATH}/include)
INCLUDE_DIRECTORIES(${HIPSPARSE_PATH}/include)
INCLUDE_DIRECTORIES(${HIPRNG_PATH}/include)
INCLUDE_DIRECTORIES(${THRUST_PATH})

IF(NOT THC_INSTALL_BIN_SUBDIR
    OR NOT THC_INSTALL_LIB_SUBDIR
    OR NOT THC_INSTALL_INCLUDE_SUBDIR
    OR NOT THC_INSTALL_CMAKE_SUBDIR)

  INCLUDE_DIRECTORIES(${TH_INCLUDE_PATH} ${TH_INCLUDE_PATH}/TH)
  LINK_DIRECTORIES(${TH_LIB_PATH})

  IF(Torch_INSTALL_BIN_SUBDIR)
    SET(THC_INSTALL_BIN_SUBDIR ${Torch_INSTALL_BIN_SUBDIR})
    SET(THC_INSTALL_LIB_SUBDIR ${Torch_INSTALL_LIB_SUBDIR})
    SET(THC_INSTALL_INCLUDE_SUBDIR ${Torch_INSTALL_INCLUDE_SUBDIR})
    SET(THC_INSTALL_CMAKE_SUBDIR ${Torch_INSTALL_CMAKE_SUBDIR})
  ELSE(Torch_INSTALL_BIN_SUBDIR)
    # not installing in a Torch context, so Torch_INSTALL_BIN_SUBDIR is not available
    SET(THC_INSTALL_BIN_SUBDIR "bin" CACHE PATH "THC install binary subdirectory")
    SET(THC_INSTALL_LIB_SUBDIR "lib" CACHE PATH "THC install library subdirectory")
    SET(THC_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "THC install include subdirectory")
    SET(THC_INSTALL_CMAKE_SUBDIR "share/cmake/THC" CACHE PATH "THC install cmake subdirectory")
  ENDIF(Torch_INSTALL_BIN_SUBDIR)

ENDIF()

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h")

SET(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")

SET(src
    THCCachingAllocator.cpp
    THCCachingHostAllocator.cpp
    THCGeneral.cc
    THCStorageCopy.c
    THCStream.cpp
    THCTensor.c
    THCTensorCopy.c
    THCTensorRandom.cpp
    THCThreadLocal.c
    )

SET(src-cuda
  THCReduceApplyUtils.cu
  THCBlas.cu
  THCSleep.cu
  THCStorage.cu
  THCStorageCopy.cu
  THCTensor.cu
  THCTensorCopy.cu
  THCTensorMath.cu
  THCTensorMath2.cu
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
  )

# loop over all types
foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THC_FILE TensorSort TensorMathCompareT TensorMathPointwise TensorMathCompare TensorMathReduce TensorMasked)
      if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
         FILE(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu"
              "#include \"../THC${THC_FILE}.cuh\"\n#include \"../generic/THC${THC_FILE}.cu\"\n#include \"../THCGenerate${THC_TYPE}Type.h\"\n")
      endif()
      LIST(APPEND src-cuda "${CMAKE_CURRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.cu")
   endforeach()
endforeach()

list(APPEND src-cuda THCHalf.cu)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-Bsymbolic")
SET(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -DGENERIC_GRID_LAUNCH=1 -Wall")


FIND_LIBRARY(HIPBLAS_LIBRARY hipblas HINTS ${HIPBLAS_PATH}/lib)
FIND_LIBRARY(HIPRNG_LIBRARY hiprng_hcc HINTS ${HIPRNG_PATH}/lib)
HIP_ADD_LIBRARY(THC SHARED ${src} ${src-cuda}) 
TARGET_LINK_LIBRARIES(THC ${TH_LIBRARIES} hip_hcc ${HIPBLAS_LIBRARY} ${HIPRNG_LIBRARY})

IF(NOT THC_SO_VERSION)
  SET(THC_SO_VERSION 0)
ENDIF(NOT THC_SO_VERSION)
MESSAGE(STATUS "THC_SO_VERSION: ${THC_SO_VERSION}")
SET_TARGET_PROPERTIES(THC PROPERTIES
  VERSION   ${THC_SO_VERSION}
  SOVERSION ${THC_SO_VERSION})

INSTALL(TARGETS THC
  RUNTIME DESTINATION "${THC_INSTALL_BIN_SUBDIR}"
  LIBRARY DESTINATION "${THC_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${THC_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          THC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THCGeneral.h
          THCBlas.h
          THCSleep.h
          THCStorage.h
          THCStorageCopy.h
          THCStream.h
          THCThreadLocal.h
          THCTensor.h
          THCTensorCopy.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorConv.h
          THCApply.cuh
          THCReduce.cuh
          THCReduceAll.cuh
          THCReduceApplyUtils.cuh
          THCAsmUtils.cuh
          THCAtomics.cuh
          THCScanUtils.cuh
          THCSortUtils.cuh
          THCAllocator.h
          THCCachingAllocator.h
          THCCachingHostAllocator.h
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
          DESTINATION "${THC_INSTALL_INCLUDE_SUBDIR}/THC/generic")
