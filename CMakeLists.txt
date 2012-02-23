# -*- cmake -*-

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

SET(hdr 
  THGeneral.h THStorage.h THTensor.h THTensorApply.h
  THBlas.h THLapack.h THLogAdd.h THRandom.h THVector.h)
SET(src 
  THGeneral.c THStorage.c THTensor.c THBlas.c THLapack.c
  THLogAdd.c THRandom.c
  THFile.c THDiskFile.c THMemoryFile.c)

SET(src ${src} ${hdr})

IF(UNIX)
  INCLUDE(CheckFunctionExists)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
ENDIF(UNIX)

ADD_LIBRARY(TH SHARED ${src})

FIND_PACKAGE(SSE)
IF (SSE2_FOUND)
  SET(CMAKE_C_FLAGS "-msse2 -D__SSE2__ ${CMAKE_C_FLAGS}")
ENDIF (SSE2_FOUND)
IF (SSE3_FOUND)
  SET(CMAKE_C_FLAGS "-msse3 -D__SSE3__ ${CMAKE_C_FLAGS}")
ENDIF (SSE3_FOUND)
IF (SSSE3_FOUND)
  SET(CMAKE_C_FLAGS "-mssse3 -D__SSSE3__ ${CMAKE_C_FLAGS}")
ENDIF (SSSE3_FOUND)
IF (SSE4.1_FOUND)
  SET(CMAKE_C_FLAGS "-msse4.1 -D__SSE4_1__ ${CMAKE_C_FLAGS}")
ENDIF (SSE4.1_FOUND)

FIND_PACKAGE(BLAS)
IF(BLAS_FOUND)
  SET(USE_BLAS 1)
  TARGET_LINK_LIBRARIES(TH ${BLAS_LIBRARIES})
ENDIF(BLAS_FOUND)

FIND_PACKAGE(LAPACK)
IF(LAPACK_FOUND)
  SET(USE_LAPACK 1)
  TARGET_LINK_LIBRARIES(TH ${LAPACK_LIBRARIES})
ENDIF(LAPACK_FOUND)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h")

INSTALL(TARGETS TH
  EXPORT torch-exports
  RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
  TH.h
  THBlas.h
  THDiskFile.h
  THFile.h
  THFilePrivate.h
  ${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h
  THGenerateAllTypes.h
  THGenerateFloatTypes.h
  THGenerateIntTypes.h
  THLapack.h
  THLogAdd.h
  THMemoryFile.h
  THRandom.h
  THStorage.h
  THTensor.h
  THTensorApply.h
  THTensorDimApply.h
  THTensorMacros.h
  THVector.h
  DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/TH")

INSTALL(FILES
  generic/THBlas.c
  generic/THBlas.h
  generic/THLapack.c
  generic/THLapack.h
  generic/THStorage.c
  generic/THStorage.h
  generic/THStorageCopy.c
  generic/THStorageCopy.h
  generic/THTensor.c
  generic/THTensor.h
  generic/THTensorConv.c
  generic/THTensorConv.h
  generic/THTensorCopy.c
  generic/THTensorCopy.h
  generic/THTensorLapack.c
  generic/THTensorLapack.h
  generic/THTensorMath.c
  generic/THTensorMath.h
  generic/THTensorRandom.c
  generic/THTensorRandom.h
  generic/THVector.c
  DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/TH/generic")

# Create THConfig.cmake
GET_TARGET_PROPERTY(TH_OUTPUT_NAME TH LOCATION)
GET_FILENAME_COMPONENT(TH_OUTPUT_NAME ${TH_OUTPUT_NAME} NAME)
SET(TH_LIBRARIES "${Torch_INSTALL_LIB}/${TH_OUTPUT_NAME}")
SET(TH_INCLUDE_DIR "${Torch_INSTALL_INCLUDE}/TH")
CONFIGURE_FILE(THConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake" 
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")
