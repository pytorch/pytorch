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
IF(C_SSE2_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE2_FLAGS} -DUSE_SSE2 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE2_FOUND)
IF(C_SSE3_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE3_FOUND)
IF(C_SSE4_1_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE4_1_FLAGS} -DUSE_SSE4_1 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE4_1_FOUND)
IF(C_SSE4_2_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE4_2_FLAGS} -DUSE_SSE4_2 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE4_2_FOUND)

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

IF(BLAS_IS_ACCELERATE)
  MESSAGE(STATUS "BLAS FOUND IS ACCELERATE: Fix for sdot")
ENDIF()

SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
FOREACH(KEYWORD "inline" "__inline__" "__inline")
  IF(NOT DEFINED C_INLINE)

    SET(CMAKE_REQUIRED_FLAGS "-Dinline=${KEYWORD} ${CMAKE_C_FLAGS}")
    CHECK_C_SOURCE_RUNS("
       static inline int static_foo()
       {
         return 0;
       }

       int main(int argc, char *argv[])
       {
         static_foo();
         return 0;
       }" C_HAS_${KEYWORD})

    IF(C_HAS_${KEYWORD})
      SET(C_INLINE TRUE)
# Right now i put it in THGeneral.h -- debatable
#      ADD_DEFINITIONS("-Dinline=${KEYWORD}")
      SET(TH_INLINE ${KEYWORD})
      MESSAGE(STATUS "C inline is supported (${KEYWORD})")
    ENDIF(C_HAS_${KEYWORD})
  ENDIF(NOT DEFINED C_INLINE)
ENDFOREACH(KEYWORD)
SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

IF(NOT DEFINED C_INLINE)
  MESSAGE(STATUS "C inline seems not supported")
# Right now i put it in THGeneral.h -- debatable
#  ADD_DEFINITIONS("-Dinline=")
SET(TH_INLINE "")
ENDIF(NOT DEFINED C_INLINE)

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


IF (WIN32 AND NOT CYGWIN)
  SET(BLAS_INSTALL_LIBRARIES "OFF"
    CACHE BOOL "Copy the required BLAS DLLs into the Torch install dirs")
ENDIF (WIN32 AND NOT CYGWIN)

MACRO(Install_Required_Library ln)
    get_filename_component(libpath ${ln} PATH)
    get_filename_component(libname ${ln} NAME_WE)
    file(GLOB libdlls "${libpath}/${libname}*.dll")
    install(PROGRAMS ${libdlls} 
      DESTINATION "${Torch_INSTALL_BIN_SUBDIR}")
ENDMACRO(Install_Required_Library libname)

IF (BLAS_FOUND AND BLAS_INSTALL_LIBRARIES)
  IF (BLAS_goto2_LIBRARY)
    Install_Required_Library(${BLAS_goto2_LIBRARY})
    Install_Required_Library("${libpath}/libgfortran")
    Install_Required_Library("${libpath}/libquadmath")
    Install_Required_Library("${libpath}/libgcc")
  ENDIF()
  IF (BLAS_openblas_LIBRARY)
    Install_Required_Library(${BLAS_openblas_LIBRARY})
    Install_Required_Library("${libpath}/libquadmath")
    Install_Required_Library("${libpath}/libgfortran")
    Install_Required_Library("${libpath}/libquadmath")
    Install_Required_Library("${libpath}/libgcc")
  ENDIF()
ENDIF()



# Create THConfig.cmake
GET_TARGET_PROPERTY(TH_OUTPUT_NAME TH LOCATION)
GET_FILENAME_COMPONENT(TH_OUTPUT_NAME ${TH_OUTPUT_NAME} NAME)
SET(TH_LIBRARIES "${Torch_INSTALL_LIB}/${TH_OUTPUT_NAME}")
SET(TH_INCLUDE_DIR "${Torch_INSTALL_INCLUDE}/TH")
CONFIGURE_FILE(THConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake" 
  DESTINATION "${Torch_INSTALL_CMAKE_SUBDIR}")
