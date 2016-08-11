cmake_minimum_required(VERSION 2.6)

# avoid some cmake warnings
IF(POLICY CMP0026)
 CMAKE_POLICY(SET CMP0026 OLD)
ENDIF()

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Can be compiled standalone
IF(NOT TH_INSTALL_BIN_SUBDIR
    OR NOT TH_INSTALL_LIB_SUBDIR
    OR NOT TH_INSTALL_INCLUDE_SUBDIR
    OR NOT TH_INSTALL_CMAKE_SUBDIR)

  SET(TH_INSTALL_BIN_SUBDIR "bin" CACHE PATH "TH install binary subdirectory")
  SET(TH_INSTALL_LIB_SUBDIR "lib" CACHE PATH "TH install library subdirectory")
  SET(TH_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "TH install include subdirectory")
  SET(TH_INSTALL_CMAKE_SUBDIR "share/cmake/TH" CACHE PATH "TH install cmake subdirectory")
ENDIF()

# flags

IF(MSVC)
  # respect the standard
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
ENDIF(MSVC)

# OpenMP support?
SET(WITH_OPENMP ON CACHE BOOL "OpenMP support if available?")
IF (APPLE AND CMAKE_COMPILER_IS_GNUCC)
  EXEC_PROGRAM (uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  STRING (REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  MESSAGE (STATUS "MAC OS Darwin Version: ${DARWIN_VERSION}")
  IF (DARWIN_VERSION GREATER 9)
    SET(APPLE_OPENMP_SUCKS 1)
  ENDIF (DARWIN_VERSION GREATER 9)
  EXECUTE_PROCESS (COMMAND ${CMAKE_C_COMPILER} -dumpversion
    OUTPUT_VARIABLE GCC_VERSION)
  IF (APPLE_OPENMP_SUCKS AND GCC_VERSION VERSION_LESS 4.6.2)
    MESSAGE(STATUS "Warning: Disabling OpenMP (unstable with this version of GCC)")
    MESSAGE(STATUS " Install GCC >= 4.6.2 or change your OS to enable OpenMP")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
    SET(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?" FORCE)
  ENDIF ()
ENDIF ()

IF (WITH_OPENMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    MESSAGE(STATUS "Compiling with OpenMP support")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  ENDIF(OPENMP_FOUND)
ENDIF (WITH_OPENMP)

# ARM specific flags
FIND_PACKAGE(ARM)
IF (NEON_FOUND)
  MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
  SET(CMAKE_C_FLAGS "-mfpu=neon -D__NEON__ ${CMAKE_C_FLAGS}")
ENDIF (NEON_FOUND)
IF (CORTEXA8_FOUND)
  MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a8 -fprefetch-loop-arrays ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA8_FOUND)
IF (CORTEXA9_FOUND)
  MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a9 ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA9_FOUND)

IF(UNIX)
  INCLUDE(CheckFunctionExists)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
  ADD_DEFINITIONS(-D_FILE_OFFSET_BITS=64)
  CHECK_FUNCTION_EXISTS(shm_open HAVE_SHM_OPEN)
  IF(HAVE_SHM_OPEN)
    ADD_DEFINITIONS(-DHAVE_SHM_OPEN=1)
  ENDIF(HAVE_SHM_OPEN)
  CHECK_FUNCTION_EXISTS(shm_unlink HAVE_SHM_UNLINK)
  IF(HAVE_SHM_UNLINK)
    ADD_DEFINITIONS(-DHAVE_SHM_UNLINK=1)
  ENDIF(HAVE_SHM_UNLINK)
  CHECK_FUNCTION_EXISTS(malloc_usable_size HAVE_MALLOC_USABLE_SIZE)
  IF(HAVE_MALLOC_USABLE_SIZE)
    ADD_DEFINITIONS(-DHAVE_MALLOC_USABLE_SIZE=1)
  ENDIF(HAVE_MALLOC_USABLE_SIZE)
ENDIF(UNIX)

FIND_PACKAGE(SSE)
IF(C_SSE2_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE2_FLAGS} -DUSE_SSE2 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE2_FOUND)
IF(C_SSE3_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE3_FLAGS} -DUSE_SSE3 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE3_FOUND)

IF(C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)
  SET(simd generic/simd/convolve.c)
  SET_SOURCE_FILES_PROPERTIES(generic/simd/convolve.c PROPERTIES COMPILE_FLAGS "-std=c99")
ENDIF(C_AVX_FOUND OR C_SSE4_2_FOUND OR C_SSE4_1_FOUND)

IF(C_SSE4_1_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE4_1_FLAGS} -DUSE_SSE4_1 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE4_1_FOUND)
IF(C_SSE4_2_FOUND)
  SET(CMAKE_C_FLAGS "${C_SSE4_2_FLAGS} -DUSE_SSE4_2 ${CMAKE_C_FLAGS}")
ENDIF(C_SSE4_2_FOUND)

IF(C_SSE4_1_FOUND OR C_SSE4_2_FOUND)
  SET_SOURCE_FILES_PROPERTIES(generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math -std=c99")
  SET(simd ${simd} generic/simd/convolve5x5_sse.c)
ENDIF(C_SSE4_1_FOUND OR C_SSE4_2_FOUND)

IF(C_AVX_FOUND)
  SET(CMAKE_C_FLAGS "-DUSE_AVX ${CMAKE_C_FLAGS}")
  SET_SOURCE_FILES_PROPERTIES(generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math -mavx -std=c99")
  SET(simd ${simd} generic/simd/convolve5x5_avx.c)
ENDIF(C_AVX_FOUND)

SET(hdr
  THGeneral.h THAllocator.h THStorage.h THTensor.h THTensorApply.h THBlas.h THMath.h
  THLapack.h THLogAdd.h THRandom.h THVector.h THAtomic.h)

SET(src
  THGeneral.c THAllocator.c THStorage.c THTensor.c THBlas.c THLapack.c
  THLogAdd.c THRandom.c THFile.c THDiskFile.c THMemoryFile.c THAtomic.c)

SET(src ${src} ${hdr} ${simd})
ADD_LIBRARY(TH SHARED ${src})
if(BUILD_STATIC)
  ADD_LIBRARY(TH_static STATIC ${src})
endif()

SET_TARGET_PROPERTIES(TH PROPERTIES
  VERSION   0
  SOVERSION 0)

CHECK_C_SOURCE_RUNS("
#include <stdatomic.h>
int main()
{
  int a;
  int oa;
  atomic_store(&a, 1);
  atomic_fetch_add(&a, 1);
  oa = atomic_load(&a);
  if(!atomic_compare_exchange_strong(&a, &oa, 3))
    return -1;
  return 0;
}
" HAS_C11_ATOMICS)

IF(NOT HAS_C11_ATOMICS)
  CHECK_C_SOURCE_RUNS("
#include <intrin.h>
int main()
{
  long a;
  _InterlockedExchange(&a, 1);
  _InterlockedExchangeAdd(&a, 1);
  if(_InterlockedCompareExchange(&a, 3, 2) != 2)
    return -1;
  return 0;
}
" HAS_MSC_ATOMICS)

  CHECK_C_SOURCE_RUNS("
int main()
{
  int a;
  __sync_lock_test_and_set(&a, 1);
  __sync_fetch_and_add(&a, 1);
  if(!__sync_bool_compare_and_swap(&a, 2, 3))
    return -1;
  return 0;
}
" HAS_GCC_ATOMICS)
ENDIF()

IF(HAS_C11_ATOMICS)
  ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)
  MESSAGE(STATUS "Atomics: using C11 intrinsics")
ELSEIF(HAS_MSC_ATOMICS)
  ADD_DEFINITIONS(-DUSE_MSC_ATOMICS=1)
  MESSAGE(STATUS "Atomics: using MSVC intrinsics")
ELSEIF(HAS_GCC_ATOMICS)
  ADD_DEFINITIONS(-DUSE_GCC_ATOMICS=1)
    MESSAGE(STATUS "Atomics: using GCC intrinsics")
ELSE()
  SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  FIND_PACKAGE(Threads)
  IF(THREADS_FOUND)
    ADD_DEFINITIONS(-DUSE_PTHREAD_ATOMICS=1)
    TARGET_LINK_LIBRARIES(TH ${CMAKE_THREAD_LIBS_INIT})
    MESSAGE(STATUS "Atomics: using pthread")
  ENDIF()
ENDIF()

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

IF (UNIX AND NOT APPLE)
   INCLUDE(CheckLibraryExists)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
   IF(NEED_LIBRT)
     TARGET_LINK_LIBRARIES(TH rt)
   ENDIF(NEED_LIBRT)
ENDIF(UNIX AND NOT APPLE)

IF(NOT MSVC)
  TARGET_LINK_LIBRARIES(TH m)
ENDIF(NOT MSVC)

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

# Is __thread supported?
INCLUDE (CheckCSourceCompiles)
CHECK_C_SOURCE_COMPILES("static __thread int x = 1; int main() { return x; }" C_HAS_THREAD)
IF(NOT DEFINED C_HAS_THREAD)
  MESSAGE(STATUS "Warning: __thread is not supported, generating thread-unsafe code")
ENDIF(NOT DEFINED C_HAS_THREAD)
IF(C_HAS_THREAD)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTH_HAVE_THREAD")
ENDIF(C_HAS_THREAD)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THGeneral.h")

INSTALL(TARGETS TH
  EXPORT TH-exports
  RUNTIME DESTINATION "${TH_INSTALL_BIN_SUBDIR}"
  LIBRARY DESTINATION "${TH_INSTALL_LIB_SUBDIR}"
  ARCHIVE DESTINATION "${TH_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
  TH.h
  THAllocator.h
  THMath.h
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
  THAtomic.h
  DESTINATION "${TH_INSTALL_INCLUDE_SUBDIR}/TH")

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
  DESTINATION "${TH_INSTALL_INCLUDE_SUBDIR}/TH/generic")


IF (WIN32 AND NOT CYGWIN)
  SET(BLAS_INSTALL_LIBRARIES "OFF"
    CACHE BOOL "Copy the required BLAS DLLs into the TH install dirs")
ENDIF (WIN32 AND NOT CYGWIN)

MACRO(Install_Required_Library ln)
    get_filename_component(libpath ${ln} PATH)
    get_filename_component(libname ${ln} NAME_WE)
    file(GLOB libdlls "${libpath}/${libname}*.dll")
    install(PROGRAMS ${libdlls}
      DESTINATION "${TH_INSTALL_BIN_SUBDIR}")
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
SET(TH_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${TH_INSTALL_LIB_SUBDIR}/${TH_OUTPUT_NAME}")
SET(TH_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${TH_INSTALL_INCLUDE_SUBDIR}/TH")
CONFIGURE_FILE(THConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/THConfig.cmake"
  DESTINATION "${TH_INSTALL_CMAKE_SUBDIR}")
