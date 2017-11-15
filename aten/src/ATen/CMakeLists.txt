
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# avoid some cmake warnings
IF(POLICY CMP0026)
 CMAKE_POLICY(SET CMP0026 OLD)
ENDIF()

if(${CMAKE_VERSION} VERSION_LESS "2.8.12")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else(${CMAKE_VERSION} VERSION_LESS "2.8.12")
  if(${CMAKE_VERSION} VERSION_LESS "3.1")
    add_compile_options(-std=c++11) # CMake 2.8.12 to 3.1
  endif(${CMAKE_VERSION} VERSION_LESS "3.1")
endif(${CMAKE_VERSION} VERSION_LESS "2.8.12")

IF(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-vla")
ENDIF(NOT MSVC)

########################
# SET_SOURCE_FILES_PROPERTIES must be in the same CMakeLists.txt file as the target that includes the file
# so we need to set these commands here rather than in src/TH
IF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
  IF(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "/Ox /fp:fast")
  ELSE(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math")
  ENDIF(MSVC)
ENDIF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
IF(C_AVX_FOUND)
  IF(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "/Ox /fp:fast ${C_AVX_FLAGS}")
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX.c PROPERTIES COMPILE_FLAGS "/Ox /arch:AVX ${C_AVX_FLAGS}")
  ELSE(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math ${C_AVX_FLAGS}")
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX.c PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX_FLAGS}")
  ENDIF(MSVC)
ENDIF(C_AVX_FOUND)

IF(C_AVX2_FOUND)
  IF(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX2.c PROPERTIES COMPILE_FLAGS "/Ox /arch:AVX2 ${C_AVX2_FLAGS}")
  ELSE(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX2.c PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX2_FLAGS}")
  ENDIF(MSVC)
ENDIF(C_AVX2_FOUND)

IF(NOT MSVC)
  SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/THAtomic.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
  SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/THAllocator.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
ENDIF()
########################

################################################################################
# Helper functions
################################################################################

FUNCTION(EXCLUDE_DIR list_name dir_name)
  # A helper that excludes all files that contain dir_name in their file path
  SET(local_list ${${list_name}})
  FOREACH(source ${local_list})
    IF(${source} MATCHES ${dir_name})
      MESSAGE(STATUS "Excluding " ${source} " from the build")
      LIST(REMOVE_ITEM local_list ${source})
    ENDIF()
  ENDFOREACH()
  SET(${list_name} ${local_list} PARENT_SCOPE)
ENDFUNCTION()

function(filter_list output input)
    unset(result)
    foreach(filename ${${input}})
        foreach(pattern ${ARGN})
            if("${filename}" MATCHES "${pattern}")
                list(APPEND result "${filename}")
            endif()
        endforeach()
    endforeach()
    set(${output} ${result} PARENT_SCOPE)
endfunction()

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch REQUIRED)
ENDIF()

IF ($ENV{TH_BINARY_BUILD})
  MESSAGE(STATUS "TH_BINARY_BUILD detected. Statically linking libstdc++")
  SET(CMAKE_CXX_FLAGS "-static-libstdc++ ${CMAKE_CXX_FLAGS}")
  IF (UNIX AND NOT APPLE)
    # hiding statically linked library symbols, this flag is not available for the linker under MACOSX
    SET(CMAKE_CXX_FLAGS "-Wl,--exclude-libs,libstdc++.a ${CMAKE_CXX_FLAGS}")
  ENDIF(UNIX AND NOT APPLE)
ENDIF()

IF(NO_CUDA)
  MESSAGE(STATUS "ignoring CUDA")
  SET(CUDA_FLAG -n)
ELSE()
  FIND_PACKAGE(CUDA 5.5 REQUIRED)
  ADD_DEFINITIONS(-DAT_CUDA_ENABLED)
  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
ENDIF()

# Can be compiled standalone
IF(NOT AT_INSTALL_BIN_DIR OR NOT AT_INSTALL_LIB_DIR OR NOT AT_INSTALL_INCLUDE_DIR OR NOT AT_INSTALL_SHARE_DIR)
  SET(AT_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
  SET(AT_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
  SET(AT_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
  SET(AT_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")
ENDIF()

FILE(GLOB base_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
FILE(GLOB base_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
FILE(GLOB native_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "native/*.cpp")

FILE(GLOB all_python RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")

IF(DEFINED ENV{PYTORCH_PYTHON})
  message(STATUS "Using python found in $ENV{PYTORCH_PYTHON}")
  SET(PYCMD "$ENV{PYTORCH_PYTHON}")
ELSE()
  SET(PYCMD "python")
ENDIF()

SET(GEN_COMMAND
    ${PYCMD} ${CMAKE_CURRENT_SOURCE_DIR}/gen.py ${CUDA_FLAG}
    -s ${CMAKE_CURRENT_SOURCE_DIR}
    ${cwrap_files}
)

EXECUTE_PROCESS(
    COMMAND ${GEN_COMMAND} --output-dependencies ${CMAKE_CURRENT_BINARY_DIR}/generated_cpp.txt
    RESULT_VARIABLE RETURN_VALUE
)
if (NOT RETURN_VALUE EQUAL 0)
    message(STATUS ${generated_cpp})
    message(FATAL_ERROR "Failed to get generated_cpp list")
endif()
file(READ ${CMAKE_CURRENT_BINARY_DIR}/generated_cpp.txt generated_cpp)

FILE(GLOB_RECURSE all_templates "templates/*")

FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ATen)

ADD_CUSTOM_COMMAND(OUTPUT ${generated_cpp}
COMMAND ${GEN_COMMAND}
DEPENDS ${all_python} ${all_templates} ${cwrap_files})

SET(all_cpp ${base_cpp} ${native_cpp} ${generated_cpp} ${ATen_CPU_SRCS})

INCLUDE_DIRECTORIES(${ATen_CPU_INCLUDE})
IF(NOT NO_CUDA)
  INCLUDE_DIRECTORIES(${ATen_CUDA_INCLUDE})
  INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")
  INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/cuda")
  SET(native_functions_cuda ${CMAKE_CURRENT_SOURCE_DIR}/native/cuda/NativeFunctionsCuda.cu)
  SET(all_cpp ${all_cpp} ${ATen_CUDA_SRCS} ${native_functions_cuda})
endif()

filter_list(generated_h generated_cpp "\\.h$")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/..)
# so the build can find the generated header files
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
IF(NOT AT_LINK_STYLE)
  SET(AT_LINK_STYLE SHARED)
ENDIF()
IF(CUDA_FOUND)
  CUDA_ADD_LIBRARY(ATen ${AT_LINK_STYLE} ${all_cpp})
ELSE()
  ADD_LIBRARY(ATen ${AT_LINK_STYLE} ${all_cpp})
ENDIF()

SET_TARGET_PROPERTIES(ATen PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
    SET_PROPERTY(TARGET ATen PROPERTY CXX_STANDARD 11)
endif(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")

IF(BLAS_FOUND)
  IF ($ENV{TH_BINARY_BUILD})
    MESSAGE(STATUS "TH_BINARY_BUILD detected. Enabling special linkage.")
    TARGET_LINK_LIBRARIES(ATen "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
  ELSE ($ENV{TH_BINARY_BUILD})
    TARGET_LINK_LIBRARIES(ATen ${BLAS_LIBRARIES})
  ENDIF ($ENV{TH_BINARY_BUILD})
ENDIF(BLAS_FOUND)

IF(LAPACK_FOUND)
  TARGET_LINK_LIBRARIES(ATen ${LAPACK_LIBRARIES})
ENDIF(LAPACK_FOUND)

IF (UNIX AND NOT APPLE)
   INCLUDE(CheckLibraryExists)
   # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
   CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
   IF(NEED_LIBRT)
     TARGET_LINK_LIBRARIES(ATen rt)
     SET(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} rt)
   ENDIF(NEED_LIBRT)
ENDIF(UNIX AND NOT APPLE)

IF(UNIX)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
  # done for lseek: https://www.gnu.org/software/libc/manual/html_node/File-Position-Primitive.html
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

IF(NOT MSVC)
  TARGET_LINK_LIBRARIES(ATen m)
ENDIF(NOT MSVC)

# Is __thread supported?
IF(NOT MSVC)
  CHECK_C_SOURCE_COMPILES("static __thread int x = 1; int main() { return x; }" C_HAS_THREAD)
ELSE(NOT MSVC)
  CHECK_C_SOURCE_COMPILES("static __declspec( thread ) int x = 1; int main() { return x; }" C_HAS_THREAD)
ENDIF(NOT MSVC)
IF(NOT C_HAS_THREAD)
  MESSAGE(STATUS "Warning: __thread is not supported, generating thread-unsafe code")
ELSE(NOT C_HAS_THREAD)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTH_HAVE_THREAD")
ENDIF(NOT C_HAS_THREAD)

IF(CUDA_FOUND)
  TARGET_LINK_LIBRARIES(ATen
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_curand_LIBRARY})
  CUDA_ADD_CUBLAS_TO_TARGET(ATen)
  IF(USE_CUDNN)
    TARGET_LINK_LIBRARIES(ATen ${CUDNN_LIBRARIES})
  ENDIF(USE_CUDNN)
  IF(USE_MAGMA)
    TARGET_LINK_LIBRARIES(ATen ${MAGMA_LIBRARIES})
    IF ($ENV{TH_BINARY_BUILD})
      # because magma is linked statically and it wants a BLAS,
      # we need to link the BLAS lib against THC. Usually TH will
      # load a BLAS library and it's all fine, but in the binary builds,
      # TH uses static linkage to MKL, so it doesn't have all symbols that
      # magma needs. So in this case, explicitly find a BLAS and link against it
      # just like in TH
      SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../TH/cmake ${CMAKE_MODULE_PATH})
      FIND_PACKAGE(BLAS)
      IF(BLAS_FOUND)
        TARGET_LINK_LIBRARIES(ATen "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
      ELSE(BLAS_FOUND)
        MESSAGE(FATAL_ERROR "Binary build needs blas to be found here")
      ENDIF(BLAS_FOUND)
    ENDIF($ENV{TH_BINARY_BUILD})
  ENDIF(USE_MAGMA)
ENDIF()

INSTALL(TARGETS ATen
  RUNTIME DESTINATION "${AT_INSTALL_BIN_DIR}"
  LIBRARY DESTINATION "${AT_INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${AT_INSTALL_LIB_DIR}")

GET_TARGET_PROPERTY(ATEN_OUTPUT_NAME ATen LOCATION)
GET_FILENAME_COMPONENT(ATEN_OUTPUT_NAME ${ATEN_OUTPUT_NAME} NAME)
SET(ATEN_LIBRARIES "${CMAKE_INSTALL_PREFIX}/${AT_INSTALL_LIB_DIR}/${ATEN_OUTPUT_NAME}")
SET(ATEN_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${AT_INSTALL_INCLUDE_DIR}")
CONFIGURE_FILE(ATenConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake")
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake-exports/ATenConfig.cmake"
  DESTINATION "${AT_INSTALL_SHARE_DIR}/cmake/ATen")

FOREACH(HEADER ${base_h})
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
FOREACH(HEADER ${generated_h})
  INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
  DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen)
ENDFOREACH()
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)
