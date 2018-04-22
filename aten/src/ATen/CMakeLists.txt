
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# avoid some cmake warnings
IF(POLICY CMP0026)
 CMAKE_POLICY(SET CMP0026 OLD)
ENDIF()

IF(MSVC AND NOT "${CMAKE_BUILD_TYPE}" MATCHES "Debug")
  SET(MSVC_OPT_FLAG "/Ox /fp:strict ")
  SET(VCOMP_LIB "vcomp")
ELSE()
  SET(MSVC_OPT_FLAG " ")
  SET(VCOMP_LIB "vcompd")
ENDIF()

########################
# SET_SOURCE_FILES_PROPERTIES must be in the same CMakeLists.txt file as the target that includes the file
# so we need to set these commands here rather than in src/TH
IF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
  IF(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/fp:fast")
  ELSE(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_sse.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math")
  ENDIF(MSVC)
ENDIF(C_SSE4_1_FOUND AND C_SSE4_2_FOUND)
IF(C_AVX_FOUND)
  IF(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/fp:fast ${C_AVX_FLAGS}")
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX.c PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/arch:AVX ${C_AVX_FLAGS}")
  ELSE(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/generic/simd/convolve5x5_avx.c PROPERTIES COMPILE_FLAGS "-O3 -ffast-math ${C_AVX_FLAGS}")
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX.c PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX_FLAGS}")
  ENDIF(MSVC)
ENDIF(C_AVX_FOUND)

IF(C_AVX2_FOUND)
  IF(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX2.cpp PROPERTIES COMPILE_FLAGS "${MSVC_OPT_FLAG}/arch:AVX2 ${C_AVX2_FLAGS}")
  ELSE(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/vector/AVX2.cpp PROPERTIES COMPILE_FLAGS "-O3 ${C_AVX2_FLAGS}")
  ENDIF(MSVC)
ENDIF(C_AVX2_FOUND)

IF(NOT MSVC AND NOT "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/THAtomic.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
  SET_SOURCE_FILES_PROPERTIES(${PROJECT_SOURCE_DIR}/src/TH/THAllocator.c PROPERTIES COMPILE_FLAGS "-fno-openmp")
ENDIF()

FILE(GLOB cpu_kernel_cpp_in RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "native/cpu/*.cpp")

LIST(APPEND CPU_CAPABILITY_NAMES "DEFAULT")
IF(MSVC)
  LIST(APPEND CPU_CAPABILITY_FLAGS "${MSVC_OPT_FLAG}")
ELSE(MSVC)
  LIST(APPEND CPU_CAPABILITY_FLAGS "-O3")
ENDIF(MSVC)

IF(CXX_AVX_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX_CPU_DEFINITION")
  LIST(APPEND CPU_CAPABILITY_NAMES "AVX")
  IF(MSVC)
    LIST(APPEND CPU_CAPABILITY_FLAGS "${MSVC_OPT_FLAG}/arch:AVX")
  ELSE(MSVC)
    LIST(APPEND CPU_CAPABILITY_FLAGS "-O3 -mavx")
  ENDIF(MSVC)
ENDIF(CXX_AVX_FOUND)

IF(CXX_AVX2_FOUND)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")
  LIST(APPEND CPU_CAPABILITY_NAMES "AVX2")
  IF(MSVC)
    LIST(APPEND CPU_CAPABILITY_FLAGS "${MSVC_OPT_FLAG}/arch:AVX2")
  ELSE(MSVC)
    LIST(APPEND CPU_CAPABILITY_FLAGS "-O3 -mavx2")
  ENDIF(MSVC)
ENDIF(CXX_AVX2_FOUND)

list(LENGTH CPU_CAPABILITY_NAMES NUM_CPU_CAPABILITY_NAMES)
math(EXPR NUM_CPU_CAPABILITY_NAMES "${NUM_CPU_CAPABILITY_NAMES}-1")

FOREACH(i RANGE ${NUM_CPU_CAPABILITY_NAMES})
  FOREACH(IMPL ${cpu_kernel_cpp_in})
    LIST(GET CPU_CAPABILITY_NAMES ${i} CPU_CAPABILITY)
    SET(NEW_IMPL ${CMAKE_CURRENT_BINARY_DIR}/${IMPL}.${CPU_CAPABILITY}.cpp)
    CONFIGURE_FILE(${IMPL} ${NEW_IMPL} COPYONLY)
    SET(cpu_kernel_cpp ${NEW_IMPL} ${cpu_kernel_cpp}) # Create list of copies
    LIST(GET CPU_CAPABILITY_FLAGS ${i} FLAGS)
    IF(MSVC)
      SET(MACRO_FLAG "/DCPU_CAPABILITY=${CPU_CAPABILITY} /DCPU_CAPABILITY_${CPU_CAPABILITY}")
    ELSE(MSVC)
      SET(MACRO_FLAG "-DCPU_CAPABILITY=${CPU_CAPABILITY} -DCPU_CAPABILITY_${CPU_CAPABILITY}")
    ENDIF(MSVC)
    SET_SOURCE_FILES_PROPERTIES(${NEW_IMPL} PROPERTIES COMPILE_FLAGS "${FLAGS} ${MACRO_FLAG}")
  ENDFOREACH()
ENDFOREACH()

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


IF ($ENV{TH_BINARY_BUILD})
  MESSAGE(STATUS "TH_BINARY_BUILD detected. Statically linking libstdc++")
  SET(CMAKE_CXX_FLAGS "-static-libstdc++ ${CMAKE_CXX_FLAGS}")
ENDIF()

# Can be compiled standalone
IF(NOT AT_INSTALL_BIN_DIR OR NOT AT_INSTALL_LIB_DIR OR NOT AT_INSTALL_INCLUDE_DIR OR NOT AT_INSTALL_SHARE_DIR)
  SET(AT_INSTALL_BIN_DIR "bin" CACHE PATH "AT install binary subdirectory")
  SET(AT_INSTALL_LIB_DIR "lib" CACHE PATH "AT install library subdirectory")
  SET(AT_INSTALL_INCLUDE_DIR "include" CACHE PATH "AT install include subdirectory")
  SET(AT_INSTALL_SHARE_DIR "share" CACHE PATH "AT install include subdirectory")
ENDIF()

# TODO: Maybe put this in the generated files directory
CONFIGURE_FILE(Config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/Config.h")

FILE(GLOB base_h RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
FILE(GLOB base_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
FILE(GLOB native_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "native/*.cpp")
FILE(GLOB native_cudnn_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "native/cudnn/*.cpp")
FILE(GLOB native_cuda_cu RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "native/cuda/*.cu")
FILE(GLOB native_mkl_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "native/mkl/*.cpp")
FILE(GLOB native_mkldnn_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "native/mkldnn/*.cpp")

FILE(GLOB_RECURSE cuda_h
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "cuda/*.cuh" "cuda/*.h" "cudnn/*.cuh" "cudnn/*.h")

FILE(GLOB cudnn_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "cudnn/*.cpp")
FILE(GLOB mkl_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "mkl/*.cpp")
FILE(GLOB mkldnn_cpp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "mkldnn/*.cpp")

FILE(GLOB all_python RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")

FILE(GLOB_RECURSE aten_cuda_cu RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "cuda/*.cu")

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
    COMMAND ${GEN_COMMAND}
      --output-dependencies ${CMAKE_CURRENT_BINARY_DIR}/generated_cpp.txt
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

# Generated headers used from a cuda (.cu) file are
# not tracked correctly in cmake . We make the libATen.so depend explicitly
# on building the generated aten files to workaround.
ADD_CUSTOM_TARGET(aten_files_are_generated
  DEPENDS ${generated_cpp}
)


SET(all_cpp ${base_cpp} ${native_cpp} ${native_cudnn_cpp} ${native_mkl_cpp} ${native_mkldnn_cpp} ${generated_cpp} ${ATen_CPU_SRCS} ${cpu_kernel_cpp})

INCLUDE_DIRECTORIES(${ATen_CPU_INCLUDE})
IF(NOT NO_CUDA)
  INCLUDE_DIRECTORIES(${ATen_CUDA_INCLUDE})
  INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")
  INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/cuda")
  SET(ATen_CUDA_SRCS ${ATen_CUDA_SRCS} ${aten_cuda_cu} ${native_cuda_cu})
  SET(all_cpp ${all_cpp} ${ATen_CUDA_SRCS})
  IF(CUDNN_FOUND)
    SET(all_cpp ${all_cpp} ${cudnn_cpp})
  ENDIF()
  IF(AT_MKL_ENABLED)
    SET(all_cpp ${all_cpp} ${mkl_cpp})
  ENDIF()
endif()

IF(AT_MKLDNN_ENABLED)
  SET(all_cpp ${all_cpp} ${mkldnn_cpp})
ENDIF()

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
ADD_DEPENDENCIES(ATen aten_files_are_generated)

set(TBB_ROOT_DIR "${PROJECT_SOURCE_DIR}/src/ATen/cpu/tbb/tbb_remote")
set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ATen/cpu/tbb)
set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")
target_include_directories(tbb_static PUBLIC ${TBB_ROOT_DIR}/include)
target_link_libraries(ATen tbb_static)

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
ELSE(NOT MSVC)
  IF(AT_MKL_MT)
    set_target_properties(ATen PROPERTIES LINK_FLAGS_RELEASE "/NODEFAULTLIB:${VCOMP_LIB}")
    set_target_properties(ATen PROPERTIES LINK_FLAGS_DEBUG "/NODEFAULTLIB:${VCOMP_LIB}")
    set_target_properties(ATen PROPERTIES STATIC_LIBRARY_FLAGS "/NODEFAULTLIB:${VCOMP_LIB}")
  ENDIF(AT_MKL_MT)
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

if(MKLDNN_FOUND)
  target_link_libraries(ATen ${MKLDNN_LIBRARIES})
endif(MKLDNN_FOUND)

# ---[ Configure cpuinfo
IF(NOT TARGET cpuinfo)
  SET(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY("cpu/cpuinfo")
ENDIF()
TARGET_LINK_LIBRARIES(ATen cpuinfo)

IF(CUDA_FOUND)
  IF ($ENV{ATEN_STATIC_CUDA})
    # CuFFT has a complicated static story (especially around CUDA < 9) because it has device callback support
    # we first have to build a fake lib that links with no device callbacks,
    # and then we link against this object file.
    # This was recommended by the CuFFT team at NVIDIA

    # build fake CuFFT lib in build dir
    EXECUTE_PROCESS(COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/empty_file.cc)
    if(${CUDA_VERSION_MAJOR} EQUAL "8")
      SET(CUFFT_FAKELINK_OPTIONS
	--generate-code arch=compute_35,code=sm_35
	--generate-code arch=compute_50,code=sm_50
	--generate-code arch=compute_60,code=sm_60)
    elseif(${CUDA_VERSION_MAJOR} EQUAL "9")
      SET(CUFFT_FAKELINK_OPTIONS
	--generate-code arch=compute_35,code=sm_35
	--generate-code arch=compute_50,code=sm_50
	--generate-code arch=compute_60,code=sm_60
	--generate-code arch=compute_70,code=sm_70)
    else()
      MESSAGE(FATAL_ERROR "Unhandled major cuda version ${CUDA_VERSION_MAJOR}")
    endif()
    ADD_CUSTOM_COMMAND(
      OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a
      COMMAND "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" -o ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a -Xcompiler -fPIC
      ${CUFFT_FAKELINK_OPTIONS}
      --device-link ${CMAKE_CURRENT_BINARY_DIR}/empty_file.cc -lcufft_static -lculibos
      )
    ADD_CUSTOM_TARGET(FAKELINKED_CUFFT_TARGET DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a)
    add_library(FAKELINKED_CUFFT STATIC IMPORTED GLOBAL)
    add_dependencies(FAKELINKED_CUFFT FAKELINKED_CUFFT_TARGET)
    set_target_properties(FAKELINKED_CUFFT PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cufft_static_library.a)

    TARGET_LINK_LIBRARIES(ATen
      ${CUDA_LIBRARIES}
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcusparse_static.a
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a
      FAKELINKED_CUFFT
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft_static.a
      )
  ELSE()
    TARGET_LINK_LIBRARIES(ATen
      ${CUDA_LIBRARIES}
      ${CUDA_cusparse_LIBRARY}
      ${CUDA_curand_LIBRARY})
    CUDA_ADD_CUBLAS_TO_TARGET(ATen)
    CUDA_ADD_CUFFT_TO_TARGET(ATen)
  ENDIF()

  if(CUDNN_FOUND)
    target_link_libraries(ATen ${CUDNN_LIBRARIES})
  endif(CUDNN_FOUND)

  IF(USE_MAGMA)
    TARGET_LINK_LIBRARIES(ATen ${MAGMA_LIBRARIES})
    IF ($ENV{TH_BINARY_BUILD})
      TARGET_LINK_LIBRARIES(ATen "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
    ENDIF($ENV{TH_BINARY_BUILD})
  ENDIF(USE_MAGMA)
  IF ($ENV{ATEN_STATIC_CUDA})
    TARGET_LINK_LIBRARIES(ATen "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a")
  ENDIF($ENV{ATEN_STATIC_CUDA})
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
FOREACH(HEADER ${cuda_h})
  # https://stackoverflow.com/questions/11096471/how-can-i-install-a-hierarchy-of-files-using-cmake
  GET_FILENAME_COMPONENT(DIR ${HEADER} DIRECTORY)
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/${DIR})
ENDFOREACH()
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)
