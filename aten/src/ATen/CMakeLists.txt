
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

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
  target_link_libraries(ATen PRIVATE THC)
ELSE()
  ADD_LIBRARY(ATen ${AT_LINK_STYLE} ${all_cpp})
ENDIF()
ADD_DEPENDENCIES(ATen aten_files_are_generated)
target_link_libraries(ATen PRIVATE TH)

set(TBB_ROOT_DIR "${PROJECT_SOURCE_DIR}/src/ATen/cpu/tbb/tbb_remote")
set(TBB_BUILD_STATIC ON CACHE BOOL " " FORCE)
set(TBB_BUILD_SHARED OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL " " FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/ATen/cpu/tbb)
set_property(TARGET tbb_static tbb_def_files PROPERTY FOLDER "dependencies")
target_include_directories(tbb_static PUBLIC ${TBB_ROOT_DIR}/include)
# Blegh.  This should be PRIVATE but some code directly depends on tbb headers
# being brought into scope by ATen.  Fix that, then make this PRIVATE.
target_link_libraries(ATen PUBLIC tbb_static)

SET_TARGET_PROPERTIES(ATen PROPERTIES VERSION 1 SOVERSION 1)

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")
    SET_PROPERTY(TARGET ATen PROPERTY CXX_STANDARD 11)
endif(NOT ${CMAKE_VERSION} VERSION_LESS "3.1")

IF(BLAS_FOUND)
  IF ($ENV{TH_BINARY_BUILD})
    MESSAGE(STATUS "TH_BINARY_BUILD detected. Enabling special linkage.")
    target_link_libraries(ATen PRIVATE "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
  ELSE ($ENV{TH_BINARY_BUILD})
    target_link_libraries(ATen PUBLIC ${BLAS_LIBRARIES})
  ENDIF ($ENV{TH_BINARY_BUILD})
ENDIF(BLAS_FOUND)

IF(LAPACK_FOUND)
  target_link_libraries(ATen PUBLIC ${LAPACK_LIBRARIES})
ENDIF(LAPACK_FOUND)

if(NEED_LIBRT)
  target_link_libraries(ATen PUBLIC rt)
endif()

IF(NOT MSVC)
  target_link_libraries(ATen PUBLIC m)
ENDIF(NOT MSVC)

if(MKLDNN_FOUND)
  target_link_libraries(ATen PUBLIC ${MKLDNN_LIBRARIES})
endif(MKLDNN_FOUND)

# ---[ Configure cpuinfo
IF(NOT TARGET cpuinfo)
  SET(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  SET(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  ADD_SUBDIRECTORY("cpu/cpuinfo")
ENDIF()
target_link_libraries(ATen PRIVATE cpuinfo)

IF(CUDA_FOUND)
  target_link_libraries(ATen
    PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_curand_LIBRARY})
  CUDA_ADD_CUBLAS_TO_TARGET(ATen)
  CUDA_ADD_CUFFT_TO_TARGET(ATen)

  if(CUDNN_FOUND)
    target_link_libraries(ATen PUBLIC ${CUDNN_LIBRARIES})
  endif(CUDNN_FOUND)

  IF(USE_MAGMA)
    target_link_libraries(ATen PRIVATE ${MAGMA_LIBRARIES})
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
        target_link_libraries(ATen PRIVATE "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
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
FOREACH(HEADER ${cuda_h})
  # https://stackoverflow.com/questions/11096471/how-can-i-install-a-hierarchy-of-files-using-cmake
  GET_FILENAME_COMPONENT(DIR ${HEADER} DIRECTORY)
  INSTALL(FILES ${HEADER} DESTINATION ${AT_INSTALL_INCLUDE_DIR}/ATen/${DIR})
ENDFOREACH()
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/ATen/Declarations.yaml
  DESTINATION ${AT_INSTALL_SHARE_DIR}/ATen)
