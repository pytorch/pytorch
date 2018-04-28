CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
CMAKE_POLICY(VERSION 3.0)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/cmake/FindCUDA")

### This library exports: 
ADD_LIBRARY(autogradpp STATIC 
  src/detail.cpp
  src/containers.cpp
  src/optimizers.cpp
)

## Build subdirectories
ADD_SUBDIRECTORY(tests)

### ATen
FIND_PATH(ATEN_INCLUDE_DIR ATen/ATen.h PATHS "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/torch/lib/include/" NO_DEFAULT_PATH)
FIND_LIBRARY(ATEN_LIBRARY ATen PATHS "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/torch/lib" NO_DEFAULT_PATH)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ATEN DEFAULT_MSG ATEN_LIBRARY ATEN_INCLUDE_DIR)
MARK_AS_ADVANCED(ATEN_INCLUDE_DIR ATEN_LIBRARY)

SET(ATEN_INCLUDE_DIRS ${ATEN_INCLUDE_DIR})
SET(ATEN_LIBRARIES ${ATEN_LIBRARY})

ADD_LIBRARY(ATen SHARED IMPORTED)
SET_TARGET_PROPERTIES(ATen PROPERTIES
  IMPORTED_LOCATION ${ATEN_LIBRARY}
  INTERFACE_INCLUDE_DIRECTORIES ${ATEN_INCLUDE_DIR})

### Python Headers for pytorch
FIND_PACKAGE(PythonLibs REQUIRED)
TARGET_INCLUDE_DIRECTORIES(autogradpp 
  PUBLIC 
    ${PYTHON_INCLUDE_DIRS} 
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytorch"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/torch/lib/include/TH"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/torch/lib/include/THC"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/autogradpp"
)

### Copy over all the relevant .so files
ADD_CUSTOM_TARGET(gen_libpythpp_so DEPENDS "${CMAKE_BINARY_DIR}/libpythpp.so")
ADD_CUSTOM_COMMAND(
  OUTPUT "${CMAKE_BINARY_DIR}/libpythpp.so"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/build/lib*/torch/_C*.so" "${CMAKE_BINARY_DIR}/libpythpp.so"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/lib"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/build/lib*/torch/lib/libshm.*" "${CMAKE_BINARY_DIR}/lib"
)
IF(NOT APPLE)
  ADD_CUSTOM_COMMAND(
    TARGET gen_libpythpp_so POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/build/lib*/torch/lib/libgloo*" "${CMAKE_BINARY_DIR}/lib"
  )
ENDIF(NOT APPLE)
ADD_LIBRARY(libpythpp SHARED IMPORTED)
ADD_DEPENDENCIES(libpythpp gen_libpythpp_so)
SET_TARGET_PROPERTIES(libpythpp PROPERTIES IMPORTED_NO_SONAME 1
  IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libpythpp.so")

IF(APPLE)
  # Set RPATHs for OSX...
  ADD_CUSTOM_COMMAND(
    TARGET gen_libpythpp_so POST_BUILD
    COMMAND install_name_tool -id "@rpath/libpythpp.so" -add_rpath "${CMAKE_BINARY_DIR}" "${CMAKE_BINARY_DIR}/libpythpp.so"
  )
  # Unfortunately this won't be picked up in the initial build...
  SET_TARGET_PROPERTIES(libpythpp PROPERTIES
    INTERFACE_LINK_LIBRARIES "-Wl,-rpath ${CMAKE_BINARY_DIR}")
ELSE(APPLE)
  ADD_LIBRARY(libgloo SHARED IMPORTED)
  SET_TARGET_PROPERTIES(libgloo PROPERTIES IMPORTED_NO_SONAME 1
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/torch/lib/libgloo.so")

  TARGET_LINK_LIBRARIES(autogradpp libgloo)
ENDIF(APPLE)

### Link stuff
TARGET_LINK_LIBRARIES(autogradpp ATen libpythpp ${PYTHON_LIBRARIES})

### Optionally, link CUDA
FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
  TARGET_INCLUDE_DIRECTORIES(autogradpp PUBLIC ${CUDA_TOOLKIT_INCLUDE})
  ADD_CUSTOM_TARGET(gen_libpythpp_nvrtc_so DEPENDS "${CMAKE_BINARY_DIR}/libpythpp_nvrtc.so")
  ADD_CUSTOM_COMMAND(
    OUTPUT "${CMAKE_BINARY_DIR}/libpythpp_nvrtc.so"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/build/lib*/torch/_nvrtc*.so" "${CMAKE_BINARY_DIR}/libpythpp_nvrtc.so"
  )

  ADD_LIBRARY(libpythpp_nvrtc SHARED IMPORTED)
  ADD_DEPENDENCIES(libpythpp_nvrtc gen_libpythpp_nvrtc_so)
  SET_TARGET_PROPERTIES(libpythpp_nvrtc PROPERTIES IMPORTED_NO_SONAME 1
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libpythpp_nvrtc.so")

  IF(APPLE)
    ADD_CUSTOM_COMMAND(
      TARGET gen_libpythpp_nvrtc_so POST_BUILD
      COMMAND install_name_tool -id "@rpath/libpythpp_nvrtc.so" -add_rpath "${CMAKE_BINARY_DIR}" "${CMAKE_BINARY_DIR}/libpythpp_nvrtc.so"
    )
  ELSE(APPLE)
    ADD_LIBRARY(libgloo_cuda SHARED IMPORTED)
    SET_TARGET_PROPERTIES(libgloo_cuda PROPERTIES IMPORTED_NO_SONAME 1
      IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/torch/lib/libgloo_cuda.so")

    TARGET_LINK_LIBRARIES(autogradpp libgloo_cuda)
  ENDIF(APPLE)
  TARGET_LINK_LIBRARIES(autogradpp libpythpp_nvrtc
    -L${CUDA_TOOLKIT_ROOT_DIR}/lib64 cuda cudart nvrtc
   )
ENDIF(CUDA_FOUND)
