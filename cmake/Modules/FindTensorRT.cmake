# Find the TensorRT package
#
# The following variables are optionally searched for defaults
#  TENSORRT_ROOT_DIR: Base directory where all TensorRT components are found
#
# The following are set after configuration is done:
#  TENSORRT_FOUND
#  TENSORRT_INCLUDE_DIRS
#  TENSORRT_LIBRARIES

message("####################################### ${TENSORRT_ROOT_DIR}")
set(_TensorRT_SEARCHES)

if(TENSORRT_ROOT_DIR)
  set(_TensorRT_SEARCH_ROOT PATHS ${TENSORRT_ROOT_DIR} NO_DEFAULT_PATH)
  list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_ROOT)
endif()

## appends some common paths
#set(_TensorRT_SEARCH_NORMAL
#        PATHS "/usr"
#        )
##list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_NORMAL)
#list(APPEND _TensorRT_SEARCH_NORMAL)

# Include dir
foreach(search ${_TensorRT_SEARCHES})
  message("####################################### ${search}")

  find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h ${${search}} PATH_SUFFIXES include)
endforeach()

find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)
message("--------------------------------------------------- ${TENSORRT_INCLUDE_DIR}")



#find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h
#        HINTS ${TENSORRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
#        PATH_SUFFIXES ${CMAKE_LIBRARY_ARCHITECTURE})


find_library(TENSORRT_LIBRARY_1 nvinfer
        HINTS ${TENSORRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

message("--------------------------------------------------- ${TENSORRT_LIBRARY_1}")


find_library(TENSORRT_LIBRARY_2 nvcaffe_parser
        HINTS ${TENSORRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_3 nvinfer_plugin
        HINTS ${TENSORRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_4 nvonnxparser
        HINTS ${TENSORRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_5 nvonnxparser_runtime
        HINTS ${TENSORRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_6 nvparsers
        HINTS ${TENSORRT_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)

#set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_1})
#list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_1})
#list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_2})
#list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_3})
#list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_4})
#list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_5})
#list(APPEND TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_6})

set(TENSORRT_LIBRARIES
  ${TENSORRT_LIBRARY_1}
  ${TENSORRT_LIBRARY_2}
  ${NSTEORRT_LIBRARY_3}
  ${TENSORRT_LIBRARY_4}
  ${TENSORRT_LIBRARY_5}
  ${TENSORRT_LIBRARY_6}
)

set(TENSORRT_FOUND true BOOL)

#include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(TensorRT DEFAULT_MSG ${TENSORRT_LIBRARY_1})

if(TENSORRT_FOUND)
  mark_as_advanced(${TENSORRT_INCLUDE_DIR} ${TENSORRT_LIBRARIES})
  message(STATUS "Found TENSORRT (include: ${TENSORRT_INCLUDE_DIR}, libraries: ${TENSORRT_LIBRARIES})")
endif(TENSORRT_FOUND)
