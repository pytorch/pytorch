# Prints accumulated Caffe2 configuration summary
function (Caffe2_print_configuration_summary)

  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                    OUTPUT_VARIABLE Caffe2_GIT_VERSION
                    RESULT_VARIABLE __git_result)
    if(NOT ${__git_result} EQUAL 0)
      set(Caffe2_GIT_VERSION "unknown")
    endif()
  endif()

  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  Git version           : ${Caffe2_GIT_VERSION}")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  Protobuf compiler     : ${PROTOBUF_PROTOC_EXECUTABLE}")
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
  message(STATUS "  Compile definitions   : ${tmp}")
  message(STATUS "")
  message(STATUS "  BUILD_SHARED_LIBS     : ${BUILD_SHARED_LIBS}")
  message(STATUS "  BUILD_PYTHON          : ${BUILD_PYTHON}")
  message(STATUS "    Python version      : ${PYTHONLIBS_VERSION_STRING}")
  message(STATUS "    Python library      : ${PYTHON_LIBRARIES}")
  message(STATUS "  BUILD_TEST            : ${BUILD_TEST}")

  message(STATUS "  USE_CUDA              : ${USE_CUDA}")
  if(${USE_CUDA})
  message(STATUS "    CUDA version        : ${CUDA_VERSION}")
  message(STATUS "  USE_CNMEM             : ${USE_CNMEM}")
  endif()

  message(STATUS "  USE_NERVANA_GPU       : ${USE_NERVANA_GPU}")
  if(${USE_NERVANA_GPU})
  message(STATUS "    NERVANA_GPU version : ${NERVANA_GPU_VERSION}")
  endif()

  message(STATUS "  USE_GLOG              : ${USE_GLOG}")

  message(STATUS "  USE_GFLAGS            : ${USE_GFLAGS}")

  message(STATUS "  USE_LMDB              : ${USE_LMDB}")
  if(${USE_LMDB})
  message(STATUS "    LMDB version        : ${LMDB_VERSION}")
  endif()

  message(STATUS "  USE_LEVELDB           : ${USE_LEVELDB}")
  if(${USE_LEVELDB})
  message(STATUS "    LevelDB version     : ${LEVELDB_VERSION}")
  message(STATUS "    Snappy version      : ${Snappy_VERSION}")
  endif()

  message(STATUS "  USE_OPENCV            : ${USE_OPENCV}")
  if(${USE_OPENCV})
  message(STATUS "    OpenCV version      : ${OpenCV_VERSION}")
  endif()

  message(STATUS "  USE_FFMPEG            : ${USE_FFMPEG}")

  message(STATUS "  USE_ZMQ               : ${USE_ZMQ}")
  message(STATUS "  USE_ROCKSDB           : ${USE_ROCKSDB}")
  message(STATUS "  USE_MPI               : ${USE_MPI}")
  message(STATUS "  USE_NCCL              : ${USE_NCCL}")
  message(STATUS "  USE_NNPACK            : ${USE_NNPACK}")
  message(STATUS "  USE_OPENMP            : ${USE_OPENMP}")
  message(STATUS "  USE_REDIS             : ${USE_REDIS}")
  message(STATUS "  USE_GLOO              : ${USE_GLOO}")

endfunction()
