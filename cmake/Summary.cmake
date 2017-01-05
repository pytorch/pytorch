
# Prints accumulated Caffe2 configuration summary
function (Caffe2_print_configuration_summary)

  find_package(Git)
  if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                    OUTPUT_VARIABLE Caffe_GIT_VERSION
                    RESULT_VARIABLE __git_result)
    if(NOT ${__git_result} EQUAL 0)
      set(Caffe_GIT_VERSION "unknown")
    endif()
  endif()

  message(STATUS "")
  message(STATUS "******** Summary ********")
  message(STATUS "General:")
  message(STATUS "  Version               : ${Caffe2_VERSION}")
  message(STATUS "  Git                   : ${Caffe2_GIT_VERSION}")
  message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
  message(STATUS "")
  message(STATUS "  BUILD_SHARED_LIBS     : ${BUILD_SHARED_LIBS}")

  message(STATUS "  USE_CUDA              : ${USE_CUDA}")
  if(${USE_CUDA})
  message(STATUS "    CUDA version        : ${CUDA_VERSION}")
  endif()

  message(STATUS "  USE_NERVANA_GPU       : ${USE_NERVANA_GPU}")
  if(${USE_NERVANA_GPU})
  message(STATUS "    NERVANA_GPU version : ${NERVANA_GPU_VERSION}")
  endif()

  message(STATUS "  USE_GLOG              : ${USE_GLOG}")
  if(${USE_GLOG})
  message(STATUS "    glog version        : ${GLOG_VERSION}")
  endif()

  message(STATUS "  USE_GFLAGS            : ${USE_GFLAGS}")
  if(${USE_GFLAGS})
  message(STATUS "    gflags version      : ${GFLAGS_VERSION}")
  endif()

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

  message(STATUS "  USE_ZMQDB             : ${USE_ZMQDB}")
  message(STATUS "  USE_ROCKSDB           : ${USE_ROCKSDB}")
  message(STATUS "  USE_MPI               : ${USE_MPI}")
  message(STATUS "  USE_OPENMP            : ${USE_OPENMP}")

endfunction()
