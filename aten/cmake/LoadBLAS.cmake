if(WIN32 AND NOT CYGWIN)
  set(BLAS_INSTALL_LIBRARIES "OFF"
    CACHE BOOL "Copy the required BLAS DLLs into the TH install dirs")
endif()

find_package(BLAS)
set(AT_MKL_ENABLED 0)
if(BLAS_FOUND)
  set(USE_BLAS 1)
  if(BLAS_INFO STREQUAL "mkl")
    add_definitions(-DTH_BLAS_MKL)
    if(NOT BLAS_INCLUDE_DIR)
      MESSAGE(FATAL_ERROR "MKL is used, but MKL header files are not found. \
        You can get them by `conda install mkl-include` if using conda (if \
        it is missing, run `conda upgrade -n root conda` first), and \
        `pip install mkl-devel` if using pip. If build fails with header files \
        available in the system, please make sure that CMake will search the \
        directory containing them, e.g., by setting CMAKE_INCLUDE_PATH.")
    endif()
    include_directories(${BLAS_INCLUDE_DIR})  # include MKL headers
    set(AT_MKL_ENABLED 1)
  endif()
endif(BLAS_FOUND)
