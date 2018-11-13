 # UBSAN triggers when compiling protobuf, so we need to disable it.
set(UBSAN_FLAG "-fsanitize=undefined")

macro(disable_ubsan)
  if (CMAKE_C_FLAGS MATCHES ${UBSAN_FLAG} OR CMAKE_CXX_FLAGS MATCHES ${UBSAN_FLAG})
    set(CAFFE2_UBSAN_ENABLED ON)
    string(REPLACE ${UBSAN_FLAG} "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE ${UBSAN_FLAG} "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  endif()
endmacro()

macro(enable_ubsan)
  if (CAFFE2_UBSAN_ENABLED)
    set(CMAKE_C_FLAGS "${UBSAN_FLAG} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${UBSAN_FLAG} ${CMAKE_CXX_FLAGS}")
  endif()
endmacro()

if(NOT BUILD_ATEN_ONLY)
# ---[ Custom Protobuf
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  disable_ubsan()
  include(${CMAKE_CURRENT_LIST_DIR}/ProtoBuf.cmake)
  enable_ubsan()
endif()
endif()

# ---[ Threads
include(${CMAKE_CURRENT_LIST_DIR}/public/threads.cmake)
if (TARGET Threads::Threads)
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS Threads::Threads)
else()
  message(FATAL_ERROR
      "Cannot find threading library. Caffe2 requires Threads to compile.")
endif()

# ---[ protobuf
if(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if(USE_LITE_PROTO)
    set(CAFFE2_USE_LITE_PROTO 1)
  endif()
endif()

# ---[ BLAS
if(NOT BUILD_ATEN_MOBILE)
  set(BLAS "MKL" CACHE STRING "Selected BLAS library")
else()
  set(BLAS "Eigen" CACHE STRING "Selected BLAS library")
endif()
set_property(CACHE BLAS PROPERTY STRINGS "Eigen;ATLAS;OpenBLAS;MKL;vecLib")
message(STATUS "The BLAS backend of choice:" ${BLAS})

if(BLAS STREQUAL "Eigen")
  # Eigen is header-only and we do not have any dependent libraries
  set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
elseif(BLAS STREQUAL "ATLAS")
  find_package(Atlas REQUIRED)
  include_directories(SYSTEM ${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${ATLAS_LIBRARIES})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS cblas)
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "MKL")
  if(BLAS_SET_BY_USER)
    find_package(MKL REQUIRED)
  else()
    find_package(MKL QUIET)
  endif()
  include(${CMAKE_CURRENT_LIST_DIR}/public/mkl.cmake)
  if(MKL_FOUND)
    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::mkl)
    set(CAFFE2_USE_MKL ON)
  else()
    message(WARNING "MKL could not be found. Defaulting to Eigen")
    set(BLAS "Eigen" CACHE STRING "Selected BLAS library")
    set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
    set(CAFFE2_USE_MKL OFF)
  endif()
elseif(BLAS STREQUAL "vecLib")
  find_package(vecLib REQUIRED)
  include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${vecLib_LINKER_LIBS})
else()
  message(FATAL_ERROR "Unrecognized blas option:" ${BLAS})
endif()


if (NOT BUILD_ATEN_MOBILE)
  set(AT_MKL_ENABLED 0)
  set(AT_MKL_MT 0)
  set(USE_BLAS 1)
  if(NOT (ATLAS_FOUND OR OPENBLAS_FOUND OR MKL_FOUND OR VECLIB_FOUND))
    find_package(BLAS)
    if (NOT BLAS_FOUND)
      set(USE_BLAS 0)
    endif()
  endif()

  if (MKL_FOUND)
    ADD_DEFINITIONS(-DTH_BLAS_MKL)
    if(NOT MKL_INCLUDE_DIR)
      MESSAGE(FATAL_ERROR "MKL is used, but MKL header files are not found. \
        You can get them by `conda install mkl-include` if using conda (if \
        it is missing, run `conda upgrade -n root conda` first), and \
        `pip install mkl-devel` if using pip. If build fails with header files \
        available in the system, please make sure that CMake will search the \
        directory containing them, e.g., by setting CMAKE_INCLUDE_PATH.")
	endif()
    if (MSVC AND MKL_LIBRARIES MATCHES ".*libiomp5md\\.lib.*")
      ADD_DEFINITIONS(-D_OPENMP_NOFORCE_MANIFEST)
      set(AT_MKL_MT 1)
    endif()
    set(AT_MKL_ENABLED 1)
  endif()
endif()

# Directory where NNPACK and cpuinfo will download and build all dependencies
set(CONFU_DEPENDENCIES_SOURCE_DIR ${PROJECT_BINARY_DIR}/confu-srcs
  CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${PROJECT_BINARY_DIR}/confu-deps
  CACHE PATH "Confu-style dependencies binary directory")

# ---[ QNNPACK
if(USE_QNNPACK)
  if (NOT IOS AND NOT (CMAKE_SYSTEM_NAME MATCHES "^(Android|Linux|Darwin)$"))
    message(WARNING
      "Target platform \"${CMAKE_SYSTEM_NAME}\" is not supported in QNNPACK. "
      "Supported platforms are Android, iOS, Linux, and macOS. "
      "Turn this warning off by USE_QNNPACK=OFF.")
    set(USE_QNNPACK OFF)
  endif()
  if (NOT IOS AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|AMD64|x86_64|armv[0-9].*|arm64|aarch64)$"))
    message(WARNING
      "Target architecture \"${CMAKE_SYSTEM_PROCESSOR}\" is not supported in QNNPACK. "
      "Supported platforms are x86, x86-64, ARM, and ARM64. "
      "Turn this warning off by USE_QNNPACK=OFF.")
    set(USE_QNNPACK OFF)
  endif()
  if (USE_QNNPACK)
    set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")

    # Directories for QNNPACK dependencies submoduled in Caffe2
    if (NOT DEFINED CPUINFO_SOURCE_DIR)
      set(CPUINFO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/cpuinfo" CACHE STRING "cpuinfo source directory")
    endif()
    if (NOT DEFINED QNNPACK_SOURCE_DIR)
      set(QNNPACK_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/QNNPACK" CACHE STRING "QNNPACK source directory")
    endif()
    if (NOT DEFINED FP16_SOURCE_DIR)
      set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
    endif()
    if (NOT DEFINED FXDIV_SOURCE_DIR)
      set(FXDIV_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FXdiv" CACHE STRING "FXdiv source directory")
    endif()
    if (NOT DEFINED PSIMD_SOURCE_DIR)
      set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
    endif()
    if (NOT DEFINED PTHREADPOOL_SOURCE_DIR)
      set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")
    endif()

    if(NOT TARGET qnnpack)
      set(QNNPACK_BUILD_TESTS OFF CACHE BOOL "")
      set(QNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
      set(QNNPACK_CUSTOM_THREADPOOL ON CACHE BOOL "")
      set(QNNPACK_LIBRARY_TYPE "static" CACHE STRING "")
      set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
      set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
      add_subdirectory(
        "${QNNPACK_SOURCE_DIR}"
        "${CONFU_DEPENDENCIES_BINARY_DIR}/QNNPACK")
      # We build static versions of QNNPACK and pthreadpool but link
      # them into a shared library for Caffe2, so they need PIC.
      set_property(TARGET qnnpack PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET pthreadpool PROPERTY POSITION_INDEPENDENT_CODE ON)
      set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()

    list(APPEND Caffe2_DEPENDENCY_LIBS qnnpack)
  endif()
endif()

# ---[ Caffe2 Int8 operators (enabled by USE_QNNPACK) depend on gemmlowp and neon2sse headers
if(USE_QNNPACK)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  include_directories(SYSTEM "${CAFFE2_THIRD_PARTY_ROOT}/gemmlowp")
  include_directories(SYSTEM "${CAFFE2_THIRD_PARTY_ROOT}/neon2sse")
endif()

# ---[ NNPACK
if(USE_NNPACK)
  include(${CMAKE_CURRENT_LIST_DIR}/External/nnpack.cmake)
  if(NNPACK_FOUND)
    if(TARGET nnpack)
      # ---[ NNPACK is being built together with Caffe2: explicitly specify dependency
      list(APPEND Caffe2_DEPENDENCY_LIBS nnpack)
    else()
      include_directories(SYSTEM ${NNPACK_INCLUDE_DIRS})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${NNPACK_LIBRARIES})
    endif()
  else()
    message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
    caffe2_update_option(USE_NNPACK OFF)
  endif()
endif()

# ---[ Caffe2 uses cpuinfo library in the thread pool
if (NOT TARGET cpuinfo)
  if (NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/cpuinfo" CACHE STRING "cpuinfo source directory")
  endif()

  set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "")
  set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
  if(MSVC)
    if (CAFFE2_USE_MSVC_STATIC_RUNTIME)
      set(CPUINFO_RUNTIME_TYPE "static" CACHE STRING "")
    else()
      set(CPUINFO_RUNTIME_TYPE "shared" CACHE STRING "")
    endif()
  endif()
  add_subdirectory(
    "${CPUINFO_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/cpuinfo")
  # We build static version of cpuinfo but link
  # them into a shared library for Caffe2, so they need PIC.
  set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS cpuinfo)

# ---[ gflags
if(USE_GFLAGS)
  include(${CMAKE_CURRENT_LIST_DIR}/public/gflags.cmake)
  if (NOT TARGET gflags)
    message(WARNING
        "gflags is not found. Caffe2 will build without gflags support but "
        "it is strongly recommended that you install gflags. Suppress this "
        "warning with -DUSE_GFLAGS=OFF")
    caffe2_update_option(USE_GFLAGS OFF)
  endif()
endif()

# ---[ Google-glog
if(USE_GLOG)
  include(${CMAKE_CURRENT_LIST_DIR}/public/glog.cmake)
  if (TARGET glog::glog)
    set(CAFFE2_USE_GOOGLE_GLOG 1)
    include_directories(SYSTEM ${GLOG_INCLUDE_DIR})
    list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS glog::glog)
  else()
    message(WARNING
        "glog is not found. Caffe2 will build without glog support but it is "
        "strongly recommended that you install glog. Suppress this warning "
        "with -DUSE_GLOG=OFF")
    caffe2_update_option(USE_GLOG OFF)
  endif()
endif()


# ---[ Googletest and benchmark
if(BUILD_TEST)
  # Preserve build options.
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)

  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(BUILD_GTEST ON CACHE BOOL "Build gtest" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "Install gtest." FORCE)
  # We currently don't need gmock right now.
  set(BUILD_GMOCK OFF CACHE BOOL "Build gmock." FORCE)
  # For Windows, we will check the runtime used is correctly passed in.
  if (NOT CAFFE2_USE_MSVC_STATIC_RUNTIME)
      set(gtest_force_shared_crt ON CACHE BOOL "force shared crt on gtest" FORCE)
  endif()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest)
  include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/googletest/googletest/include)

  # We will not need to test benchmark lib itself.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # We will not need to install benchmark since we link it statically.
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../third_party/benchmark/include)

  # Recover build options.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libs" FORCE)
endif()

# ---[ FBGEMM
if(USE_FBGEMM)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  if(NOT DEFINED FBGEMM_SOURCE_DIR)
    set(FBGEMM_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/fbgemm" CACHE STRING "FBGEMM source directory")
  endif()
  if(NOT CAFFE2_COMPILER_SUPPORTS_AVX512F_EXTENSIONS)
    message(WARNING
      "A compiler with AVX512 support is required for FBGEMM. "
      "Not compiling with FBGEMM. "
      "Turn this warning off by USE_FBGEMM=OFF.")
    set(USE_FBGEMM OFF)
  endif()
  if(MSVC)
    set(USE_FBGEMM OFF)
  endif()
  if(USE_FBGEMM AND NOT TARGET fbgemm)
    set(FBGEMM_BUILD_TESTS OFF CACHE BOOL "")
    set(FBGEMM_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(FBGEMM_LIBRARY_TYPE "static" CACHE STRING "")
    add_subdirectory("${FBGEMM_SOURCE_DIR}")
    set_property(TARGET fbgemm_avx2 PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET fbgemm_avx512 PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET fbgemm PROPERTY POSITION_INDEPENDENT_CODE ON)
  endif()

  if(USE_FBGEMM)
    list(APPEND Caffe2_DEPENDENCY_LIBS fbgemm)
  endif()
endif()

if(USE_FBGEMM)
  set(CAFFE2_THIRD_PARTY_ROOT "${PROJECT_SOURCE_DIR}/third_party")
  include_directories(SYSTEM "${CAFFE2_THIRD_PARTY_ROOT}")
endif()


# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB)
  if(LMDB_FOUND)
    include_directories(SYSTEM ${LMDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LMDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with LMDB. Suppress this warning with -DUSE_LMDB=OFF")
    caffe2_update_option(USE_LMDB OFF)
  endif()
endif()

if (USE_OPENCL)
  message(INFO "USING OPENCL")
  find_package(OpenCL REQUIRED)
  include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS})
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../caffe2/contrib/opencl)
  list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCL_LIBRARIES})
endif()

# ---[ LevelDB
# ---[ Snappy
if(USE_LEVELDB)
  find_package(LevelDB)
  find_package(Snappy)
  if(LEVELDB_FOUND AND SNAPPY_FOUND)
    include_directories(SYSTEM ${LevelDB_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LevelDB_LIBRARIES})
    include_directories(SYSTEM ${Snappy_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Snappy_LIBRARIES})
  else()
    message(WARNING "Not compiling with LevelDB. Suppress this warning with -DUSE_LEVELDB=OFF")
    caffe2_update_option(USE_LEVELDB OFF)
  endif()
endif()

# ---[ NUMA
if(USE_NUMA)
  if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "NUMA is currently only supported under Linux.")
    caffe2_update_option(USE_NUMA OFF)
  else()
    find_package(Numa)
    if(NUMA_FOUND)
      include_directories(SYSTEM ${Numa_INCLUDE_DIR})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${Numa_LIBRARIES})
    else()
      message(WARNING "Not compiling with NUMA. Suppress this warning with -DUSE_NUMA=OFF")
      caffe2_update_option(USE_NUMA OFF)
    endif()
  endif()
endif()

# ---[ ZMQ
if(USE_ZMQ)
  find_package(ZMQ)
  if(ZMQ_FOUND)
    include_directories(SYSTEM ${ZMQ_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${ZMQ_LIBRARIES})
  else()
    message(WARNING "Not compiling with ZMQ. Suppress this warning with -DUSE_ZMQ=OFF")
    caffe2_update_option(USE_ZMQ OFF)
  endif()
endif()

# ---[ Redis
if(USE_REDIS)
  find_package(Hiredis)
  if(HIREDIS_FOUND)
    include_directories(SYSTEM ${Hiredis_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Hiredis_LIBRARIES})
  else()
    message(WARNING "Not compiling with Redis. Suppress this warning with -DUSE_REDIS=OFF")
    caffe2_update_option(USE_REDIS OFF)
  endif()
endif()


# ---[ OpenCV
if(USE_OPENCV)
  # OpenCV 3
  find_package(OpenCV 3 QUIET COMPONENTS core highgui imgproc imgcodecs videoio video)
  if(NOT OpenCV_FOUND)
    # OpenCV 2
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
  endif()
  if(OpenCV_FOUND)
    include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  else()
    message(WARNING "Not compiling with OpenCV. Suppress this warning with -DUSE_OPENCV=OFF")
    caffe2_update_option(USE_OPENCV OFF)
  endif()
endif()

# ---[ FFMPEG
if(USE_FFMPEG)
  find_package(FFmpeg REQUIRED)
  if (FFMPEG_FOUND)
    message("Found FFMPEG/LibAV libraries")
    include_directories(SYSTEM ${FFMPEG_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${FFMPEG_LIBRARIES})
  else ()
    message("Not compiling with FFmpeg. Suppress this warning with -DUSE_FFMPEG=OFF")
    caffe2_update_option(USE_FFMPEG OFF)
  endif ()
endif()

# ---[ Caffe2 depends on FP16 library for half-precision conversions
if (NOT TARGET fp16)
  if (NOT DEFINED FP16_SOURCE_DIR)
    set(FP16_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/FP16" CACHE STRING "FP16 source directory")
  endif()

  set(FP16_BUILD_TESTS OFF CACHE BOOL "")
  set(FP16_BUILD_BENCHMARKS OFF CACHE BOOL "")
  add_subdirectory(
    "${FP16_SOURCE_DIR}"
    "${CONFU_DEPENDENCIES_BINARY_DIR}/FP16")
endif()
list(APPEND Caffe2_DEPENDENCY_LIBS fp16)

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
if (USE_SYSTEM_EIGEN_INSTALL)
  find_package(Eigen3)
  if(EIGEN3_FOUND)
    message(STATUS "Found system Eigen at " ${EIGEN3_INCLUDE_DIR})
  else()
    message(STATUS "Did not find system Eigen. Using third party subdirectory.")
    set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
    caffe2_update_option(USE_SYSTEM_EIGEN_INSTALL OFF)
  endif()
else()
  message(STATUS "Using third party subdirectory Eigen.")
  set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen)
endif()
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# ---[ Python + Numpy
if(BUILD_PYTHON)
  # If not given a Python installation, then use the current active Python
  if(NOT DEFINED PYTHON_EXECUTABLE)
    execute_process(
      COMMAND "which" "python" RESULT_VARIABLE _exitcode OUTPUT_VARIABLE _py_exe)
    if(${_exitcode} EQUAL 0)
      if (NOT MSVC)
        string(STRIP ${_py_exe} PYTHON_EXECUTABLE)
      endif()
      message(STATUS "Setting Python to ${PYTHON_EXECUTABLE}")
    endif()
  endif()

  # Check that Python works
  if(DEFINED PYTHON_EXECUTABLE)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" "--version"
        RESULT_VARIABLE _exitcode)
    if(NOT ${_exitcode} EQUAL 0)
      message(FATAL_ERROR "The Python executable ${PYTHON_EXECUTABLE} cannot be run. Make sure that it is an absolute path.")
    endif()
  endif()

  # Seed PYTHON_INCLUDE_DIR and PYTHON_LIBRARY to be consistent with the
  # executable that we already found (if we didn't actually find an executable
  # then these will just use "python", but at least they'll be consistent with
  # each other).
  if(NOT DEFINED PYTHON_INCLUDE_DIR)
    # distutils.sysconfig, if it's installed, is more accurate than sysconfig,
    # which sometimes outputs directories that do not exist
    pycmd_no_exit(_py_inc _exitcode "from distutils import sysconfig; print(sysconfig.get_python_inc())")
    if("${_exitcode}" EQUAL 0 AND IS_DIRECTORY "${_py_inc}")
      SET(PYTHON_INCLUDE_DIR "${_py_inc}")
      message(STATUS "Setting Python's include dir to ${_py_inc} from distutils.sysconfig")
    else()
      pycmd_no_exit(_py_inc _exitcode "from sysconfig import get_paths; print(get_paths()['include'])")
      if("${_exitcode}" EQUAL 0 AND IS_DIRECTORY "${_py_inc}")
        SET(PYTHON_INCLUDE_DIR "${_py_inc}")
        message(STATUS "Setting Python's include dir to ${_py_inc} from sysconfig")
      endif()
    endif()
  endif(NOT DEFINED PYTHON_INCLUDE_DIR)

  if(NOT DEFINED PYTHON_LIBRARY)
    pycmd_no_exit(_py_lib _exitcode "from sysconfig import get_paths; print(get_paths()['stdlib'])")
    if("${_exitcode}" EQUAL 0 AND EXISTS "${_py_lib}" AND EXISTS "${_py_lib}")
      SET(PYTHON_LIBRARY "${_py_lib}")
      if (MSVC)
        STRING(REPLACE "Lib" "libs" _py_static_lib ${_py_lib})
        link_directories(${_py_static_lib})
      endif()
      message(STATUS "Setting Python's library to ${PYTHON_LIBRARY}")
    endif()
  endif(NOT DEFINED PYTHON_LIBRARY)

  # These should fill in the rest of the variables, like versions, but resepct
  # the variables we set above
  set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5 2.8 2.7 2.6)
  find_package(PythonInterp 2.7)
  find_package(PythonLibs 2.7)
  find_package(NumPy)
  if(PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})
    caffe2_update_option(USE_NUMPY OFF)
    if(NUMPY_FOUND)
      caffe2_update_option(USE_NUMPY ON)
      include_directories(SYSTEM ${NUMPY_INCLUDE_DIR})
    endif()
    # Observers are required in the python build
    caffe2_update_option(USE_OBSERVERS ON)
  else()
    message(WARNING "Python dependencies not met. Not compiling with python. Suppress this warning with -DBUILD_PYTHON=OFF")
    caffe2_update_option(BUILD_PYTHON OFF)
  endif()
endif()

# ---[ pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
  find_package(pybind11)
endif()

if(pybind11_FOUND)
    message(STATUS "System pybind11 found")
    message(STATUS "pybind11 include dirs: " "${pybind11_INCLUDE_DIRS}")
    include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
else()
    message(STATUS "Using third_party/pybind11.")
    include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/pybind11/include)
endif()

# ---[ MPI
if(USE_MPI)
  find_package(MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${MPI_CXX_LIBRARIES})
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)
    if(OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
                      OUTPUT_VARIABLE _output)
      if(_output MATCHES "smcuda")
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
        set(CAFFE2_FORCE_FALLBACK_CUDA_MPI 1)
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DUSE_MPI=OFF")
    caffe2_update_option(USE_MPI OFF)
  endif()
endif()

# ---[ OpenMP
if(USE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Adding " ${OpenMP_CXX_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(WARNING "Not compiling with OpenMP. Suppress this warning with -DUSE_OPENMP=OFF")
    caffe2_update_option(USE_OPENMP OFF)
  endif()
endif()


# ---[ Android specific ones
if(ANDROID)
  list(APPEND Caffe2_DEPENDENCY_LIBS log)
endif()

# ---[ CUDA
if(USE_CUDA)
  # public/*.cmake uses CAFFE2_USE_*
  set(CAFFE2_USE_CUDA ${USE_CUDA})
  set(CAFFE2_USE_CUDNN ${USE_CUDNN})
  set(CAFFE2_USE_NVRTC ${USE_NVRTC})
  set(CAFFE2_USE_TENSORRT ${USE_TENSORRT})
  include(${CMAKE_CURRENT_LIST_DIR}/public/cuda.cmake)
  if(CAFFE2_USE_CUDA)
    # A helper variable recording the list of Caffe2 dependent libraries
    # caffe2::cudart is dealt with separately, due to CUDA_ADD_LIBRARY
    # design reason (it adds CUDA_LIBRARIES itself).
    set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cufft caffe2::curand)
    if(CAFFE2_USE_NVRTC)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cuda caffe2::nvrtc)
    else()
      caffe2_update_option(USE_NVRTC OFF)
    endif()
    if(CAFFE2_USE_CUDNN)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cudnn)
    else()
      caffe2_update_option(USE_CUDNN OFF)
    endif()
    if(CAFFE2_STATIC_LINK_CUDA)
      # When statically linking, this must be the order of the libraries
      LIST(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS
          "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a" caffe2::cublas)
    else()
      LIST(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::cublas)
    endif()
    if(CAFFE2_USE_TENSORRT)
      list(APPEND Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS caffe2::tensorrt)
    else()
      caffe2_update_option(USE_TENSORRT OFF)
    endif()
  else()
    message(WARNING
      "Not compiling with CUDA. Suppress this warning with "
      "-DUSE_CUDA=OFF.")
    caffe2_update_option(USE_CUDA OFF)
    caffe2_update_option(USE_CUDNN OFF)
    caffe2_update_option(USE_NVRTC OFF)
    caffe2_update_option(USE_TENSORRT OFF)
    set(CAFFE2_USE_CUDA OFF)
    set(CAFFE2_USE_CUDNN OFF)
    set(CAFFE2_USE_NVRTC OFF)
    set(CAFFE2_USE_TENSORRT OFF)
  endif()
endif()

# ---[ HIP
if(NOT BUILD_ATEN_MOBILE)
  include(${CMAKE_CURRENT_LIST_DIR}/public/LoadHIP.cmake)
  if(PYTORCH_FOUND_HIP)
    message(INFO "Compiling with HIP for AMD.")
    caffe2_update_option(USE_ROCM ON)

    list(APPEND HIP_CXX_FLAGS -fPIC)
    list(APPEND HIP_CXX_FLAGS -D__HIP_PLATFORM_HCC__=1)
    list(APPEND HIP_CXX_FLAGS -DCUDA_HAS_FP16=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_OPERATORS__=1)
    list(APPEND HIP_CXX_FLAGS -D__HIP_NO_HALF_CONVERSIONS__=1)
    list(APPEND HIP_CXX_FLAGS -DHIP_VERSION=${HIP_VERSION_MAJOR})
    list(APPEND HIP_CXX_FLAGS -Wno-macro-redefined)
    list(APPEND HIP_CXX_FLAGS -Wno-inconsistent-missing-override)
    list(APPEND HIP_CXX_FLAGS -Wno-exceptions)
    list(APPEND HIP_CXX_FLAGS -Wno-shift-count-negative)
    list(APPEND HIP_CXX_FLAGS -Wno-shift-count-overflow)
    list(APPEND HIP_CXX_FLAGS -Wno-unused-command-line-argument)
    list(APPEND HIP_CXX_FLAGS -Wno-duplicate-decl-specifier)
    list(APPEND HIP_CXX_FLAGS -DCAFFE2_USE_MIOPEN)
    list(APPEND HIP_CXX_FLAGS -DROCBLAS_FP16=0)

    if(CMAKE_BUILD_TYPE MATCHES Debug)
       list(APPEND HIP_CCX_FLAGS -g)
       list(APPEND HIP_CCX_FLAGS -O0)
    endif(CMAKE_BUILD_TYPE MATCHES Debug)

    set(HIP_HCC_FLAGS ${HIP_CXX_FLAGS})
    # Ask hcc to generate device code during compilation so we can use
    # host linker to link.
    list(APPEND HIP_HCC_FLAGS -fno-gpu-rdc)
    list(APPEND HIP_HCC_FLAGS -amdgpu-target=${HCC_AMDGPU_TARGET})

    set(Caffe2_HIP_INCLUDES
      ${hip_INCLUDE_DIRS} ${hcc_INCLUDE_DIRS} ${hsa_INCLUDE_DIRS} ${rocrand_INCLUDE_DIRS} ${hiprand_INCLUDE_DIRS} ${rocblas_INCLUDE_DIRS} ${miopen_INCLUDE_DIRS} ${thrust_INCLUDE_DIRS} $<INSTALL_INTERFACE:include> ${Caffe2_HIP_INCLUDES})

    # This is needed for library added by hip_add_library (same for hip_add_executable)
    hip_include_directories(${Caffe2_HIP_INCLUDES})

    set(Caffe2_HIP_DEPENDENCY_LIBS
      ${rocrand_LIBRARIES} ${hiprand_LIBRARIES} ${PYTORCH_HIP_HCC_LIBRARIES} ${PYTORCH_MIOPEN_LIBRARIES} ${hipsparse_LIBRARIES})
    # TODO: There is a bug in rocblas and rocfft's cmake files that exports the wrong targets name in ${rocblas_LIBRARIES} and ${rocfft_LIBRARIES} respectively
    list(APPEND Caffe2_HIP_DEPENDENCY_LIBS
      roc::rocblas roc::rocfft)

    # TODO: Currently pytorch hipify script uses a feature called
    # "disabled_modules" that effectively ifdef out a file, but
    # without doing extra processing in the callers, which results in
    # some unresolved symbols in the shared lib
    # (libcaffe2_hip.so). Remove this when all disabled_modules are
    # eliminated.
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--unresolved-symbols=ignore-in-shared-libs ${CMAKE_EXE_LINKER_FLAGS}")
  else()
    caffe2_update_option(USE_ROCM OFF)
  endif()
endif()

# ---[ ROCm
if(USE_ROCM)
 include_directories(SYSTEM ${HIP_PATH}/include)
 include_directories(SYSTEM ${ROCBLAS_PATH}/include)
 include_directories(SYSTEM ${ROCFFT_PATH}/include)
 include_directories(SYSTEM ${HIPSPARSE_PATH}/include)
 include_directories(SYSTEM ${HIPRAND_PATH}/include)
 include_directories(SYSTEM ${ROCRAND_PATH}/include)
 include_directories(SYSTEM ${THRUST_PATH})
endif()

# ---[ NCCL
if(USE_NCCL)
  if(NOT USE_CUDA)
    message(WARNING
        "Not using CUDA, so disabling NCCL. Suppress this warning with "
        "-DUSE_NCCL=OFF.")
    caffe2_update_option(USE_NCCL OFF)
  elseif(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "NCCL is currently only supported under Linux.")
    caffe2_update_option(USE_NCCL OFF)
  else()
    include(${CMAKE_CURRENT_LIST_DIR}/External/nccl.cmake)
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS __caffe2_nccl)
  endif()
endif()

# ---[ CUB
if(USE_CUDA)
  find_package(CUB)
  if(CUB_FOUND)
    include_directories(SYSTEM ${CUB_INCLUDE_DIRS})
  else()
    include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/cub)
  endif()
endif()

if(USE_GLOO)
  if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "Gloo can only be used on Linux.")
    caffe2_update_option(USE_GLOO OFF)
  elseif(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "Gloo can only be used on 64-bit systems.")
    caffe2_update_option(USE_GLOO OFF)
  else()
    set(GLOO_INSTALL ON CACHE BOOL "" FORCE)
    set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "" FORCE)

    # Temporarily override variables to avoid building Gloo tests/benchmarks
    set(__BUILD_TEST ${BUILD_TEST})
    set(__BUILD_BENCHMARK ${BUILD_BENCHMARK})
    set(BUILD_TEST OFF)
    set(BUILD_BENCHMARK OFF)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/gloo)
    # Here is a little bit hacky. We have to put PROJECT_BINARY_DIR in front
    # of PROJECT_SOURCE_DIR with/without conda system. The reason is that
    # gloo generates a new config.h in the binary diretory.
    include_directories(BEFORE SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/gloo)
    include_directories(BEFORE SYSTEM ${PROJECT_BINARY_DIR}/third_party/gloo)
    set(BUILD_TEST ${__BUILD_TEST})
    set(BUILD_BENCHMARK ${__BUILD_BENCHMARK})

    # Add explicit dependency if NCCL is built from third_party.
    # Without dependency, make -jN with N>1 can fail if the NCCL build
    # hasn't finished when CUDA targets are linked.
    if(NCCL_EXTERNAL)
      add_dependencies(gloo_cuda nccl_external)
    endif()
    # Pick the right dependency depending on USE_CUDA
    list(APPEND Caffe2_DEPENDENCY_LIBS gloo)
    if(USE_CUDA)
      list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS gloo_cuda)
    endif()
  endif()
endif()

# ---[ profiling
if(USE_PROF)
  find_package(htrace)
  if(htrace_FOUND)
    set(USE_PROF_HTRACE ON)
  else()
    message(WARNING "htrace not found. Caffe2 will build without htrace prof")
  endif()
endif()

if (USE_MOBILE_OPENGL)
  if (ANDROID)
    list(APPEND Caffe2_DEPENDENCY_LIBS EGL GLESv2)
  elseif (IOS)
    message(STATUS "TODO item for adding ios opengl dependency")
  else()
    message(WARNING "mobile opengl is only used in android or ios builds.")
    caffe2_update_option(USE_MOBILE_OPENGL OFF)
  endif()
endif()

# ---[ ARM Compute Library: check compatibility.
if (USE_ACL)
  if (NOT ANDROID)
    message(WARNING "ARM Compute Library is only supported for Android builds.")
    caffe2_update_option(USE_ACL OFF)
  else()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^armv")
      # 32-bit ARM (armv7, armv7-a, armv7l, etc)
      set(ACL_ARCH "armv7a")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
      # 64-bit ARM
      set(ACL_ARCH "arm64-v8a")
    else()
      message(WARNING "ARM Compute Library is only supported for ARM/ARM64 builds.")
      caffe2_update_option(USE_ACL OFF)
    endif()
  endif()
endif()

# ---[ ARM Compute Library: build the target.
if (USE_ACL)
  list(APPEND ARM_COMPUTE_INCLUDE_DIRS "third_party/ComputeLibrary/")
  list(APPEND ARM_COMPUTE_INCLUDE_DIRS "third_party/ComputeLibrary/include")
  include_directories(SYSTEM ${ARM_COMPUTE_INCLUDE_DIRS})
  string (REPLACE ";" " -I" ANDROID_STL_INCLUDE_FLAGS "-I${ANDROID_STL_INCLUDE_DIRS}")
  set (ARM_COMPUTE_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/ComputeLibrary/")
  set (ARM_COMPUTE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libarm_compute.a")
  set (ARM_COMPUTE_CORE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core.a")
  set (ARM_COMPUTE_LIBS ${ARM_COMPUTE_LIB} ${ARM_COMPUTE_CORE_LIB})

  add_custom_command(
      OUTPUT ${ARM_COMPUTE_LIBS}
      COMMAND
        /bin/sh -c "export PATH=\"$PATH:$(dirname ${CMAKE_CXX_COMPILER})\" && \
        scons -C \"${ARM_COMPUTE_SRC_DIR}\" -Q \
          examples=no validation_tests=no benchmark_tests=no standalone=yes \
          embed_kernels=yes opencl=no gles_compute=yes \
          os=android arch=${ACL_ARCH} \
          extra_cxx_flags=\"${ANDROID_CXX_FLAGS} ${ANDROID_STL_INCLUDE_FLAGS}\"" &&
        /bin/sh -c "cp ${ARM_COMPUTE_SRC_DIR}/build/libarm_compute-static.a ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute.a" &&
        /bin/sh -c "cp ${ARM_COMPUTE_SRC_DIR}/build/libarm_compute_core-static.a ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core.a" &&
        /bin/sh -c "rm -r ${ARM_COMPUTE_SRC_DIR}/build"
      COMMENT "Building ARM compute library" VERBATIM)
  add_custom_target(arm_compute_build ALL DEPENDS ${ARM_COMPUTE_LIBS})

  add_library(arm_compute_core STATIC IMPORTED)
  add_dependencies(arm_compute_core arm_compute_build)
  set_property(TARGET arm_compute_core PROPERTY IMPORTED_LOCATION ${ARM_COMPUTE_CORE_LIB})

  add_library(arm_compute STATIC IMPORTED)
  add_dependencies(arm_compute arm_compute_build)
  set_property(TARGET arm_compute PROPERTY IMPORTED_LOCATION ${ARM_COMPUTE_LIB})

  list(APPEND Caffe2_DEPENDENCY_LIBS arm_compute arm_compute_core)
endif()

if (USE_SNPE AND ANDROID)
  if (SNPE_LOCATION AND SNPE_HEADERS)
    message(STATUS "Using SNPE location specified by -DSNPE_LOCATION: " ${SNPE_LOCATION})
    message(STATUS "Using SNPE headers specified by -DSNPE_HEADERS: " ${SNPE_HEADERS})
    include_directories(SYSTEM ${SNPE_HEADERS})
    add_library(snpe SHARED IMPORTED)
    set_property(TARGET snpe PROPERTY IMPORTED_LOCATION ${SNPE_LOCATION})
    list(APPEND Caffe2_DEPENDENCY_LIBS snpe)
  else()
    caffe2_update_option(USE_SNPE OFF)
  endif()
endif()

if (USE_METAL)
  if (NOT IOS)
    message(WARNING "Metal is only used in ios builds.")
    caffe2_update_option(USE_METAL OFF)
  endif()
endif()

if (USE_NNAPI AND NOT ANDROID)
  message(WARNING "NNApi is only used in android builds.")
  caffe2_update_option(USE_NNAPI OFF)
endif()

if (NOT BUILD_ATEN_MOBILE AND BUILD_CAFFE2_OPS)
  if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
    list(APPEND Caffe2_DEPENDENCY_LIBS aten_op_header_gen)
    if (USE_CUDA)
      list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS aten_op_header_gen)
    endif()
    include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten)
  endif()
endif()

if (USE_ZSTD)
  list(APPEND Caffe2_DEPENDENCY_LIBS libzstd_static)
  include_directories(SYSTEM ${CMAKE_CURRENT_LIST_DIR}/../third_party/zstd/lib)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/zstd/build/cmake)
  set_property(TARGET libzstd_static PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# ---[ Onnx
if(NOT BUILD_ATEN_ONLY)
if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if(EXISTS "${CAFFE2_CUSTOM_PROTOC_EXECUTABLE}")
    set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${CAFFE2_CUSTOM_PROTOC_EXECUTABLE})
  endif()
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  # We will build onnx as static libs and embed it directly into the binary.
  if (MSVC AND BUILD_SHARED_LIBS)
    # That also means we want to export all symbols from the shared
    # library we are building
    set(ONNX_BUILD_MAIN_LIB ON)
  endif()
  set(BUILD_SHARED_LIBS OFF)
  set(ONNX_USE_MSVC_STATIC_RUNTIME ${CAFFE2_USE_MSVC_STATIC_RUNTIME})
  set(ONNX_USE_LITE_PROTO ${CAFFE2_USE_LITE_PROTO})
  # If linking local protobuf, make sure ONNX has the same protobuf
  # patches as Caffe2 and Caffe proto. This forces some functions to
  # not be inline and instead route back to the statically-linked protobuf.
  if (CAFFE2_LINK_LOCAL_PROTOBUF)
    set(ONNX_PROTO_POST_BUILD_SCRIPT ${PROJECT_SOURCE_DIR}/cmake/ProtoBufPatch.cmake)
  endif()
  # Add op schemas in "ai.onnx.pytorch" domain
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../caffe2/onnx/torch_ops")
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx)

  include_directories(${ONNX_INCLUDE_DIRS})
  add_definitions(-DONNX_NAMESPACE=${ONNX_NAMESPACE})
  # In mobile build we care about code size, and so we need drop
  # everything (e.g. checker, optimizer) in onnx but the pb definition.
  if (ANDROID OR IOS)
    caffe2_interface_library(onnx_proto onnx_library)
  else()
    caffe2_interface_library(onnx onnx_library)
  endif()
  list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_library)
  list(APPEND Caffe2_DEPENDENCY_LIBS onnxifi_loader)
  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()
endif()

# --[ TensorRT integration with onnx-trt
if (CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
  if (USE_TENSORRT)
    set(CMAKE_CUDA_COMPILER ${CUDA_NVCC_EXECUTABLE})
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx-tensorrt)
    include_directories("${CMAKE_CURRENT_LIST_DIR}/../third_party/onnx-tensorrt")
    caffe2_interface_library(nvonnxparser_static onnx_trt_library)
    list(APPEND Caffe2_DEPENDENCY_WHOLE_LINK_LIBS onnx_trt_library)
    set(CAFFE2_USE_TRT 1)
  endif()
endif()

# --[ ATen checks
if (NOT BUILD_ATEN_MOBILE)
  set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  set(TORCH_NVCC_FLAGS $ENV{TORCH_NVCC_FLAGS})

  # RPATH stuff
  # see https://cmake.org/Wiki/CMake_RPATH_handling
  if (APPLE)
    set(CMAKE_MACOSX_RPATH ON)
  endif()
  set(CMAKE_SKIP_BUILD_RPATH  FALSE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
  list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
  if ("${isSystemDir}" STREQUAL "-1")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
  endif()

  # Top-level build config
  ############################################
  # Flags
  # When using MSVC
  # Detect CUDA architecture and get best NVCC flags
  # finding cuda must be first because other things depend on the result
  #
  # NB: We MUST NOT run this find_package if NOT USE_CUDA is set, because upstream
  # FindCUDA has a bug where it will still attempt to make use of NOTFOUND
  # compiler variables to run various probe tests.  We could try to fix
  # this, but since FindCUDA upstream is subsumed by first-class support
  # for CUDA language, it seemed not worth fixing.

  IF (MSVC)
    # we want to respect the standard, and we are bored of those **** .
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
    LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler /wd4819 -Xcompiler /wd4503 -Xcompiler /wd4190 -Xcompiler /wd4244 -Xcompiler /wd4251 -Xcompiler /wd4275 -Xcompiler /wd4522")
  ENDIF()

  IF (NOT MSVC)
    IF (CMAKE_VERSION VERSION_LESS "3.1")
      SET(CMAKE_C_FLAGS "-std=c11 ${CMAKE_C_FLAGS}")
    ELSE ()
      SET(CMAKE_C_STANDARD 11)
    ENDIF ()
  ENDIF()

  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9")
      if (CUDA_VERSION VERSION_LESS "8.0")
        MESSAGE(STATUS "Found gcc >=5 and CUDA <= 7.5, adding workaround C++ flags")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
      endif()
    endif()
  endif()

  LIST(APPEND CUDA_NVCC_FLAGS -Wno-deprecated-gpu-targets)
  LIST(APPEND CUDA_NVCC_FLAGS --expt-extended-lambda)

  if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    SET(CMAKE_CXX_STANDARD 11)
  endif()

  LIST(APPEND CUDA_NVCC_FLAGS ${TORCH_NVCC_FLAGS})
  LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
  IF (CMAKE_POSITION_INDEPENDENT_CODE AND NOT MSVC)
    LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
  ENDIF()

  IF (CUDA_HAS_FP16 OR NOT ${CUDA_VERSION} LESS 7.5)
    MESSAGE(STATUS "Found CUDA with FP16 support, compiling with torch.cuda.HalfTensor")
    LIST(APPEND CUDA_NVCC_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__")
    add_compile_options(-DCUDA_HAS_FP16=1)
  ELSE()
    MESSAGE(STATUS "Could not find CUDA with FP16 support, compiling without torch.CudaHalfTensor")
  ENDIF()

  OPTION(NDEBUG "disable asserts (WARNING: this may result in silent UB e.g. with out-of-bound indices)")
  IF (NOT NDEBUG)
    MESSAGE(STATUS "Removing -DNDEBUG from compile flags")
    STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS "" ${CMAKE_C_FLAGS})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_DEBUG "" ${CMAKE_C_FLAGS_DEBUG})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "" ${CMAKE_C_FLAGS_RELEASE})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS "" ${CMAKE_CXX_FLAGS})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_DEBUG "" ${CMAKE_CXX_FLAGS_DEBUG})
    STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "" ${CMAKE_CXX_FLAGS_RELEASE})
  ENDIF()

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
      add_compile_options(-Wno-unknown-pragmas)
      SET(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?" FORCE)
    ENDIF()
  ENDIF()

  IF (WITH_OPENMP AND NOT CHECKED_OPENMP)
    FIND_PACKAGE(OpenMP)
    SET(CHECKED_OPENMP ON CACHE BOOL "already checked for OpenMP")

    # OPENMP_FOUND is not cached in FindOpenMP.cmake (all other variables are cached)
    # see https://github.com/Kitware/CMake/blob/master/Modules/FindOpenMP.cmake
    SET(OPENMP_FOUND ${OPENMP_FOUND} CACHE BOOL "OpenMP Support found")
  ENDIF()

  IF (OPENMP_FOUND)
    MESSAGE(STATUS "Compiling with OpenMP support")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()


  SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

  FIND_PACKAGE(MAGMA)
  IF (USE_CUDA AND MAGMA_FOUND)
    INCLUDE_DIRECTORIES(SYSTEM ${MAGMA_INCLUDE_DIR})
    SET(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${CUDA_INCLUDE_DIRS}")
    INCLUDE(CheckPrototypeDefinition)
    check_prototype_definition(magma_get_sgeqrf_nb
     "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
     "0"
     "magma.h"
      MAGMA_V2)
    IF (MAGMA_V2)
      add_definitions(-DMAGMA_V2)
    ENDIF (MAGMA_V2)

    SET(USE_MAGMA 1)
    MESSAGE(STATUS "Compiling with MAGMA support")
    MESSAGE(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
    MESSAGE(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
    MESSAGE(STATUS "MAGMA V2 check: ${MAGMA_V2}")
  ELSE()
    MESSAGE(STATUS "MAGMA not found. Compiling without MAGMA support")
  ENDIF()

  # ARM specific flags
  FIND_PACKAGE(ARM)
  IF (ASIMD_FOUND)
    MESSAGE(STATUS "asimd/Neon found with compiler flag : -D__NEON__")
    add_compile_options(-D__NEON__)
  ELSEIF (NEON_FOUND)
    MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
    add_compile_options(-mfpu=neon -D__NEON__)
  ENDIF ()
  IF (CORTEXA8_FOUND)
    MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
    add_compile_options(-mcpu=cortex-a8 -fprefetch-loop-arrays)
  ENDIF ()
  IF (CORTEXA9_FOUND)
    MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
    add_compile_options(-mcpu=cortex-a9)
  ENDIF()

  CHECK_INCLUDE_FILE(cpuid.h HAVE_CPUID_H)
  # Check for a cpuid intrinsic
  IF (HAVE_CPUID_H)
      CHECK_C_SOURCE_COMPILES("#include <cpuid.h>
          int main()
          {
              unsigned int eax, ebx, ecx, edx;
              return __get_cpuid(0, &eax, &ebx, &ecx, &edx);
          }" HAVE_GCC_GET_CPUID)
  ENDIF()
  IF (HAVE_GCC_GET_CPUID)
    add_compile_options(-DHAVE_GCC_GET_CPUID)
  ENDIF()

  CHECK_C_SOURCE_COMPILES("#include <stdint.h>
      static inline void cpuid(uint32_t *eax, uint32_t *ebx,
      			 uint32_t *ecx, uint32_t *edx)
      {
        uint32_t a = *eax, b, c = *ecx, d;
        asm volatile ( \"cpuid\" : \"+a\"(a), \"=b\"(b), \"+c\"(c), \"=d\"(d) );
        *eax = a; *ebx = b; *ecx = c; *edx = d;
      }
      int main() {
        uint32_t a,b,c,d;
        cpuid(&a, &b, &c, &d);
        return 0;
      }" NO_GCC_EBX_FPIC_BUG)

  IF (NOT NO_GCC_EBX_FPIC_BUG)
    add_compile_options(-DUSE_GCC_GET_CPUID)
  ENDIF()

  FIND_PACKAGE(AVX) # checks AVX and AVX2

  # we don't set -mavx and -mavx2 flags globally, but only for specific files
  # however, we want to enable the AVX codepaths, so we still need to
  # add USE_AVX and USE_AVX2 macro defines
  IF (C_AVX_FOUND)
    MESSAGE(STATUS "AVX compiler support found")
    add_compile_options(-DUSE_AVX)
  ENDIF()
  IF (C_AVX2_FOUND)
    MESSAGE(STATUS "AVX2 compiler support found")
    add_compile_options(-DUSE_AVX2)
  ENDIF()

  CHECK_C_SOURCE_RUNS("
  #include <stdatomic.h>
  // ATOMIC_INT_LOCK_FREE is flaky on some older gcc versions
  // so if this define is not usable a preprocessor definition
  // we fail this check and fall back to GCC atomics
  #if ATOMIC_INT_LOCK_FREE == 2
  #define TH_ATOMIC_IPC_REFCOUNT 1
  #endif
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

  IF (NOT HAS_C11_ATOMICS)
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

  IF (HAS_C11_ATOMICS)
    ADD_DEFINITIONS(-DUSE_C11_ATOMICS=1)
    MESSAGE(STATUS "Atomics: using C11 intrinsics")
  ELSEIF (HAS_MSC_ATOMICS)
    ADD_DEFINITIONS(-DUSE_MSC_ATOMICS=1)
    MESSAGE(STATUS "Atomics: using MSVC intrinsics")
  ELSEIF (HAS_GCC_ATOMICS)
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

  IF (WIN32 AND NOT CYGWIN)
    SET(BLAS_INSTALL_LIBRARIES "OFF"
      CACHE BOOL "Copy the required BLAS DLLs into the TH install dirs")
  ENDIF()

  FIND_PACKAGE(LAPACK)
  IF (LAPACK_FOUND)
    SET(USE_LAPACK 1)
  ENDIF()

  if (NOT USE_CUDA)
    message("disabling CUDA because NOT USE_CUDA is set")
    SET(AT_CUDA_ENABLED 0)
  else()
    SET(AT_CUDA_ENABLED 1)
    find_package(CUDA 5.5 REQUIRED)
  endif()

  IF (NOT AT_CUDA_ENABLED OR NOT CUDNN_FOUND)
    MESSAGE(STATUS "CuDNN not found. Compiling without CuDNN support")
    set(AT_CUDNN_ENABLED 0)
  ELSE()
    include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
    set(AT_CUDNN_ENABLED 1)
  ENDIF()

  IF (NOT USE_ROCM)
    message("disabling ROCM because NOT USE_ROCM is set")
    MESSAGE(STATUS "MIOpen not found. Compiling without MIOpen support")
    set(AT_ROCM_ENABLED 0)
  ELSE()
    INCLUDE_DIRECTORIES(BEFORE ${MIOPEN_INCLUDE_DIRS})
    set(AT_ROCM_ENABLED 1)
  ENDIF()

  SET(AT_MKLDNN_ENABLED 0)
  SET(CAFFE2_USE_MKLDNN OFF)
  IF (USE_MKLDNN)
    FIND_PACKAGE(MKLDNN)
    INCLUDE(${CMAKE_CURRENT_LIST_DIR}/public/mkldnn.cmake)
    IF(MKLDNN_FOUND)
      SET(AT_MKLDNN_ENABLED 1)
      INCLUDE_DIRECTORIES(SYSTEM ${MKLDNN_INCLUDE_DIR})
      IF(BUILD_CAFFE2_OPS)
        SET(CAFFE2_USE_MKLDNN ON)
        LIST(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::mkldnn)
      ENDIF(BUILD_CAFFE2_OPS)
    ELSE()
      MESSAGE(WARNING "MKLDNN could not be found.")
    ENDIF()
  ELSE()
    MESSAGE("disabling MKLDNN because USE_MKLDNN is not set")
  ENDIF()

  IF(UNIX AND NOT APPLE)
     INCLUDE(CheckLibraryExists)
     # https://github.com/libgit2/libgit2/issues/2128#issuecomment-35649830
     CHECK_LIBRARY_EXISTS(rt clock_gettime "time.h" NEED_LIBRT)
     IF(NEED_LIBRT)
       list(APPEND Caffe2_DEPENDENCY_LIBS rt)
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

  # Is __thread supported?
  IF(NOT MSVC)
    CHECK_C_SOURCE_COMPILES("static __thread int x = 1; int main() { return x; }" C_HAS_THREAD)
  ELSE(NOT MSVC)
    CHECK_C_SOURCE_COMPILES("static __declspec( thread ) int x = 1; int main() { return x; }" C_HAS_THREAD)
  ENDIF(NOT MSVC)
  IF(NOT C_HAS_THREAD)
    MESSAGE(STATUS "Warning: __thread is not supported, generating thread-unsafe code")
  ELSE(NOT C_HAS_THREAD)
    add_compile_options(-DTH_HAVE_THREAD)
  ENDIF(NOT C_HAS_THREAD)
endif()

#
# End ATen checks
#
