# ---[ cuda

# Poor man's include guard
if(TARGET torch::cudart)
  return()
endif()

# sccache is only supported in CMake master and not in the newest official
# release (3.11.3) yet. Hence we need our own Modules_CUDA_fix to enable sccache.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../Modules_CUDA_fix)

# We don't want to statically link cudart, because we rely on it's dynamic linkage in
# python (follow along torch/cuda/__init__.py and usage of cudaGetErrorName).
# Technically, we can link cudart here statically, and link libtorch_python.so
# to a dynamic libcudart.so, but that's just wasteful.
# However, on Windows, if this one gets switched off, the error "cuda: unknown error"
# will be raised when running the following code:
# >>> import torch
# >>> torch.cuda.is_available()
# >>> torch.cuda.current_device()
# More details can be found in the following links.
# https://github.com/pytorch/pytorch/issues/20635
# https://github.com/pytorch/pytorch/issues/17108
if(NOT MSVC)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
endif()

# Find CUDA.
find_package(CUDA)
if(NOT CUDA_FOUND)
  message(WARNING
    "Caffe2: CUDA cannot be found. Depending on whether you are building "
    "Caffe2 or a Caffe2 dependent library, the next warning / error will "
    "give you more info.")
  set(CAFFE2_USE_CUDA OFF)
  return()
endif()
message(STATUS "Caffe2: CUDA detected: " ${CUDA_VERSION})
message(STATUS "Caffe2: CUDA nvcc is: " ${CUDA_NVCC_EXECUTABLE})
message(STATUS "Caffe2: CUDA toolkit directory: " ${CUDA_TOOLKIT_ROOT_DIR})
if(CUDA_VERSION VERSION_LESS 9.0)
  message(FATAL_ERROR "PyTorch requires CUDA 9.0 and above.")
endif()

if(CUDA_FOUND)
  # Sometimes, we may mismatch nvcc with the CUDA headers we are
  # compiling with, e.g., if a ccache nvcc is fed to us by CUDA_NVCC_EXECUTABLE
  # but the PATH is not consistent with CUDA_HOME.  It's better safe
  # than sorry: make sure everything is consistent.
  if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    # When using Visual Studio, it attempts to lock the whole binary dir when
    # `try_run` is called, which will cause the build to fail.
    string(RANDOM BUILD_SUFFIX)
    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}/${BUILD_SUFFIX}")
  else()
    set(PROJECT_RANDOM_BINARY_DIR "${PROJECT_BINARY_DIR}")
  endif()
  set(file "${PROJECT_BINARY_DIR}/detect_cuda_version.cc")
  file(WRITE ${file} ""
    "#include <cuda.h>\n"
    "#include <cstdio>\n"
    "int main() {\n"
    "  printf(\"%d.%d\", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);\n"
    "  return 0;\n"
    "}\n"
    )
  try_run(run_result compile_result ${PROJECT_RANDOM_BINARY_DIR} ${file}
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
    LINK_LIBRARIES ${CUDA_LIBRARIES}
    RUN_OUTPUT_VARIABLE cuda_version_from_header
    COMPILE_OUTPUT_VARIABLE output_var
    )
  if(NOT compile_result)
    message(FATAL_ERROR "Caffe2: Couldn't determine version from header: " ${output_var})
  endif()
  message(STATUS "Caffe2: Header version is: " ${cuda_version_from_header})
  if(NOT cuda_version_from_header STREQUAL ${CUDA_VERSION_STRING})
    # Force CUDA to be processed for again next time
    # TODO: I'm not sure if this counts as an implementation detail of
    # FindCUDA
    set(${cuda_version_from_findcuda} ${CUDA_VERSION_STRING})
    unset(CUDA_TOOLKIT_ROOT_DIR_INTERNAL CACHE)
    # Not strictly necessary, but for good luck.
    unset(CUDA_VERSION CACHE)
    # Error out
    message(FATAL_ERROR "FindCUDA says CUDA version is ${cuda_version_from_findcuda} (usually determined by nvcc), "
      "but the CUDA headers say the version is ${cuda_version_from_header}.  This often occurs "
      "when you set both CUDA_HOME and CUDA_NVCC_EXECUTABLE to "
      "non-standard locations, without also setting PATH to point to the correct nvcc.  "
      "Perhaps, try re-running this command again with PATH=${CUDA_TOOLKIT_ROOT_DIR}/bin:$PATH.  "
      "See above log messages for more diagnostics, and see https://github.com/pytorch/pytorch/issues/8092 for more details.")
  endif()
endif()

# Find cuDNN.
if(CAFFE2_STATIC_LINK_CUDA AND NOT USE_STATIC_CUDNN)
  message(WARNING "cuDNN will be linked statically because CAFFE2_STATIC_LINK_CUDA is ON. "
    "Set USE_STATIC_CUDNN to ON to suppress this warning.")
endif()
if(CAFFE2_STATIC_LINK_CUDA OR USE_STATIC_CUDNN)
  set(CUDNN_STATIC ON CACHE BOOL "")
else()
  set(CUDNN_STATIC OFF CACHE BOOL "")
endif()

find_package(CUDNN)

if(CAFFE2_USE_CUDNN AND NOT CUDNN_FOUND)
  message(WARNING
    "Caffe2: Cannot find cuDNN library. Turning the option off")
  set(CAFFE2_USE_CUDNN OFF)
endif()

# Optionally, find TensorRT
if(CAFFE2_USE_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
  find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
  find_package_handle_standard_args(
    TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
  if(TENSORRT_FOUND)
    execute_process(COMMAND /bin/sh -c "[ -r \"${TENSORRT_INCLUDE_DIR}/NvInferVersion.h\" ] && awk '/^\#define NV_TENSORRT_MAJOR/ {print $3}' \"${TENSORRT_INCLUDE_DIR}/NvInferVersion.h\"" OUTPUT_VARIABLE TENSORRT_VERSION_MAJOR)
    execute_process(COMMAND /bin/sh -c "[ -r \"${TENSORRT_INCLUDE_DIR}/NvInferVersion.h\" ] && awk '/^\#define NV_TENSORRT_MINOR/ {print $3}' \"${TENSORRT_INCLUDE_DIR}/NvInferVersion.h\"" OUTPUT_VARIABLE TENSORRT_VERSION_MINOR)
    if(TENSORRT_VERSION_MAJOR)
      string(STRIP ${TENSORRT_VERSION_MAJOR} TENSORRT_VERSION_MAJOR)
      string(STRIP ${TENSORRT_VERSION_MINOR} TENSORRT_VERSION_MINOR)
      set(TENSORRT_VERSION "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}")
      #CAFFE2_USE_TRT is set in Dependencies
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTENSORRT_VERSION_MAJOR=${TENSORRT_VERSION_MAJOR}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTENSORRT_VERSION_MINOR=${TENSORRT_VERSION_MINOR}")
    else()
      message(WARNING "Caffe2: Cannot find ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h. Assuming TRT 5.0 which is no longer supported. Turning the option off.")
      set(CAFFE2_USE_TENSORRT OFF)
    endif()
  else()
    message(WARNING
      "Caffe2: Cannot find TensorRT library. Turning the option off.")
    set(CAFFE2_USE_TENSORRT OFF)
  endif()
endif()

# ---[ Extract versions
if(CAFFE2_USE_CUDNN)
  # Get cuDNN version
  if(EXISTS ${CUDNN_INCLUDE_PATH}/cudnn_version.h)
    file(READ ${CUDNN_INCLUDE_PATH}/cudnn_version.h CUDNN_HEADER_CONTENTS)
  else()
    file(READ ${CUDNN_INCLUDE_PATH}/cudnn.h CUDNN_HEADER_CONTENTS)
  endif()
  string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
               CUDNN_VERSION_MAJOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
  string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
               CUDNN_VERSION_MINOR "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
               CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
  string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
               CUDNN_VERSION_PATCH "${CUDNN_HEADER_CONTENTS}")
  string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
               CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
  # Assemble cuDNN version
  if(NOT CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "?")
  else()
    set(CUDNN_VERSION
        "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  endif()
  message(STATUS "Found cuDNN: v${CUDNN_VERSION}  (include: ${CUDNN_INCLUDE_PATH}, library: ${CUDNN_LIBRARY_PATH})")
  if(CUDNN_VERSION VERSION_LESS "7.0.0")
    message(FATAL_ERROR "PyTorch requires cuDNN 7 and above.")
  endif()
endif()

# ---[ CUDA libraries wrapper

# find libcuda.so and lbnvrtc.so
# For libcuda.so, we will find it under lib, lib64, and then the
# stubs folder, in case we are building on a system that does not
# have cuda driver installed. On windows, we also search under the
# folder lib/x64.
find_library(CUDA_CUDA_LIB cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/stubs lib64/stubs lib/x64)
find_library(CUDA_NVRTC_LIB nvrtc
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
  if(CUDA_NVRTC_LIB AND NOT CUDA_NVRTC_SHORTHASH)
 execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
    "import hashlib;hash=hashlib.sha256();hash.update(open('${CUDA_NVRTC_LIB}','rb').read());print(hash.hexdigest()[:8])"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE CUDA_NVRTC_SHORTHASH)
  if(NOT _retval EQUAL 0)
    message(WARNING "Failed to compute shorthash for libnvrtc.so")
    set(CUDA_NVRTC_SHORTHASH "XXXXXXXX")
  else()
    string(STRIP "${CUDA_NVRTC_SHORTHASH}" CUDA_NVRTC_SHORTHASH)
    message(STATUS "${CUDA_NVRTC_LIB} shorthash is ${CUDA_NVRTC_SHORTHASH}")
  endif()
endif()

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if CAFFE2_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static
# libraries installed. This flag should only be used for binary builds, so
# end-users should never have this flag set.

# cuda
add_library(caffe2::cuda UNKNOWN IMPORTED)
set_property(
    TARGET caffe2::cuda PROPERTY IMPORTED_LOCATION
    ${CUDA_CUDA_LIB})
set_property(
    TARGET caffe2::cuda PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cudart. CUDA_LIBRARIES is actually a list, so we will make an interface
# library.
add_library(torch::cudart INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_cudart_static_LIBRARY}")
    if(NOT WIN32)
      set_property(
          TARGET torch::cudart APPEND PROPERTY INTERFACE_LINK_LIBRARIES
          rt dl)
    endif()
else()
    set_property(
        TARGET torch::cudart PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_LIBRARIES})
endif()
set_property(
    TARGET torch::cudart PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# nvToolsExt
add_library(torch::nvtoolsext INTERFACE IMPORTED)
if(MSVC)
  if(NOT NVTOOLEXT_HOME)
    set(NVTOOLEXT_HOME "C:/Program Files/NVIDIA Corporation/NvToolsExt")
  endif()
  if(DEFINED ENV{NVTOOLSEXT_PATH})
    set(NVTOOLEXT_HOME $ENV{NVTOOLSEXT_PATH})
    file(TO_CMAKE_PATH ${NVTOOLEXT_HOME} NVTOOLEXT_HOME)
  endif()
  set_target_properties(
      torch::nvtoolsext PROPERTIES
      INTERFACE_LINK_LIBRARIES ${NVTOOLEXT_HOME}/lib/x64/nvToolsExt64_1.lib
      INTERFACE_INCLUDE_DIRECTORIES ${NVTOOLEXT_HOME}/include)

elseif(APPLE)
  set_property(
      TARGET torch::nvtoolsext PROPERTY INTERFACE_LINK_LIBRARIES
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvrtc.dylib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/libnvToolsExt.dylib)

else()
  find_library(LIBNVTOOLSEXT libnvToolsExt.so PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/)
  set_property(
      TARGET torch::nvtoolsext PROPERTY INTERFACE_LINK_LIBRARIES
      ${LIBNVTOOLSEXT})
endif()

# cudnn
# static linking is handled by USE_STATIC_CUDNN environment variable
if(CAFFE2_USE_CUDNN)
  add_library(caffe2::cudnn UNKNOWN IMPORTED)
  set_property(
      TARGET caffe2::cudnn PROPERTY IMPORTED_LOCATION
      ${CUDNN_LIBRARY_PATH})
  set_property(
      TARGET caffe2::cudnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${CUDNN_INCLUDE_PATH})
  if(CUDNN_STATIC AND NOT WIN32)
    set_property(
        TARGET caffe2::cudnn PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a" dl)
    # Lines below use target_link_libraries because we support cmake 3.5+.
    # For cmake 3.13+, target_link_options to set INTERFACE_LINK_OPTIONS would be better.
    # https://cmake.org/cmake/help/v3.5/command/target_link_libraries.html warns
    # "Item names starting with -, but not -l or -framework, are treated as linker flags.
    #  Note that such flags will be treated like any other library link item for purposes
    #  of transitive dependencies, so they are generally safe to specify only as private
    #  link items that will not propagate to dependents."
    # Propagating to a dependent (torch_cuda) is exactly what we want here, so we are
    # flouting the warning, but I can't think of a better (3.5+ compatible) way.
    target_link_libraries(caffe2::cudnn INTERFACE
        "-Wl,--exclude-libs,libcudnn_static.a")
  endif()
endif()

# curand
add_library(caffe2::curand UNKNOWN IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
    set_property(
        TARGET caffe2::curand PROPERTY IMPORTED_LOCATION
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand_static.a")
    set_property(
        TARGET caffe2::curand PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a" dl)
else()
    set_property(
        TARGET caffe2::curand PROPERTY IMPORTED_LOCATION
        ${CUDA_curand_LIBRARY})
endif()
set_property(
    TARGET caffe2::curand PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# cufft. CUDA_CUFFT_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(caffe2::cufft INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
    set_property(
        TARGET caffe2::cufft PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft_static_nocallback.a"
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libculibos.a" dl)
else()
    set_property(
        TARGET caffe2::cufft PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_CUFFT_LIBRARIES})
endif()
set_property(
    TARGET caffe2::cufft PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# TensorRT
if(CAFFE2_USE_TENSORRT)
  add_library(caffe2::tensorrt UNKNOWN IMPORTED)
  set_property(
      TARGET caffe2::tensorrt PROPERTY IMPORTED_LOCATION
      ${TENSORRT_LIBRARY})
  set_property(
      TARGET caffe2::tensorrt PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${TENSORRT_INCLUDE_DIR})
endif()

# cublas. CUDA_CUBLAS_LIBRARIES is actually a list, so we will make an
# interface library similar to cudart.
add_library(caffe2::cublas INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublas_static.a")
    if(CUDA_VERSION VERSION_GREATER_EQUAL 10.1)
      set_property(
        TARGET caffe2::cublas APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcublasLt_static.a")
      # Add explicit dependency to cudart_static to fix
      # libcublasLt_static.a.o): undefined reference to symbol 'cudaStreamWaitEvent'
      # error adding symbols: DSO missing from command line
      set_property(
        TARGET caffe2::cublas APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        "${CUDA_cudart_static_LIBRARY}" rt dl)
    endif()
else()
    set_property(
        TARGET caffe2::cublas PROPERTY INTERFACE_LINK_LIBRARIES
        ${CUDA_CUBLAS_LIBRARIES})
endif()
set_property(
    TARGET caffe2::cublas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# nvrtc
add_library(caffe2::nvrtc UNKNOWN IMPORTED)
set_property(
    TARGET caffe2::nvrtc PROPERTY IMPORTED_LOCATION
    ${CUDA_NVRTC_LIB})
set_property(
    TARGET caffe2::nvrtc PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${CUDA_INCLUDE_DIRS})

# Note: in theory, we can add similar dependent library wrappers. For
# now, Caffe2 only uses the above libraries, so we will only wrap
# these.

# Special care for windows platform: we know that 32-bit windows does not
# support cuda.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message(FATAL_ERROR
            "CUDA support not available with 32-bit windows. Did you "
            "forget to set Win64 in the generator target?")
    return()
  endif()
endif()

# Add onnx namepsace definition to nvcc
if(ONNX_NAMESPACE)
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
else()
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=onnx_c2")
endif()

# CUDA 9.0 & 9.1 require GCC version <= 5
# Although they support GCC 6, but a bug that wasn't fixed until 9.2 prevents
# them from compiling the std::tuple header of GCC 6.
# See Sec. 2.2.1 of
# https://developer.download.nvidia.com/compute/cuda/9.2/Prod/docs/sidebar/CUDA_Toolkit_Release_Notes.pdf
if((CUDA_VERSION VERSION_EQUAL   9.0) OR
    (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 9.2))
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
      NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 6.0 AND
      CUDA_HOST_COMPILER STREQUAL CMAKE_C_COMPILER)
    message(FATAL_ERROR
      "CUDA ${CUDA_VERSION} is not compatible with std::tuple from GCC version "
      ">= 6. Please upgrade to CUDA 9.2 or set the following environment "
      "variable to use another version (for example): \n"
      "  export CUDAHOSTCXX='/usr/bin/gcc-5'\n")
  endif()
endif()

# CUDA 9.0 / 9.1 require MSVC version < 19.12
# CUDA 9.2 require MSVC version < 19.13
# CUDA 10.0 require MSVC version < 19.20
if((CUDA_VERSION VERSION_EQUAL   9.0) OR
    (CUDA_VERSION VERSION_GREATER 9.0  AND CUDA_VERSION VERSION_LESS 9.2))
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
      NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.12 AND
      NOT DEFINED ENV{CUDAHOSTCXX})
        message(FATAL_ERROR
          "CUDA ${CUDA_VERSION} is not compatible with MSVC toolchain version "
          ">= 19.12. (a.k.a Visual Studio 2017 Update 5, VS 15.5) "
          "Please upgrade to CUDA >= 9.2 or set the following environment "
          "variable to use another version (for example): \n"
          "  set \"CUDAHOSTCXX=C:\\Program Files (x86)\\Microsoft Visual Studio"
          "\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.11.25503\\bin\\HostX64\\x64\\cl.exe\"\n")
  endif()
elseif(CUDA_VERSION VERSION_EQUAL   9.2)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
      NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.14 AND
      NOT DEFINED ENV{CUDAHOSTCXX})
    message(FATAL_ERROR
      "CUDA ${CUDA_VERSION} is not compatible with MSVC toolchain version "
      ">= 19.14. (a.k.a Visual Studio 2017 Update 7, VS 15.7) "
      "Please upgrade to CUDA >= 10.0 or set the following environment "
      "variable to use another version (for example): \n"
      "  set \"CUDAHOSTCXX=C:\\Program Files (x86)\\Microsoft Visual Studio"
      "\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.13.26132\\bin\\HostX64\\x64\\cl.exe\"\n")
  endif()
elseif(CUDA_VERSION VERSION_EQUAL   10.0)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND
      NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.20 AND
      NOT DEFINED ENV{CUDAHOSTCXX})
    message(FATAL_ERROR
      "CUDA ${CUDA_VERSION} is not compatible with MSVC toolchain version "
      ">= 19.20. (a.k.a Visual Studio 2019, VS 16.0) "
      "Please upgrade to CUDA >= 10.1 or set the following environment "
      "variable to use another version (for example): \n"
      "  set \"CUDAHOSTCXX=C:\\Program Files (x86)\\Microsoft Visual Studio"
      "\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.16.27023\\bin\\HostX64\\x64\\cl.exe\"\n")
  endif()
endif()

# Don't activate VC env again for Ninja generators with MSVC on Windows if CUDAHOSTCXX is not defined
# by adding --use-local-env.
if(MSVC AND CMAKE_GENERATOR STREQUAL "Ninja" AND NOT DEFINED ENV{CUDAHOSTCXX})
  list(APPEND CUDA_NVCC_FLAGS "--use-local-env")
  # For CUDA < 9.2, --cl-version xxx is also required.
  # We could detect cl version according to the following variable
  # https://cmake.org/cmake/help/latest/variable/MSVC_TOOLSET_VERSION.html#variable:MSVC_TOOLSET_VERSION.
  # 140       = VS 2015 (14.0)
  # 141       = VS 2017 (15.0)
  if(CUDA_VERSION VERSION_LESS 9.2)
    if(MSVC_TOOLSET_VERSION EQUAL 140)
      list(APPEND CUDA_NVCC_FLAGS "--cl-version" "2015")
    elseif(MSVC_TOOLSET_VERSION EQUAL 141)
      list(APPEND CUDA_NVCC_FLAGS "--cl-version" "2017")
    else()
      message(STATUS "We could not auto-detect the cl-version for MSVC_TOOLSET_VERSION=${MSVC_TOOLSET_VERSION}")
    endif()
  endif()
endif()

# setting nvcc arch flags
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# disable some nvcc diagnostic that appears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration
             set_but_not_used field_without_dll_interface
             base_class_has_different_dll_interface
             dll_interface_conflict_none_assumed
             dll_interface_conflict_dllexport_assumed
             implicit_return_from_non_void_function
             unsigned_compare_with_zero
             declared_but_not_referenced
             bad_friend_decl)
  list(APPEND SUPPRESS_WARNING_FLAGS --diag_suppress=${diag})
endforeach()
string(REPLACE ";" "," SUPPRESS_WARNING_FLAGS "${SUPPRESS_WARNING_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS -Xcudafe ${SUPPRESS_WARNING_FLAGS})

# Set C++14 support
set(CUDA_PROPAGATE_HOST_FLAGS_BLOCKLIST "-Werror")
if(MSVC)
  list(APPEND CUDA_NVCC_FLAGS "--Werror" "cross-execution-space-call")
  list(APPEND CUDA_NVCC_FLAGS "--no-host-device-move-forward")
else()
  list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-fPIC")
endif()

# OpenMP flags for NVCC with Clang-cl
if("${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC"
  AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CUDA_PROPAGATE_HOST_FLAGS_BLOCKLIST "-Xclang" "-fopenmp")
  if(MSVC_TOOLSET_VERSION LESS 142)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-openmp")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-openmp:experimental")
  endif()
endif()

# Debug and Release symbol support
if(MSVC)
  if(${CAFFE2_USE_MSVC_STATIC_RUNTIME})
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MT$<$<CONFIG:Debug>:d>")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-MD$<$<CONFIG:Debug>:d>")
  endif()
  if(CUDA_NVCC_FLAGS MATCHES "Zi")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-FS")
  endif()
elseif(CUDA_DEVICE_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS "-g" "-G")  # -G enables device code debugging symbols
endif()

# Set expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
