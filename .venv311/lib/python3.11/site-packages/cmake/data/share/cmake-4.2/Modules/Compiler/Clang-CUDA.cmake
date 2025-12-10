include(Compiler/Clang)
__compiler_clang(CUDA)

# Set explicitly, because __compiler_clang() doesn't set this if we're simulating MSVC.
set(CMAKE_DEPFILE_FLAGS_CUDA "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
if((NOT DEFINED CMAKE_DEPENDS_USE_COMPILER OR CMAKE_DEPENDS_USE_COMPILER)
    AND CMAKE_GENERATOR MATCHES "Makefiles|WMake")
  # dependencies are computed by the compiler itself
  set(CMAKE_CUDA_DEPFILE_FORMAT gcc)
  set(CMAKE_CUDA_DEPENDS_USE_COMPILER TRUE)
endif()

# C++03 isn't supported for CXX, but is for CUDA, so we need to set these manually.
# Do this before __compiler_clang_cxx_standards() since that adds the feature.
set(CMAKE_CUDA03_STANDARD_COMPILE_OPTION "-std=c++03")
set(CMAKE_CUDA03_EXTENSION_COMPILE_OPTION "-std=gnu++03")
__compiler_clang_cxx_standards(CUDA)

set(CMAKE_CUDA_COMPILER_HAS_DEVICE_LINK_PHASE TRUE)
set(_CMAKE_COMPILE_AS_CUDA_FLAG "-x cuda")
set(_CMAKE_CUDA_WHOLE_FLAG "-c")
set(_CMAKE_CUDA_RDC_FLAG "-fgpu-rdc")
set(_CMAKE_CUDA_PTX_FLAG "--cuda-device-only -S")

# Device linking is just regular linking so these are the same.
set(CMAKE_CUDA_DEVICE_LINKER_WRAPPER_FLAG ${CMAKE_CUDA_LINKER_WRAPPER_FLAG})
set(CMAKE_CUDA_DEVICE_LINKER_WRAPPER_FLAG_SEP ${CMAKE_CUDA_LINKER_WRAPPER_FLAG_SEP})

set(CMAKE_CUDA_DEVICE_LINK_MODE DRIVER)

# RulePlaceholderExpander expands crosscompile variables like sysroot and target only for CMAKE_<LANG>_COMPILER. Override the default.
set(CMAKE_CUDA_LINK_EXECUTABLE "<CMAKE_CUDA_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")
set(CMAKE_CUDA_CREATE_SHARED_LIBRARY "<CMAKE_CUDA_COMPILER> <CMAKE_SHARED_LIBRARY_CUDA_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>${__IMPLICIT_LINKS}")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")
set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC "cudadevrt;cudart_static")
set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED "cudadevrt;cudart")
set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_NONE   "")

# Clang doesn't support CUDA device LTO
set(_CMAKE_CUDA_IPO_SUPPORTED_BY_CMAKE NO)
set(_CMAKE_CUDA_IPO_MAY_BE_SUPPORTED_BY_COMPILER NO)

if(UNIX)
  list(APPEND CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC "rt" "pthread" "dl")
endif()
