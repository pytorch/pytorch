# ---[ xpu

# Poor man's include guard
if(TARGET torch::xpurt)
  return()
endif()

set(XPU_HOST_CXX_FLAGS)

# Find SYCL library.
find_package(SYCLToolkit REQUIRED)
if(NOT SYCL_FOUND)
  set(PYTORCH_FOUND_XPU FALSE)
  # Exit early to avoid populating XPU_HOST_CXX_FLAGS.
  return()
endif()
set(PYTORCH_FOUND_XPU TRUE)

# SYCL library interface
add_library(torch::sycl INTERFACE IMPORTED)

set_property(
    TARGET torch::sycl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    ${SYCL_INCLUDE_DIR})
set_property(
    TARGET torch::sycl PROPERTY INTERFACE_LINK_LIBRARIES
    ${SYCL_LIBRARY})

# libsycl.so transitively pulls libur_loader.so.0, which has DT_NEEDED libz.so.1.
# libz is not a SYCL toolkit artifact and has no find-module path through us, so
# propagate an rpath-link entry to the Python prefix's lib dir (where libz lives
# in conda envs). Conda's compat linker has a narrow sysroot view and won't find
# libz there otherwise. ELF/GNU-ld only.
if(NOT WIN32 AND NOT APPLE AND Python_EXECUTABLE)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE _xpu_py_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_xpu_py_prefix AND EXISTS "${_xpu_py_prefix}/lib")
    set_property(TARGET torch::sycl APPEND PROPERTY
      INTERFACE_LINK_OPTIONS "LINKER:-rpath-link,${_xpu_py_prefix}/lib")
  endif()
endif()

# xpurt
add_library(torch::xpurt INTERFACE IMPORTED)
set_property(
    TARGET torch::xpurt PROPERTY INTERFACE_LINK_LIBRARIES
    torch::sycl)

# setting xpu arch flags
torch_xpu_get_arch_list(XPU_ARCH_FLAGS)
# propagate to torch-xpu-ops
set(TORCH_XPU_ARCH_LIST ${XPU_ARCH_FLAGS})

# Ensure SYCL device code compiles with C++20 (matching CMAKE_CXX_STANDARD).
# SYCL_FLAGS flows into SYCL_COMPILE_FLAGS in torch-xpu-ops' BuildFlags.cmake
# and is passed directly to icpx on the device compilation command line.
list(APPEND SYCL_FLAGS -std=c++20)

# Ensure USE_XPU is enabled.
string(APPEND XPU_HOST_CXX_FLAGS " -DUSE_XPU")
string(APPEND XPU_HOST_CXX_FLAGS " -DSYCL_COMPILER_VERSION=${SYCL_COMPILER_VERSION}")

if(DEFINED ENV{XPU_ENABLE_KINETO})
  set(XPU_ENABLE_KINETO TRUE)
else()
  set(XPU_ENABLE_KINETO FALSE)
endif()

if(WIN32)
  if(${SYCL_COMPILER_VERSION} GREATER_EQUAL 20250101)
    set(XPU_ENABLE_KINETO TRUE)
  endif()
else()
  set(XPU_ENABLE_KINETO TRUE)
endif()