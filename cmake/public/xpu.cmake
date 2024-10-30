# ---[ xpu

# Poor man's include guard
if(TARGET torch::xpurt)
  return()
endif()

# Find SYCL library.
find_package(SYCLToolkit REQUIRED)
if(NOT SYCL_FOUND)
  set(PYTORCH_FOUND_XPU FALSE)
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

# xpurt
add_library(torch::xpurt INTERFACE IMPORTED)
set_property(
    TARGET torch::xpurt PROPERTY INTERFACE_LINK_LIBRARIES
    torch::sycl)

# setting xpu arch flags
torch_xpu_get_arch_list(XPU_ARCH_FLAGS)
# propagate to torch-xpu-ops
set(TORCH_XPU_ARCH_LIST ${XPU_ARCH_FLAGS})

if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND SYCL_COMPILER_VERSION VERSION_LESS_EQUAL 20240703)
  # for ABI compatibility on Linux
  string(APPEND XPU_EXTRA_FLAGS " -D__INTEL_PREVIEW_BREAKING_CHANGES")
  set(DNNL_CXX_FLAGS "-DCMAKE_CXX_FLAGS=-fpreview-breaking-changes")
endif()
