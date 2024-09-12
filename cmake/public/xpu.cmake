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

#find oneDPL Library
find_package(oneDPL REQUIRED)
if(oneDPL_FOUND)
  set(ONEDPL_DEVICE_LOWER_BOUND_WORKS TRUE)
  return()
endif()
set(ONEDPL_DEVICE_LOWER_BOUND_WORKS FALSE)

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
