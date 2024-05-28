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

# Find PTI-sdk library.
set(Pti_DIR $ENV{Pti_DIR} CACHE PATH "Installed PTI-sdk's cmake directory for find package.")
find_package(Pti REQUIRED)
if(NOT TARGET Pti::pti_view)
  set(PYTORCH_FOUND_XPU_PTI FALSE)
  return()
endif()
set(PYTORCH_FOUND_XPU_PTI TRUE)
