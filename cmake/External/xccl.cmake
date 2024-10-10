if(NOT __XCCL_INCLUDED)
  set(__XCCL_INCLUDED TRUE)

  # XCCL_ROOT, XCCL_LIBRARY_DIR, XCCL_INCLUDE_DIR are handled by FindXCCL.cmake.
  find_package(XCCL REQUIRED)
  if(XCCL_FOUND)
    add_library(torch::xccl INTERFACE IMPORTED)
    set_property(
      TARGET torch::xccl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${XCCL_INCLUDE_DIR})
    set_property(
      TARGET torch::xccl PROPERTY INTERFACE_LINK_LIBRARIES
      ${XCCL_LIBRARY})
  endif()
endif()
