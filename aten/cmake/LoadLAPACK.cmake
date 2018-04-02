find_package(LAPACK)
if(LAPACK_FOUND)
  # This variable is used in a cmakedefine
  set(USE_LAPACK 1)
endif()
