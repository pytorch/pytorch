# Finds the XCCL libraries.

# This will define the following variables:

# ``XCCL_FOUND``
#  True if the system has the XCCL libraries.
# ``XCCL_INCLUDE_DIR``
#  Include directories needed to use XCCL.
# ``XCCL_LIBRARIES``
#  Libraries needed to link to XCCL.

# Cache Variables

# The following cache variables may also be set:

# ``XCCL_INCLUDE_DIR``
# ``XCCL_LIBRARIES``

if (XCCL_INCLUDE_DIR AND XCCL_LIBRARIES)

  # in cache already
  SET(XCCL_FOUND TRUE)

else (XCCL_INCLUDE_DIR AND XCCL_LIBRARIES)

    FIND_PATH(XCCL_INCLUDE_DIR ucp/api/ucp.h
              HINTS
              ${HPCX_XCCL_DIR}/include
              ${XCCL_HOME}/include
              PATHS
              /usr/include/
              /usr/local/include/
              )

    FIND_LIBRARY(XCCL_LIBRARIES NAMES ucp uct ucs ucm
                 HINTS
                 ${HPCX_XCCL_DIR}/lib
                 ${XCCL_HOME}/lib
                 PATHS
                 /usr/lib
                 /usr/local/lib
                 )

  if (XCCL_INCLUDE_DIR AND XCCL_LIBRARIES)
     set(XCCL_FOUND TRUE)
  endif (XCCL_INCLUDE_DIR AND XCCL_LIBRARIES)

  if (XCCL_FOUND)
    if (NOT XCCL_FIND_QUIETLY)
        message(STATUS "Found XCCL: ${XCCL_LIBRARIES}")
    endif (NOT XCCL_FIND_QUIETLY)
  else (XCCL_FOUND)
    if (XCCL_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find XCCL")
    endif (XCCL_FIND_REQUIRED)
  endif (XCCL_FOUND)

  MARK_AS_ADVANCED(XCCL_INCLUDE_DIR XCCL_LIBRARIES)

endif (XCCL_INCLUDE_DIR AND XCCL_LIBRARIES)
