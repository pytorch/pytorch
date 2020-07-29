#[=======================================================================[.rst:
FindUCX
-------

Finds the UCX libraries.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``UCX_FOUND``
  True if the system has the UCX libraries.
``UCX_INCLUDE_DIR``
  Include directories needed to use UCX.
``UCX_LIBRARIES``
  Libraries needed to link to UCX.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``UCX_INCLUDE_DIR``
``UCX_LIBRARIES``

#]=======================================================================]

if (UCX_INCLUDE_DIR AND UCX_LIBRARIES)

  # in cache already
  SET(UCX_FOUND TRUE)

else (UCX_INCLUDE_DIR AND UCX_LIBRARIES)
 
    FIND_PATH(UCX_INCLUDE_DIR ucp/api/ucp.h
              HINTS
              ${HPCX_UCX_DIR}/include
              ${UCX_HOME}/include
              PATHS
              /usr/include/
              /usr/local/include/
              )

    FIND_LIBRARY(UCX_LIBRARIES NAMES ucp uct ucs ucm
                 HINTS
                 ${HPCX_UCX_DIR}/lib
                 ${UCX_HOME}/lib
                 PATHS
                 /usr/lib
                 /usr/local/lib
                 )

  if (UCX_INCLUDE_DIR AND UCX_LIBRARIES)
     set(UCX_FOUND TRUE)
  endif (UCX_INCLUDE_DIR AND UCX_LIBRARIES)

  if (UCX_FOUND)
    if (NOT UCX_FIND_QUIETLY)
        message(STATUS "Found UCX: ${UCX_LIBRARIES}")
    endif (NOT UCX_FIND_QUIETLY)
  else (UCX_FOUND)
    if (UCX_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find UCX")
    endif (UCX_FIND_REQUIRED)
  endif (UCX_FOUND)

  MARK_AS_ADVANCED(UCX_INCLUDE_DIR UCX_LIBRARIES)

endif (UCX_INCLUDE_DIR AND UCX_LIBRARIES)