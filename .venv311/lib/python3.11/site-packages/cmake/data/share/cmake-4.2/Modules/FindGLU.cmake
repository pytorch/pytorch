# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Use of this file is deprecated, and is here for backwards compatibility with CMake 1.4
# GLU library is now found by FindOpenGL.cmake
#

message(STATUS
  "WARNING: you are using the obsolete 'GLU' package, please use 'OpenGL' instead")

include(${CMAKE_CURRENT_LIST_DIR}/FindOpenGL.cmake)

if (OPENGL_GLU_FOUND)
  set (GLU_LIBRARY ${OPENGL_LIBRARIES})
  set (GLU_INCLUDE_PATH ${OPENGL_INCLUDE_DIR})
endif ()
