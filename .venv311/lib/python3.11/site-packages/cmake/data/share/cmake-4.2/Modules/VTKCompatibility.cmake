# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Not needed for "modern" VTK.
if (EXISTS "${VTK_SOURCE_DIR}/CMake/vtkModule.cmake")
  return ()
endif ()

if(APPLE)
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_C_CREATE_SHARED_LIBRARY}")
  set(CMAKE_CXX_CREATE_SHARED_MODULE "${CMAKE_C_CREATE_SHARED_MODULE}")
  string( REGEX REPLACE "CMAKE_C_COMPILER"
    CMAKE_CXX_COMPILER CMAKE_CXX_CREATE_SHARED_MODULE
    "${CMAKE_CXX_CREATE_SHARED_MODULE}")
  string( REGEX REPLACE "CMAKE_C_COMPILER"
    CMAKE_CXX_COMPILER CMAKE_CXX_CREATE_SHARED_LIBRARY
    "${CMAKE_CXX_CREATE_SHARED_LIBRARY}")
endif()

set(VTKFTGL_BINARY_DIR "${VTK_BINARY_DIR}/Utilities/ftgl"
  CACHE INTERNAL "")
set(VTKFREETYPE_BINARY_DIR "${VTK_BINARY_DIR}/Utilities/freetype"
  CACHE INTERNAL "")
set(VTKFTGL_SOURCE_DIR "${VTK_SOURCE_DIR}/Utilities/ftgl"
  CACHE INTERNAL "")
set(VTKFREETYPE_SOURCE_DIR "${VTK_SOURCE_DIR}/Utilities/freetype"
  CACHE INTERNAL "")

set(VTK_GLEXT_FILE "${VTK_SOURCE_DIR}/Utilities/ParseOGLExt/headers/glext.h"
  CACHE FILEPATH
  "Location of the OpenGL extensions header file (glext.h).")
set(VTK_GLXEXT_FILE
  "${VTK_SOURCE_DIR}/Utilities/ParseOGLExt/headers/glxext.h" CACHE FILEPATH
  "Location of the GLX extensions header file (glxext.h).")
set(VTK_WGLEXT_FILE "${VTK_SOURCE_DIR}/Utilities/ParseOGLExt/headers/wglext.h"
  CACHE FILEPATH
  "Location of the WGL extensions header file (wglext.h).")

# work around an old bug in VTK
set(TIFF_RIGHT_VERSION 1)

# for very old VTK (versions prior to 4.2)
macro(SOURCE_FILES)
  message (FATAL_ERROR "You are trying to build a very old version of VTK (prior to VTK 4.2). To do this you need to use CMake 2.0 as it was the last version of CMake to support VTK 4.0.")
endmacro()
