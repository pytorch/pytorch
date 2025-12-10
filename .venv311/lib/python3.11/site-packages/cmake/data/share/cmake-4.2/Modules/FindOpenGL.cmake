# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenGL
----------

Finds the OpenGL and OpenGL Utility Library (GLU), for using OpenGL in a
CMake project:

.. code-block:: cmake

  find_package(OpenGL [COMPONENTS <components>...] [...])

OpenGL (Open Graphics Library) is a cross-platform API for rendering 2D and
3D graphics.  It is widely used in CAD, games, and visualization software.

* *GL* refers to the core OpenGL library, which provides the fundamental
  graphics rendering API.

* *GLU* (OpenGL Utility Library) is a companion library that offers utility
  functions built on top of OpenGL, such as tessellation and more complex
  shape drawing.

.. versionchanged:: 3.2
  X11 is no longer added as a dependency on Unix/Linux systems.

.. versionadded:: 3.10
  GLVND (GL Vendor-Neutral Dispatch library) support on Linux.  See the
  :ref:`Linux Specific` section below.

Components
^^^^^^^^^^

This module supports optional components which can be specified with the
:command:`find_package` command:

.. code-block:: cmake

  find_package(OpenGL [COMPONENTS <components>...])

Supported components are:

``EGL``
  .. versionadded:: 3.10

  The EGL interface between OpenGL, OpenGL ES and the underlying windowing
  system.

``GLX``
  .. versionadded:: 3.10

  An extension to X that interfaces OpenGL, OpenGL ES with X window system.

``OpenGL``
  .. versionadded:: 3.10

  The cross platform API for 3D graphics.

``GLES2``
  .. versionadded:: 3.27

  A subset of OpenGL API for embedded systems with limited capabilities.

``GLES3``
  .. versionadded:: 3.27

  A subset of OpenGL API for embedded systems with more capabilities.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``OpenGL::GL``
  .. versionadded:: 3.8

  Target encapsulating the usage requirements of platform-specific OpenGL
  libraries, available if OpenGL is found.

``OpenGL::GLU``
  .. versionadded:: 3.8

  Target encapsulating the OpenGL Utility Library (GLU) usage requirements,
  available if GLU is found.

Additionally, the following GLVND-specific library imported targets are
provided:

``OpenGL::OpenGL``
  .. versionadded:: 3.10

  Target encapsulating the libOpenGL usage requirements, available if
  system is GLVND-based and OpenGL is found.

``OpenGL::GLX``
  .. versionadded:: 3.10

  Target encapsulating the usage requirements of the OpenGL Extension to the
  the X Window System (GLX), available if OpenGL and GLX are found.

``OpenGL::EGL``
  .. versionadded:: 3.10

  Target encapsulating the EGL usage requirements, available if OpenGL and EGL
  are found.

``OpenGL::GLES2``
  .. versionadded:: 3.27

  Target encapsulating the GLES2 usage requirements, available if OpenGL and
  GLES2 are found.

``OpenGL::GLES3``
  .. versionadded:: 3.27

  Target encapsulating the GLES3 usage requirements, available if OpenGL and
  GLES3 are found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OpenGL_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether OpenGL and all requested components were found.

``OPENGL_XMESA_FOUND``
  Boolean indicating whether OpenGL XMESA was found.

``OPENGL_GLU_FOUND``
  Boolean indicating whether GLU was found.

``OpenGL_OpenGL_FOUND``
  .. versionadded:: 3.10

  Boolean indicating whether the GLVND OpenGL library was found.

``OpenGL_GLX_FOUND``
  .. versionadded:: 3.10

  Boolean indicating whether GLVND GLX was found.

``OpenGL_EGL_FOUND``
  .. versionadded:: 3.10

  Boolean indicating whether GLVND EGL was found.

``OpenGL_GLES2_FOUND``
  .. versionadded:: 3.27

  Boolean indicating whether GLES2 was found.

``OpenGL_GLES3_FOUND``
  .. versionadded:: 3.27

  Boolean indicating whether GLES3 was found.

``OPENGL_INCLUDE_DIRS``
  .. versionadded:: 3.29

  Paths to the OpenGL include directories.

``OPENGL_EGL_INCLUDE_DIRS``
  .. versionadded:: 3.10

  Path to the EGL include directory.

``OPENGL_LIBRARIES``
  Paths to the OpenGL library, windowing system libraries, and GLU libraries.
  On Linux, this assumes GLX and is never correct for EGL-based targets.
  Clients are encouraged to use the ``OpenGL::*`` imported targets instead.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OPENGL_INCLUDE_DIR``
  The path to the OpenGL include directory.
  The ``OPENGL_INCLUDE_DIRS`` variable is preferred.

``OPENGL_GLU_INCLUDE_DIR``
  .. versionadded:: 3.29

  Path to the OpenGL GLU include directory.

``OPENGL_egl_LIBRARY``
  .. versionadded:: 3.10

  Path to the GLVND EGL library.

``OPENGL_glu_LIBRARY``
  Path to the GLU library.

``OPENGL_glx_LIBRARY``
  .. versionadded:: 3.10

  Path to the GLVND GLX library.

``OPENGL_opengl_LIBRARY``
  .. versionadded:: 3.10

  Path to the GLVND OpenGL library

``OPENGL_gl_LIBRARY``
  Path to the OpenGL library.

``OPENGL_gles2_LIBRARY``
  .. versionadded:: 3.27

  Path to the OpenGL GLES2 library.

``OPENGL_gles3_LIBRARY``
  .. versionadded:: 3.27

  Path to the OpenGL GLES3 library.

Hints
^^^^^

This module accepts the following variables:

``OpenGL_GL_PREFERENCE``
  .. versionadded:: 3.10

  This variable is supported on Linux systems to specify the preferred way to
  provide legacy GL interfaces in case multiple choices are available.  The
  value may be one of:

  ``GLVND``
    If the GLVND OpenGL and GLX libraries are available, prefer them.
    This forces ``OPENGL_gl_LIBRARY`` to be empty.

    .. versionchanged:: 3.11
      This is the default, unless policy :policy:`CMP0072` is set to ``OLD``
      and no components are requested (since components
      correspond to GLVND libraries).

  ``LEGACY``
    Prefer to use the legacy libGL library, if available.

.. _`Linux Specific`:

Linux-specific
^^^^^^^^^^^^^^

Some Linux systems utilize GLVND as a new ABI for OpenGL.  GLVND separates
context libraries from OpenGL itself; OpenGL lives in "libOpenGL", and
contexts are defined in "libGLX" or "libEGL".  GLVND is currently the only way
to get OpenGL 3+ functionality via EGL in a manner portable across vendors.
Projects may use GLVND explicitly with target ``OpenGL::OpenGL`` and either
``OpenGL::GLX`` or ``OpenGL::EGL``.

Projects may use the ``OpenGL::GL`` target (or ``OPENGL_LIBRARIES`` variable)
to use legacy GL interfaces.  These will use the legacy GL library located
by ``OPENGL_gl_LIBRARY``, if available.  If ``OPENGL_gl_LIBRARY`` is empty or
not found and GLVND is available, the ``OpenGL::GL`` target will use GLVND
``OpenGL::OpenGL`` and ``OpenGL::GLX`` (and the ``OPENGL_LIBRARIES``
variable will use the corresponding libraries).  Thus, for non-EGL-based
Linux targets, the ``OpenGL::GL`` target is most portable.

The ``OpenGL_GL_PREFERENCE`` variable may be set to specify the preferred way
to provide legacy GL interfaces in case multiple choices are available.

For EGL targets the client must rely on GLVND support on the user's system.
Linking should use the ``OpenGL::OpenGL OpenGL::EGL`` targets.  Using GLES*
libraries is theoretically possible in place of ``OpenGL::OpenGL``, but this
module does not currently support that; contributions welcome.

``OPENGL_egl_LIBRARY`` and ``OPENGL_EGL_INCLUDE_DIRS`` are defined in the case of
GLVND.  For non-GLVND Linux and other systems these are left undefined.

macOS-Specific
^^^^^^^^^^^^^^

On macOS this module defaults to using the macOS-native framework
version of OpenGL.  To use the X11 version of OpenGL on macOS, one
can disable searching of frameworks using the :variable:`CMAKE_FIND_FRAMEWORK`
variable.  For example:

.. code-block:: cmake

  find_package(X11)
  if(APPLE AND X11_FOUND)
    set(CMAKE_FIND_FRAMEWORK NEVER)
    find_package(OpenGL)
    unset(CMAKE_FIND_FRAMEWORK)
  else()
    find_package(OpenGL)
  endif()

An end user building this project may need to point CMake at their
X11 installation, e.g., with ``-DOpenGL_ROOT=/opt/X11``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OPENGL_FOUND``
  .. deprecated:: 4.2
    Use ``OpenGL_FOUND``, which has the same value.

  Boolean indicating whether OpenGL and all requested components were found.

Examples
^^^^^^^^

Finding the OpenGL library and linking it to a project target:

.. code-block:: cmake

  find_package(OpenGL)
  target_link_libraries(project_target PRIVATE OpenGL::OpenGL)

See Also
^^^^^^^^

* The :module:`FindGLEW` module to find OpenGL Extension Wrangler Library
  (GLEW).
* The :module:`FindGLUT` module to find OpenGL Utility Toolkit (GLUT)
  library.
* The :module:`FindVulkan` module to find Vulkan graphics API.
#]=======================================================================]

set(_OpenGL_REQUIRED_VARS OPENGL_gl_LIBRARY)

# Provide OPENGL_USE_<C> variables for each component.
foreach(component ${OpenGL_FIND_COMPONENTS})
  string(TOUPPER ${component} _COMPONENT)
  set(OPENGL_USE_${_COMPONENT} 1)
endforeach()

set(_OpenGL_CACHE_VARS)

if (WIN32)

  if(BORLAND)
    set (OPENGL_gl_LIBRARY import32 CACHE STRING "OpenGL library for win32")
    set (OPENGL_glu_LIBRARY import32 CACHE STRING "GLU library for win32")
  else()
    set (OPENGL_gl_LIBRARY opengl32 CACHE STRING "OpenGL library for win32")
    set (OPENGL_glu_LIBRARY glu32 CACHE STRING "GLU library for win32")
  endif()

  list(APPEND _OpenGL_CACHE_VARS
    OPENGL_gl_LIBRARY
    OPENGL_glu_LIBRARY
    )
elseif (APPLE)
  # The OpenGL.framework provides both gl and glu in OpenGL
  # XQuartz provides libgl and libglu
  find_library(OPENGL_gl_LIBRARY NAMES OpenGL GL DOC
    "OpenGL GL library")
  find_library(OPENGL_glu_LIBRARY NAMES OpenGL GLU DOC
    "OpenGL GLU library")
  find_path(OPENGL_INCLUDE_DIR NAMES OpenGL/gl.h GL/gl.h DOC
    "Include for OpenGL")
  find_path(OPENGL_GLU_INCLUDE_DIR NAMES OpenGL/glu.h GL/glu.h DOC
    "Include for the OpenGL GLU library")
  list(APPEND _OpenGL_REQUIRED_VARS OPENGL_INCLUDE_DIR)

  list(APPEND _OpenGL_CACHE_VARS
    OPENGL_INCLUDE_DIR
    OPENGL_GLU_INCLUDE_DIR
    OPENGL_gl_LIBRARY
    OPENGL_glu_LIBRARY
    )
else()
  if (CMAKE_ANDROID_NDK)
    set(_OPENGL_INCLUDE_PATH ${CMAKE_ANDROID_NDK}/sysroot/usr/include)
    set(_OPENGL_LIB_PATH ${CMAKE_ANDROID_NDK}/platforms/android-${CMAKE_SYSTEM_VERSION}/arch-${CMAKE_ANDROID_ARCH}/usr/lib)
  elseif (CMAKE_SYSTEM_NAME MATCHES "HP-UX")
    # Handle HP-UX cases where we only want to find OpenGL in either hpux64
    # or hpux32 depending on if we're doing a 64 bit build.
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
      set(_OPENGL_LIB_PATH
        /opt/graphics/OpenGL/lib/hpux32/)
    else()
      set(_OPENGL_LIB_PATH
        /opt/graphics/OpenGL/lib/hpux64/
        /opt/graphics/OpenGL/lib/pa20_64)
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL Haiku)
    set(_OPENGL_LIB_PATH
      /boot/develop/lib/x86)
    set(_OPENGL_INCLUDE_PATH
      /boot/develop/headers/os/opengl)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # CMake doesn't support arbitrary globs in search paths.
    file(GLOB _OPENGL_LIB_PATH
      # The NVidia driver installation tool on Linux installs libraries to a
      # `nvidia-<version>` subdirectory.
      "/usr/lib/nvidia-*"
      "/usr/lib32/nvidia-*")
  endif()

  # The first line below is to make sure that the proper headers
  # are used on a Linux machine with the NVidia drivers installed.
  # They replace Mesa with NVidia's own library but normally do not
  # install headers and that causes the linking to
  # fail since the compiler finds the Mesa headers but NVidia's library.
  # Make sure the NVIDIA directory comes BEFORE the others.
  #  - Atanas Georgiev <atanas@cs.columbia.edu>
  find_path(OPENGL_INCLUDE_DIR GL/gl.h
    /usr/share/doc/NVIDIA_GLX-1.0/include
    /usr/openwin/share/include
    /opt/graphics/OpenGL/include
    ${_OPENGL_INCLUDE_PATH}
  )
  find_path(OPENGL_GLX_INCLUDE_DIR GL/glx.h ${_OPENGL_INCLUDE_PATH})
  find_path(OPENGL_EGL_INCLUDE_DIR EGL/egl.h ${_OPENGL_INCLUDE_PATH})
  find_path(OPENGL_GLES2_INCLUDE_DIR GLES2/gl2.h ${_OPENGL_INCLUDE_PATH})
  find_path(OPENGL_GLES3_INCLUDE_DIR GLES3/gl3.h ${_OPENGL_INCLUDE_PATH})
  find_path(OPENGL_xmesa_INCLUDE_DIR GL/xmesa.h
    /usr/share/doc/NVIDIA_GLX-1.0/include
    /usr/openwin/share/include
    /opt/graphics/OpenGL/include
  )

  find_path(OPENGL_GLU_INCLUDE_DIR GL/glu.h ${_OPENGL_INCLUDE_PATH})

  list(APPEND _OpenGL_CACHE_VARS
    OPENGL_INCLUDE_DIR
    OPENGL_GLX_INCLUDE_DIR
    OPENGL_EGL_INCLUDE_DIR
    OPENGL_GLES2_INCLUDE_DIR
    OPENGL_GLES3_INCLUDE_DIR
    OPENGL_xmesa_INCLUDE_DIR
    OPENGL_GLU_INCLUDE_DIR
    )

  # Search for the GLVND libraries.  We do this regardless of COMPONENTS; we'll
  # take into account the COMPONENTS logic later.
  find_library(OPENGL_opengl_LIBRARY
    NAMES OpenGL
    PATHS ${_OPENGL_LIB_PATH}
  )

  find_library(OPENGL_glx_LIBRARY
    NAMES GLX
    PATHS ${_OPENGL_LIB_PATH}
    PATH_SUFFIXES libglvnd
  )

  find_library(OPENGL_egl_LIBRARY
    NAMES EGL
    PATHS ${_OPENGL_LIB_PATH}
    PATH_SUFFIXES libglvnd
  )

  find_library(OPENGL_gles2_LIBRARY
    NAMES GLESv2
    PATHS ${_OPENGL_LIB_PATH}
  )

  find_library(OPENGL_gles3_LIBRARY
    NAMES GLESv3
          GLESv2 # mesa provides only libGLESv2
    PATHS ${_OPENGL_LIB_PATH}
  )

  find_library(OPENGL_glu_LIBRARY
    NAMES GLU MesaGLU
    PATHS ${OPENGL_gl_LIBRARY}
          /opt/graphics/OpenGL/lib
          /usr/openwin/lib
          /usr/shlib
  )

  list(APPEND _OpenGL_CACHE_VARS
    OPENGL_opengl_LIBRARY
    OPENGL_glx_LIBRARY
    OPENGL_egl_LIBRARY
    OPENGL_gles2_LIBRARY
    OPENGL_gles3_LIBRARY
    OPENGL_glu_LIBRARY
    )

  set(_OpenGL_GL_POLICY_WARN 0)
  if(NOT DEFINED OpenGL_GL_PREFERENCE)
    set(OpenGL_GL_PREFERENCE "")
  endif()
  if(NOT OpenGL_GL_PREFERENCE STREQUAL "")
    # A preference has been explicitly specified.
    if(NOT OpenGL_GL_PREFERENCE MATCHES "^(GLVND|LEGACY)$")
      message(FATAL_ERROR
        "OpenGL_GL_PREFERENCE value '${OpenGL_GL_PREFERENCE}' not recognized.  "
        "Allowed values are 'GLVND' and 'LEGACY'."
        )
    endif()
  elseif(OpenGL_FIND_COMPONENTS)
    # No preference was explicitly specified, but the caller did request
    # at least one GLVND component.  Prefer GLVND for legacy GL.
    set(OpenGL_GL_PREFERENCE "GLVND")
  else()
    # No preference was explicitly specified and no GLVND components were
    # requested.  Use a policy to choose the default.
    cmake_policy(GET CMP0072 _OpenGL_GL_POLICY)
    if("x${_OpenGL_GL_POLICY}x" STREQUAL "xNEWx")
      set(OpenGL_GL_PREFERENCE "GLVND")
    else()
      set(OpenGL_GL_PREFERENCE "LEGACY")
      if("x${_OpenGL_GL_POLICY}x" STREQUAL "xx")
        set(_OpenGL_GL_POLICY_WARN 1)
      endif()
    endif()
    unset(_OpenGL_GL_POLICY)
  endif()

  if("x${OpenGL_GL_PREFERENCE}x" STREQUAL "xGLVNDx" AND OPENGL_opengl_LIBRARY AND OPENGL_glx_LIBRARY)
    # We can provide legacy GL using GLVND libraries.
    # Do not use any legacy GL library.
    set(OPENGL_gl_LIBRARY "")
  else()
    # We cannot provide legacy GL using GLVND libraries.
    # Search for the legacy GL library.
    find_library(OPENGL_gl_LIBRARY
      NAMES GL MesaGL
      PATHS /opt/graphics/OpenGL/lib
            /usr/openwin/lib
            /usr/shlib
            ${_OPENGL_LIB_PATH}
      PATH_SUFFIXES libglvnd
      )
    list(APPEND _OpenGL_CACHE_VARS OPENGL_gl_LIBRARY)
  endif()

  if(_OpenGL_GL_POLICY_WARN AND OPENGL_gl_LIBRARY AND OPENGL_opengl_LIBRARY AND OPENGL_glx_LIBRARY)
    cmake_policy(GET_WARNING CMP0072 _cmp0072_warning)
    message(AUTHOR_WARNING
      "${_cmp0072_warning}\n"
      "FindOpenGL found both a legacy GL library:\n"
      "  OPENGL_gl_LIBRARY: ${OPENGL_gl_LIBRARY}\n"
      "and GLVND libraries for OpenGL and GLX:\n"
      "  OPENGL_opengl_LIBRARY: ${OPENGL_opengl_LIBRARY}\n"
      "  OPENGL_glx_LIBRARY: ${OPENGL_glx_LIBRARY}\n"
      "OpenGL_GL_PREFERENCE has not been set to \"GLVND\" or \"LEGACY\", so for "
      "compatibility with CMake 3.10 and below the legacy GL library will be used."
      )
  endif()
  unset(_OpenGL_GL_POLICY_WARN)

  # FPHSA cannot handle "this OR that is required", so we conditionally set what
  # it must look for.  First clear any previous config we might have done:
  set(_OpenGL_REQUIRED_VARS)

  # now we append the libraries as appropriate.  The complicated logic
  # basically comes down to "use libOpenGL when we can, and add in specific
  # context mechanisms when requested, or we need them to preserve the previous
  # default where glx is always available."
  if((NOT OPENGL_USE_EGL AND
      NOT OPENGL_opengl_LIBRARY AND
          OPENGL_glx_LIBRARY AND
      NOT OPENGL_gl_LIBRARY) OR
     (NOT OPENGL_USE_EGL AND
      NOT OPENGL_USE_GLES3 AND
      NOT OPENGL_USE_GLES2 AND
      NOT OPENGL_glx_LIBRARY AND
      NOT OPENGL_gl_LIBRARY) OR
     (NOT OPENGL_USE_EGL AND
          OPENGL_opengl_LIBRARY AND
          OPENGL_glx_LIBRARY) OR
     (NOT OPENGL_USE_GLES3 AND
      NOT OPENGL_USE_GLES2 AND
          OPENGL_USE_EGL))
    list(APPEND _OpenGL_REQUIRED_VARS OPENGL_opengl_LIBRARY)
  endif()

  # GLVND GLX library.  Preferred when available.
  if((NOT OPENGL_USE_OPENGL AND
      NOT OPENGL_USE_GLX AND
      NOT OPENGL_USE_EGL AND
      NOT OPENGL_USE_GLES3 AND
      NOT OPENGL_USE_GLES2 AND
      NOT OPENGL_glx_LIBRARY AND
      NOT OPENGL_gl_LIBRARY) OR
     (    OPENGL_USE_GLX AND
      NOT OPENGL_USE_EGL AND
      NOT OPENGL_USE_GLES3 AND
      NOT OPENGL_USE_GLES2 AND
      NOT OPENGL_glx_LIBRARY AND
      NOT OPENGL_gl_LIBRARY) OR
     (NOT OPENGL_USE_EGL AND
      NOT OPENGL_USE_GLES3 AND
      NOT OPENGL_USE_GLES2 AND
          OPENGL_opengl_LIBRARY AND
          OPENGL_glx_LIBRARY) OR
     (OPENGL_USE_GLX AND OPENGL_USE_EGL))
    list(APPEND _OpenGL_REQUIRED_VARS OPENGL_glx_LIBRARY)
  endif()

  # GLVND EGL library.
  if(OPENGL_USE_EGL)
    list(APPEND _OpenGL_REQUIRED_VARS OPENGL_egl_LIBRARY)
  endif()

  # GLVND GLES2 library.
  if(OPENGL_USE_GLES2)
    list(APPEND _OpenGL_REQUIRED_VARS OPENGL_gles2_LIBRARY)
  endif()

  # GLVND GLES3 library.
  if(OPENGL_USE_GLES3)
    list(APPEND _OpenGL_REQUIRED_VARS OPENGL_gles3_LIBRARY)
  endif()

  # Old-style "libGL" library: used as a fallback when GLVND isn't available.
  if((NOT OPENGL_USE_EGL AND
      NOT OPENGL_opengl_LIBRARY AND
          OPENGL_glx_LIBRARY AND
          OPENGL_gl_LIBRARY) OR
     (NOT OPENGL_USE_EGL AND
      NOT OPENGL_glx_LIBRARY AND
          OPENGL_gl_LIBRARY))
    list(PREPEND _OpenGL_REQUIRED_VARS OPENGL_gl_LIBRARY)
  endif()

  # We always need the 'gl.h' include dir.
  if(OPENGL_USE_EGL)
    list(APPEND _OpenGL_REQUIRED_VARS OPENGL_EGL_INCLUDE_DIR)
  else()
    list(APPEND _OpenGL_REQUIRED_VARS OPENGL_INCLUDE_DIR)
  endif()

  unset(_OPENGL_INCLUDE_PATH)
  unset(_OPENGL_LIB_PATH)

  find_library(OPENGL_glu_LIBRARY
    NAMES GLU MesaGLU
    PATHS ${OPENGL_gl_LIBRARY}
          /opt/graphics/OpenGL/lib
          /usr/openwin/lib
          /usr/shlib
  )
endif ()

if(OPENGL_xmesa_INCLUDE_DIR)
  set( OPENGL_XMESA_FOUND "YES" )
else()
  set( OPENGL_XMESA_FOUND "NO" )
endif()

if(OPENGL_glu_LIBRARY AND (WIN32 OR OPENGL_GLU_INCLUDE_DIR))
  set( OPENGL_GLU_FOUND "YES" )
else()
  set( OPENGL_GLU_FOUND "NO" )
endif()

# OpenGL_OpenGL_FOUND is a bit unique in that it is okay if /either/ libOpenGL
# or libGL is found.
# Using libGL with libEGL is never okay, though; we handle that case later.
if(NOT OPENGL_opengl_LIBRARY AND NOT OPENGL_gl_LIBRARY)
  set(OpenGL_OpenGL_FOUND FALSE)
else()
  set(OpenGL_OpenGL_FOUND TRUE)
endif()

if(OPENGL_glx_LIBRARY AND OPENGL_GLX_INCLUDE_DIR)
  set(OpenGL_GLX_FOUND TRUE)
else()
  set(OpenGL_GLX_FOUND FALSE)
endif()

if(OPENGL_egl_LIBRARY AND OPENGL_EGL_INCLUDE_DIR)
  set(OpenGL_EGL_FOUND TRUE)
else()
  set(OpenGL_EGL_FOUND FALSE)
endif()

if(OPENGL_gles2_LIBRARY AND OPENGL_GLES2_INCLUDE_DIR)
  set(OpenGL_GLES2_FOUND TRUE)
else()
  set(OpenGL_GLES2_FOUND FALSE)
endif()

if(OPENGL_gles3_LIBRARY AND OPENGL_GLES3_INCLUDE_DIR)
  set(OpenGL_GLES3_FOUND TRUE)
else()
  set(OpenGL_GLES3_FOUND FALSE)
endif()

# User-visible names should be plural.
if(OPENGL_EGL_INCLUDE_DIR)
  set(OPENGL_EGL_INCLUDE_DIRS ${OPENGL_EGL_INCLUDE_DIR})
endif()

include(FindPackageHandleStandardArgs)
if (CMAKE_FIND_PACKAGE_NAME STREQUAL "GLU")
  # FindGLU include()'s this module. It's an old pattern, but rather than
  # trying to suppress this from outside the module (which is then sensitive to
  # the contents, detect the case in this module and suppress it explicitly.
  set(FPHSA_NAME_MISMATCHED 1)
endif ()
find_package_handle_standard_args(OpenGL REQUIRED_VARS ${_OpenGL_REQUIRED_VARS}
                                  HANDLE_COMPONENTS)
unset(FPHSA_NAME_MISMATCHED)
unset(_OpenGL_REQUIRED_VARS)

# OpenGL:: targets
if(OpenGL_FOUND)
  set(OPENGL_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR})

  # ::OpenGL is a GLVND library, and thus Linux-only: we don't bother checking
  # for a framework version of this library.
  if(OPENGL_opengl_LIBRARY AND NOT TARGET OpenGL::OpenGL)
    if(IS_ABSOLUTE "${OPENGL_opengl_LIBRARY}")
      add_library(OpenGL::OpenGL UNKNOWN IMPORTED)
      set_target_properties(OpenGL::OpenGL PROPERTIES IMPORTED_LOCATION
                            "${OPENGL_opengl_LIBRARY}")
    else()
      add_library(OpenGL::OpenGL INTERFACE IMPORTED)
      set_target_properties(OpenGL::OpenGL PROPERTIES IMPORTED_LIBNAME
                            "${OPENGL_opengl_LIBRARY}")
    endif()
    set_target_properties(OpenGL::OpenGL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                          "${OPENGL_INCLUDE_DIR}")
    set(_OpenGL_EGL_IMPL OpenGL::OpenGL)
  endif()

  # ::GLX is a GLVND library, and thus Linux-only: we don't bother checking
  # for a framework version of this library.
  if(OpenGL_GLX_FOUND AND NOT TARGET OpenGL::GLX AND TARGET OpenGL::OpenGL)
    if(IS_ABSOLUTE "${OPENGL_glx_LIBRARY}")
      add_library(OpenGL::GLX UNKNOWN IMPORTED)
      set_target_properties(OpenGL::GLX PROPERTIES IMPORTED_LOCATION
                            "${OPENGL_glx_LIBRARY}")
    else()
      add_library(OpenGL::GLX INTERFACE IMPORTED)
      set_target_properties(OpenGL::GLX PROPERTIES IMPORTED_LIBNAME
                            "${OPENGL_glx_LIBRARY}")
    endif()
    set_target_properties(OpenGL::GLX PROPERTIES INTERFACE_LINK_LIBRARIES
                          OpenGL::OpenGL)
    set_target_properties(OpenGL::GLX PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                          "${OPENGL_GLX_INCLUDE_DIR}")
    list(APPEND OPENGL_INCLUDE_DIRS ${OPENGL_GLX_INCLUDE_DIR})
  endif()

  # ::GLES2 is a GLVND library, and thus Linux-only: we don't bother checking
  # for a framework version of this library.
  if(OpenGL_GLES2_FOUND AND NOT TARGET OpenGL::GLES2)

    # Initialize target
    if(NOT OPENGL_gles2_LIBRARY)
      add_library(OpenGL::GLES2 INTERFACE IMPORTED)
    else()
      if(IS_ABSOLUTE "${OPENGL_gles2_LIBRARY}")
        add_library(OpenGL::GLES2 UNKNOWN IMPORTED)
        set_target_properties(OpenGL::GLES2 PROPERTIES
          IMPORTED_LOCATION "${OPENGL_gles2_LIBRARY}"
        )
      else()
        add_library(OpenGL::GLES2 INTERFACE IMPORTED)
        set_target_properties(OpenGL::GLES2 PROPERTIES
          IMPORTED_LIBNAME "${OPENGL_gles2_LIBRARY}"
        )
      endif()
    endif()

    # Attach target properties
    set_target_properties(OpenGL::GLES2
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
          "${OPENGL_GLES2_INCLUDE_DIR}"
    )
    list(APPEND OPENGL_INCLUDE_DIRS ${OPENGL_GLES2_INCLUDE_DIR})

    if (OPENGL_USE_GLES2)
      set(_OpenGL_EGL_IMPL OpenGL::GLES2)
    endif ()

  endif()

  # ::GLES3 is a GLVND library, and thus Linux-only: we don't bother checking
  # for a framework version of this library.
  if(OpenGL_GLES3_FOUND AND NOT TARGET OpenGL::GLES3)

    # Initialize target
    if(NOT OPENGL_gles3_LIBRARY)
      add_library(OpenGL::GLES3 INTERFACE IMPORTED)
    else()
      if(IS_ABSOLUTE "${OPENGL_gles3_LIBRARY}")
        add_library(OpenGL::GLES3 UNKNOWN IMPORTED)
        set_target_properties(OpenGL::GLES3 PROPERTIES
          IMPORTED_LOCATION "${OPENGL_gles3_LIBRARY}"
        )
      else()
        add_library(OpenGL::GLES3 INTERFACE IMPORTED)
        set_target_properties(OpenGL::GLES3 PROPERTIES
          IMPORTED_LIBNAME "${OPENGL_gles3_LIBRARY}"
        )
      endif()
    endif()

    # Attach target properties
    set_target_properties(OpenGL::GLES3 PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES
        "${OPENGL_GLES3_INCLUDE_DIR}"
    )
    list(APPEND OPENGL_INCLUDE_DIRS ${OPENGL_GLES3_INCLUDE_DIR})

    if (OPENGL_USE_GLES3)
      set(_OpenGL_EGL_IMPL OpenGL::GLES3)
    endif ()

  endif()

  if(OPENGL_gl_LIBRARY AND NOT TARGET OpenGL::GL)
    # A legacy GL library is available, so use it for the legacy GL target.
    if(IS_ABSOLUTE "${OPENGL_gl_LIBRARY}")
      add_library(OpenGL::GL UNKNOWN IMPORTED)
      set_target_properties(OpenGL::GL PROPERTIES
        IMPORTED_LOCATION "${OPENGL_gl_LIBRARY}")
    else()
      add_library(OpenGL::GL INTERFACE IMPORTED)
      set_target_properties(OpenGL::GL PROPERTIES
        IMPORTED_LIBNAME "${OPENGL_gl_LIBRARY}")
    endif()
    set_target_properties(OpenGL::GL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${OPENGL_INCLUDE_DIR}")
  elseif(NOT TARGET OpenGL::GL AND TARGET OpenGL::OpenGL AND TARGET OpenGL::GLX)
    # A legacy GL library is not available, but we can provide the legacy GL
    # target using GLVND OpenGL+GLX.
    add_library(OpenGL::GL INTERFACE IMPORTED)
    set_target_properties(OpenGL::GL PROPERTIES INTERFACE_LINK_LIBRARIES
                          OpenGL::OpenGL)
    set_property(TARGET OpenGL::GL APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 OpenGL::GLX)
    set_target_properties(OpenGL::GL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                          "${OPENGL_INCLUDE_DIR}")
  endif()

  # ::EGL is a GLVND library, and thus Linux-only: we don't bother checking
  # for a framework version of this library.
  # Note we test whether _OpenGL_EGL_IMPL is set. Based on the OpenGL implementation,
  # _OpenGL_EGL_IMPL will be one of OpenGL::OpenGL, OpenGL::GLES2, OpenGL::GLES3
  if(_OpenGL_EGL_IMPL AND OpenGL_EGL_FOUND AND NOT TARGET OpenGL::EGL)
    if(IS_ABSOLUTE "${OPENGL_egl_LIBRARY}")
      add_library(OpenGL::EGL UNKNOWN IMPORTED)
      set_target_properties(OpenGL::EGL PROPERTIES IMPORTED_LOCATION
                            "${OPENGL_egl_LIBRARY}")
    else()
      add_library(OpenGL::EGL INTERFACE IMPORTED)
      set_target_properties(OpenGL::EGL PROPERTIES IMPORTED_LIBNAME
                            "${OPENGL_egl_LIBRARY}")
    endif()
    set_target_properties(OpenGL::EGL PROPERTIES INTERFACE_LINK_LIBRARIES
                          "${_OpenGL_EGL_IMPL}")
    # Note that EGL's include directory is different from OpenGL/GLX's!
    set_target_properties(OpenGL::EGL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                          "${OPENGL_EGL_INCLUDE_DIR}")
    list(APPEND OPENGL_INCLUDE_DIRS ${OPENGL_EGL_INCLUDE_DIR})
  endif()

  if(OPENGL_GLU_FOUND AND NOT TARGET OpenGL::GLU)
    if(IS_ABSOLUTE "${OPENGL_glu_LIBRARY}")
      add_library(OpenGL::GLU UNKNOWN IMPORTED)
      set_target_properties(OpenGL::GLU PROPERTIES
        IMPORTED_LOCATION "${OPENGL_glu_LIBRARY}")
    else()
      add_library(OpenGL::GLU INTERFACE IMPORTED)
      set_target_properties(OpenGL::GLU PROPERTIES
        IMPORTED_LIBNAME "${OPENGL_glu_LIBRARY}")
    endif()
    set_target_properties(OpenGL::GLU PROPERTIES
      INTERFACE_LINK_LIBRARIES OpenGL::GL)
    # Note that GLU's include directory may be different from OpenGL's!
    set_target_properties(OpenGL::GLU PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                          "${OPENGL_GLU_INCLUDE_DIR}")
    list(APPEND OPENGL_INCLUDE_DIRS ${OPENGL_GLU_INCLUDE_DIR})
  endif()

  # OPENGL_LIBRARIES mirrors OpenGL::GL's logic ...
  if(OPENGL_gl_LIBRARY)
    set(OPENGL_LIBRARIES ${OPENGL_gl_LIBRARY})
  elseif(TARGET OpenGL::OpenGL AND TARGET OpenGL::GLX)
    set(OPENGL_LIBRARIES ${OPENGL_opengl_LIBRARY} ${OPENGL_glx_LIBRARY})
  else()
    set(OPENGL_LIBRARIES "")
  endif()
  # ... and also includes GLU, if available.
  if(TARGET OpenGL::GLU)
    list(APPEND OPENGL_LIBRARIES ${OPENGL_glu_LIBRARY})
  endif()
endif()

list(REMOVE_DUPLICATES OPENGL_INCLUDE_DIRS)

# This deprecated setting is for backward compatibility with CMake1.4
set(OPENGL_LIBRARY ${OPENGL_LIBRARIES})
# This deprecated setting is for backward compatibility with CMake1.4
set(OPENGL_INCLUDE_PATH ${OPENGL_INCLUDE_DIR})

mark_as_advanced(${_OpenGL_CACHE_VARS})
unset(_OpenGL_CACHE_VARS)
