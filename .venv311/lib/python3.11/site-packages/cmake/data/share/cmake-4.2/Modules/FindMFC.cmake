# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMFC
-------

Finds the native Microsoft Foundation Class Library (MFC) for developing MFC
applications on Windows:

.. code-block:: cmake

  find_package(MFC [...])

.. note::

  MFC is an optional component in Visual Studio and must be installed
  separately for this module to succeed.

Once the MFC libraries and headers are found, no additional manual linking is
needed, as they are part of the development environment.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``MFC_FOUND``
  Boolean indicating whether MFC support was found.

Examples
^^^^^^^^

Using this module to check if the application can link to the MFC libraries:

.. code-block:: cmake

  find_package(MFC)

  if(MFC_FOUND)
    # Example logic when MFC is available...
    set(CMAKE_MFC_FLAG 2)
    add_executable(app WIN32 main.cpp)
    target_compile_definitions(app PRIVATE _AFXDLL)
  endif()

See Also
^^^^^^^^

* The :variable:`CMAKE_MFC_FLAG` variable.
#]=======================================================================]

# Assume no MFC support
set(MFC_FOUND "NO")

# Only attempt the try_compile call if it has a chance to succeed:
set(MFC_ATTEMPT_TRY_COMPILE 0)
if(WIN32 AND NOT UNIX AND NOT BORLAND AND NOT MINGW)
  set(MFC_ATTEMPT_TRY_COMPILE 1)
endif()

if(MFC_ATTEMPT_TRY_COMPILE)
  if(NOT DEFINED MFC_HAVE_MFC)
    set(CHECK_INCLUDE_FILE_VAR "afxwin.h")
    file(READ ${CMAKE_ROOT}/Modules/CheckIncludeFile.cxx.in _CIF_SOURCE_CONTENT)
    string(CONFIGURE "${_CIF_SOURCE_CONTENT}" _CIF_SOURCE_CONTENT)
    message(CHECK_START "Looking for MFC")
    # Try both shared and static as the root project may have set the /MT flag
    try_compile(MFC_HAVE_MFC
      SOURCE_FROM_VAR CheckIncludeFile.cxx _CIF_SOURCE_CONTENT
      CMAKE_FLAGS
      -DCMAKE_MFC_FLAG:STRING=2
      -DCOMPILE_DEFINITIONS:STRING=-D_AFXDLL
      OUTPUT_VARIABLE OUTPUT)
    if(NOT MFC_HAVE_MFC)
      try_compile(MFC_HAVE_MFC
        SOURCE_FROM_VAR CheckIncludeFile.cxx _CIF_SOURCE_CONTENT
        CMAKE_FLAGS
        -DCMAKE_MFC_FLAG:STRING=1
        OUTPUT_VARIABLE OUTPUT)
    endif()
    if(MFC_HAVE_MFC)
      message(CHECK_PASS "found")
      set(MFC_HAVE_MFC 1 CACHE INTERNAL "Have MFC?")
    else()
      message(CHECK_FAIL "not found")
      set(MFC_HAVE_MFC 0 CACHE INTERNAL "Have MFC?")
    endif()
  endif()

  if(MFC_HAVE_MFC)
    set(MFC_FOUND "YES")
  endif()
endif()
