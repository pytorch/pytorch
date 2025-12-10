# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLua50
---------

.. note::

  This module is intended specifically for Lua version branch 5.0, which is
  obsolete and not maintained anymore.  In new code use the latest supported
  Lua version and the version-agnostic module :module:`FindLua` instead.

Finds the Lua library:

.. code-block:: cmake

  find_package(Lua50 [...])

Lua is a embeddable scripting language.

When working with Lua, its library headers are intended to be included in
project source code as:

.. code-block:: c

  #include <lua.h>

and not:

.. code-block:: c

  #include <lua/lua.h>

This is because, the location of Lua headers may differ across platforms and may
exist in locations other than ``lua/``.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Lua50_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether Lua was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LUA_INCLUDE_DIR``
  The directory containing the Lua header files, such as ``lua.h``,
  ``lualib.h``, and ``lauxlib.h``, needed to use Lua.

``LUA_LIBRARIES``
  Libraries needed to link against to use Lua.  This list includes both ``lua``
  and ``lualib`` libraries.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``LUA50_FOUND``
  .. deprecated:: 4.2
    Use ``Lua50_FOUND``, which has the same value.

  Boolean indicating whether Lua was found.

Examples
^^^^^^^^

Finding the Lua 5.0 library and creating an interface :ref:`imported target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(Lua50)

  if(Lua50_FOUND AND NOT TARGET Lua50::Lua50)
    add_library(Lua50::Lua50 INTERFACE IMPORTED)
    set_target_properties(
      Lua50::Lua50
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LUA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${LUA_LIBRARIES}"
    )
  endif()

  target_link_libraries(project_target PRIVATE Lua50::Lua50)

See Also
^^^^^^^^

* The :module:`FindLua` module to find Lua in version-agnostic way.
#]=======================================================================]

find_path(LUA_INCLUDE_DIR lua.h
  HINTS
    ENV LUA_DIR
  PATH_SUFFIXES lua50 lua5.0 lua5 lua
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /opt
)

find_library(LUA_LIBRARY_lua
  NAMES lua50 lua5.0 lua-5.0 lua5 lua
  HINTS
    ENV LUA_DIR
  PATH_SUFFIXES lib
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /opt
)

# In an OS X framework, lualib is usually included as part of the framework
# (like GLU in OpenGL.framework)
if(${LUA_LIBRARY_lua} MATCHES "framework")
  set( LUA_LIBRARIES "${LUA_LIBRARY_lua}" CACHE STRING "Lua framework")
else()
  find_library(LUA_LIBRARY_lualib
    NAMES lualib50 lualib5.0 lualib5 lualib
    HINTS
      ENV LUALIB_DIR
      ENV LUA_DIR
    PATH_SUFFIXES lib
    PATHS
    /opt
  )
  if(LUA_LIBRARY_lualib AND LUA_LIBRARY_lua)
    # include the math library for Unix
    if(UNIX AND NOT APPLE)
      find_library(MATH_LIBRARY_FOR_LUA m)
      set( LUA_LIBRARIES "${LUA_LIBRARY_lualib};${LUA_LIBRARY_lua};${MATH_LIBRARY_FOR_LUA}" CACHE STRING "This is the concatenation of lua and lualib libraries")
    # For Windows and Mac, don't need to explicitly include the math library
    else()
      set( LUA_LIBRARIES "${LUA_LIBRARY_lualib};${LUA_LIBRARY_lua}" CACHE STRING "This is the concatenation of lua and lualib libraries")
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Lua50  DEFAULT_MSG  LUA_LIBRARIES LUA_INCLUDE_DIR)

mark_as_advanced(LUA_INCLUDE_DIR LUA_LIBRARIES)
