# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLua51
---------

.. note::

  This module is intended specifically for Lua version branch 5.1, which is
  obsolete and not maintained anymore.  In new code use the latest supported
  Lua version and the version-agnostic module :module:`FindLua` instead.

Finds the Lua library:

.. code-block:: cmake

  find_package(Lua51 [<version>] [...])

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

``Lua51_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) Lua was found.

``Lua51_VERSION``
  .. versionadded:: 4.2

  The version of Lua 5.1 found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LUA_INCLUDE_DIR``
  The directory containing the Lua header files, such as ``lua.h``,
  ``lualib.h``, and ``lauxlib.h``, needed to use Lua.

``LUA_LIBRARIES``
  Libraries needed to link against to use Lua.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``LUA51_FOUND``
  .. deprecated:: 4.2
    Use ``Lua51_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) Lua was found.

``LUA_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``Lua51_VERSION``, which has the same value.

  The version of Lua 5.1 found.

Examples
^^^^^^^^

Finding the Lua 5.1 library and creating an interface :ref:`imported target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(Lua51)

  if(Lua51_FOUND AND NOT TARGET Lua51::Lua51)
    add_library(Lua51::Lua51 INTERFACE IMPORTED)
    set_target_properties(
      Lua51::Lua51
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LUA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${LUA_LIBRARIES}"
    )
  endif()

  target_link_libraries(project_target PRIVATE Lua51::Lua51)

See Also
^^^^^^^^

* The :module:`FindLua` module to find Lua in version-agnostic way.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(LUA_INCLUDE_DIR lua.h
  HINTS
    ENV LUA_DIR
  PATH_SUFFIXES lua51 lua5.1 lua-5.1 lua
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /opt
)

find_library(LUA_LIBRARY
  NAMES lua51 lua5.1 lua-5.1 lua
  HINTS
    ENV LUA_DIR
  PATH_SUFFIXES lib
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /opt
)

if(LUA_LIBRARY)
  # include the math library for Unix
  if(UNIX AND NOT APPLE AND NOT BEOS AND NOT HAIKU)
    find_library(LUA_MATH_LIBRARY m)
    set( LUA_LIBRARIES "${LUA_LIBRARY};${LUA_MATH_LIBRARY}" CACHE STRING "Lua Libraries")
  # For Windows and Mac, don't need to explicitly include the math library
  else()
    set( LUA_LIBRARIES "${LUA_LIBRARY}" CACHE STRING "Lua Libraries")
  endif()
endif()

if(LUA_INCLUDE_DIR AND EXISTS "${LUA_INCLUDE_DIR}/lua.h")
  file(STRINGS "${LUA_INCLUDE_DIR}/lua.h" lua_version_str REGEX "^#define[ \t]+LUA_RELEASE[ \t]+\"Lua .+\"")

  string(REGEX REPLACE "^#define[ \t]+LUA_RELEASE[ \t]+\"Lua ([^\"]+)\".*" "\\1" Lua51_VERSION "${lua_version_str}")
  set(LUA_VERSION_STRING "${Lua51_VERSION}")
  unset(lua_version_str)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Lua51
                                  REQUIRED_VARS LUA_LIBRARIES LUA_INCLUDE_DIR
                                  VERSION_VAR Lua51_VERSION)

mark_as_advanced(LUA_INCLUDE_DIR LUA_LIBRARIES LUA_LIBRARY LUA_MATH_LIBRARY)

cmake_policy(POP)
