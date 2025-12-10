# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CMakeFindFrameworks
-------------------

.. deprecated:: 3.31
  This module does nothing, unless policy :policy:`CMP0173` is set to ``OLD``.

helper module to find OSX frameworks

This module reads hints about search locations from variables::

  CMAKE_FIND_FRAMEWORK_EXTRA_LOCATIONS - Extra directories
#]=======================================================================]

cmake_policy(GET CMP0173 _cmp0173)
if(_cmp0173 STREQUAL "NEW")
  message(FATAL_ERROR
    "CMakeFindFrameworks.cmake is not maintained and lacks support for more "
    "recent framework handling. It will be removed in a future version of "
    "CMake. Update the code to use find_library() instead. "
    "Use of this module is now an error according to policy CMP0173."
  )
elseif(_cmp0173 STREQUAL "")
  # CMake will have already emitted the standard policy warning for the point
  # of inclusion. We only need to add the context-specific info here.
  message(AUTHOR_WARNING
    "CMakeFindFrameworks.cmake is not maintained and lacks support for more "
    "recent framework handling. It will be removed in a future version of "
    "CMake. Update the code to use find_library() instead."
  )
endif ()
unset(_cmp0173)

if(NOT CMAKE_FIND_FRAMEWORKS_INCLUDED)
  set(CMAKE_FIND_FRAMEWORKS_INCLUDED 1)
  macro(CMAKE_FIND_FRAMEWORKS fwk)
    set(${fwk}_FRAMEWORKS)
    if(APPLE)
      # 'Frameworks' directory from Brew (Apple Silicon and Intel)
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(_brew_framework_path /opt/homebrew/Frameworks)
      else()
        set(_brew_framework_path /usr/local/Frameworks)
      endif()

      file(TO_CMAKE_PATH "$ENV{CMAKE_FRAMEWORK_PATH}" _cmff_CMAKE_FRAMEWORK_PATH)
      set(_cmff_search_paths
            ${CMAKE_FRAMEWORK_PATH}
            ${_cmff_CMAKE_FRAMEWORK_PATH}
            ~/Library/Frameworks
            ${_brew_framework_path}
            /Library/Frameworks
            /System/Library/Frameworks
            /Network/Library/Frameworks
            ${CMAKE_SYSTEM_FRAMEWORK_PATH})

      # For backwards compatibility reasons,
      # CMAKE_FIND_FRAMEWORK_EXTRA_LOCATIONS includes ${fwk}.framework
      list(TRANSFORM _cmff_search_paths APPEND /${fwk}.framework)
      list(APPEND _cmff_search_paths ${CMAKE_FIND_FRAMEWORK_EXTRA_LOCATIONS})

      list(REMOVE_DUPLICATES _cmff_search_paths)

      foreach(dir IN LISTS _cmff_search_paths)
        if(EXISTS ${dir})
          set(${fwk}_FRAMEWORKS ${${fwk}_FRAMEWORKS} ${dir})
        endif()
      endforeach()
    endif()
  endmacro()
endif()
