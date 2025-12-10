# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Block multiple inclusion because "CMakeCInformation.cmake" includes
# "Platform/${CMAKE_SYSTEM_NAME}" even though the generic module
# "CMakeSystemSpecificInformation.cmake" already included it.
# The extra inclusion is a work-around documented next to the include()
# call, so this can be removed when the work-around is removed.
if(__WINDOWS_PATHS_INCLUDED)
  return()
endif()
set(__WINDOWS_PATHS_INCLUDED 1)

# Add the program-files folder(s) to the list of installation
# prefixes.
#
# Windows 64-bit Binary:
#   ENV{ProgramFiles(x86)} = [C:\Program Files (x86)]
#   ENV{ProgramFiles} = [C:\Program Files]
#   ENV{ProgramW6432} = [C:\Program Files] or <not set>
#
# Windows 32-bit Binary on 64-bit Windows:
#   ENV{ProgramFiles(x86)} = [C:\Program Files (x86)]
#   ENV{ProgramFiles} = [C:\Program Files (x86)]
#   ENV{ProgramW6432} = [C:\Program Files]
#
# Reminder when adding new locations computed from environment variables
# please make sure to keep Help/variable/CMAKE_SYSTEM_PREFIX_PATH.rst
# synchronized
set(_programfiles "")
foreach(v "ProgramW6432" "ProgramFiles" "ProgramFiles(x86)")
  if(DEFINED "ENV{${v}}")
    file(TO_CMAKE_PATH "$ENV{${v}}" _env_programfiles)
    list(APPEND _programfiles "${_env_programfiles}")
    unset(_env_programfiles)
  endif()
endforeach()
if(DEFINED "ENV{SystemDrive}")
  foreach(d "Program Files" "Program Files (x86)")
    if(EXISTS "$ENV{SystemDrive}/${d}")
      list(APPEND _programfiles "$ENV{SystemDrive}/${d}")
    endif()
  endforeach()
endif()
if(_programfiles)
  list(REMOVE_DUPLICATES _programfiles)
  list(APPEND CMAKE_SYSTEM_PREFIX_PATH ${_programfiles})
endif()
unset(_programfiles)

# Add the CMake install location.
get_filename_component(_CMAKE_INSTALL_DIR "${CMAKE_ROOT}" PATH)
get_filename_component(_CMAKE_INSTALL_DIR "${_CMAKE_INSTALL_DIR}" PATH)
list(APPEND CMAKE_SYSTEM_PREFIX_PATH "${_CMAKE_INSTALL_DIR}")

if (NOT CMAKE_FIND_NO_INSTALL_PREFIX)
  # Add other locations.
  list(APPEND CMAKE_SYSTEM_PREFIX_PATH
    # Project install destination.
    "${CMAKE_INSTALL_PREFIX}"
    )
  if (CMAKE_STAGING_PREFIX)
    list(APPEND CMAKE_SYSTEM_PREFIX_PATH
      # User-supplied staging prefix.
      "${CMAKE_STAGING_PREFIX}"
    )
  endif()
endif()
_cmake_record_install_prefix()

if(CMAKE_CROSSCOMPILING AND NOT CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
  # MinGW (useful when cross compiling from linux with CMAKE_FIND_ROOT_PATH set)
  list(APPEND CMAKE_SYSTEM_PREFIX_PATH /)
endif()

list(APPEND CMAKE_SYSTEM_INCLUDE_PATH
  )

# mingw can also link against dlls which can also be in /bin, so list this too
if (NOT CMAKE_FIND_NO_INSTALL_PREFIX)
  list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
    "${CMAKE_INSTALL_PREFIX}/bin"
  )
  if (CMAKE_STAGING_PREFIX)
    list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
      "${CMAKE_STAGING_PREFIX}/bin"
    )
  endif()
endif()
list(APPEND CMAKE_SYSTEM_LIBRARY_PATH
  "${_CMAKE_INSTALL_DIR}/bin"
  /bin
  )

list(APPEND CMAKE_SYSTEM_PROGRAM_PATH
  )
