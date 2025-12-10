# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Function parse implicit linker options.
# This is used internally by CMake and should not be included by user
# code.

function(cmake_parse_library_architecture lang implicit_dirs implicit_objs output_var)
  unset(library_arch)
  # Detect library architecture directory name.
  if(CMAKE_LIBRARY_ARCHITECTURE_REGEX)
    foreach(dir IN LISTS implicit_dirs)
      if("${dir}" MATCHES "/lib/${CMAKE_LIBRARY_ARCHITECTURE_REGEX}$")
        get_filename_component(arch "${dir}" NAME)
        set(library_arch "${arch}")
        break()
      endif()
    endforeach()

    foreach(obj IN LISTS implicit_objs)
      get_filename_component(dir "${obj}" DIRECTORY)
      if("${dir}" MATCHES "(/usr)?/lib/${CMAKE_LIBRARY_ARCHITECTURE_REGEX}$")
        get_filename_component(arch "${dir}" NAME)
        set(library_arch "${arch}")
        break()
      endif()
    endforeach()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL QCC)
    foreach(dir ${implicit_dirs})
      if (dir MATCHES "/lib$")
        get_filename_component(assumedArchDir "${dir}" DIRECTORY)
        get_filename_component(archParentDir "${assumedArchDir}" DIRECTORY)
        if (archParentDir STREQUAL CMAKE_SYSROOT)
          get_filename_component(archDirName "${assumedArchDir}" NAME)
          set(library_arch "${archDirName}")
          break()
        endif()
      endif()
    endforeach()
  endif()

  # Return results.
  if(library_arch)
    set(${output_var} "${library_arch}" PARENT_SCOPE)
  endif()
endfunction()
