# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Function to identify the linker.  This is used internally by CMake and should
# not be included by user code.
# If successful, sets CMAKE_<lang>_COMPILER_LINKER_ID and
# CMAKE_<lang>_COMPILER_LINKER_VERSION

function(cmake_determine_linker_id lang linker)
  if (NOT linker)
    # linker was not identified
    unset(CMAKE_${lang}_COMPILER_LINKER_ID PARENT_SCOPE)
    unset(CMAKE_${lang}_COMPILER_LINKER_VERSION PARENT_SCOPE)
    unset(CMAKE_${lang}_COMPILER_LINKER_FRONTEND_VARIANT PARENT_SCOPE)
    return()
  endif()

  set(linker_id)
  set(linker_frontend)
  set(linker_version)

  # Compute the linker ID and version.
  foreach(flags IN ITEMS
      "-v"        # AppleClang, GNU, GNUgold, MOLD
      "-V"        # AIX, Solaris
      "--version" # LLD
      )
    execute_process(COMMAND "${linker}" ${flags}
                    OUTPUT_VARIABLE linker_desc
                    ERROR_VARIABLE linker_desc
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_STRIP_TRAILING_WHITESPACE
                    COMMAND_ERROR_IS_FATAL NONE
    )

    string(JOIN "\" \"" flags_string ${flags})
    string(REGEX REPLACE "\n\n.*" "" linker_desc_head "${linker_desc}")
    message(CONFIGURE_LOG
      "Running the ${lang} compiler's linker: \"${linker}\" \"${flags_string}\"\n"
      "${linker_desc_head}\n"
      )

    if(CMAKE_EFFECTIVE_SYSTEM_NAME STREQUAL "Apple" AND linker_desc MATCHES "@\\(#\\)PROGRAM:ld.+PROJECT:[a-z0-9]+-([0-9.]+).+")
      set(linker_id "AppleClang")
      set(linker_frontend "GNU")
      set(linker_version "${CMAKE_MATCH_1}")
      break()
    elseif(linker_desc MATCHES "mold \\(sold\\) ([0-9.]+)")
      set(linker_id "MOLD")
      set(linker_frontend "GNU")
      set(linker_version "${CMAKE_MATCH_1}")
      break()
    elseif(linker_desc MATCHES "mold ([0-9.]+)")
      set(linker_id "MOLD")
      set(linker_frontend "GNU")
      set(linker_version "${CMAKE_MATCH_1}")
      break()
    elseif(linker_desc MATCHES "LLD ([0-9.]+)")
      set(linker_id "LLD")
      set(linker_frontend "GNU")
      set(linker_version "${CMAKE_MATCH_1}")
      if(WIN32 AND NOT linker_desc MATCHES "compatible with GNU")
        set(linker_frontend "MSVC")
      endif()
      break()
    elseif(linker_desc MATCHES "GNU ld (\\([^)]+\\)|version) ([0-9.]+)")
      set(linker_id "GNU")
      set(linker_frontend "GNU")
      set(linker_version "${CMAKE_MATCH_2}")
      break()
    elseif(linker_desc MATCHES "GNU gold \\([^)]+\\) ([0-9.]+)")
      set(linker_id "GNUgold")
      set(linker_frontend "GNU")
      set(linker_version "${CMAKE_MATCH_1}")
      break()
    elseif(linker_desc MATCHES "Microsoft \\(R\\) Incremental Linker Version ([0-9.]+)")
      set(linker_id "MSVC")
      set(linker_frontend "MSVC")
      set(linker_version "${CMAKE_MATCH_1}")
      break()
    elseif (CMAKE_SYSTEM_NAME STREQUAL "SunOS" AND linker_desc MATCHES "Solaris Link Editors: ([0-9.-]+)")
      set(linker_id "Solaris")
      set(linker_version "${CMAKE_MATCH_1}")
      break()
    elseif (CMAKE_SYSTEM_NAME STREQUAL "AIX" AND linker_desc MATCHES " LD ([0-9.]+)")
      set(linker_id "AIX")
      set(linker_version "${CMAKE_MATCH_1}")
      break()
    endif()
  endforeach()

  set(CMAKE_${lang}_COMPILER_LINKER_ID "${linker_id}" PARENT_SCOPE)
  if (linker_frontend)
    set(CMAKE_${lang}_COMPILER_LINKER_FRONTEND_VARIANT "${linker_frontend}" PARENT_SCOPE)
  else()
    unset(CMAKE_${lang}_COMPILER_LINKER_FRONTEND_VARIANT PARENT_SCOPE)
  endif()
  if (linker_version)
    set(CMAKE_${lang}_COMPILER_LINKER_VERSION "${linker_version}" PARENT_SCOPE)
  else()
    unset(CMAKE_${lang}_COMPILER_LINKER_VERSION PARENT_SCOPE)
  endif()
endfunction()
