# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple linkers; use include blocker.
include_guard()

block(SCOPE_FOR POLICIES)
cmake_policy(SET CMP0140 NEW)

function(__linker_gnu lang)
  # define flags for linker depfile generation
  set(CMAKE_${lang}_LINKER_DEPFILE_FLAGS "LINKER:--dependency-file=<DEP_FILE>")
  set(CMAKE_${lang}_LINKER_DEPFILE_FORMAT gcc)

  if(NOT CMAKE_EXECUTABLE_FORMAT STREQUAL "ELF")
    # Only ELF binary format supports this capability
    set(CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED FALSE)
  endif()

  if(NOT DEFINED CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED)
    ## Ensure ninja tool is recent enough...
    if(CMAKE_GENERATOR MATCHES "^Ninja")
      # Ninja 1.10 or upper is required
      execute_process(COMMAND "${CMAKE_MAKE_PROGRAM}" --version
        OUTPUT_VARIABLE _ninja_version
        ERROR_VARIABLE _ninja_version)
      if (_ninja_version MATCHES "[0-9]+(\\.[0-9]+)*")
        set (_ninja_version "${CMAKE_MATCH_0}")
      endif()
      if (_ninja_version VERSION_LESS "1.10")
        set(CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED FALSE)
      endif()
    endif()

    if (NOT DEFINED CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED)
      ## check if this feature is supported by the linker
      if (CMAKE_${lang}_COMPILER_LINKER AND CMAKE_${lang}_COMPILER_LINKER_ID MATCHES "GNU|LLD|MOLD")
        execute_process(COMMAND "${CMAKE_${lang}_COMPILER_LINKER}" --help
                        OUTPUT_VARIABLE _linker_capabilities
                        ERROR_VARIABLE _linker_capabilities)
        if(_linker_capabilities MATCHES "--dependency-file")
          set(CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED TRUE)
        else()
          set(CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED FALSE)
        endif()
      else()
        set(CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED FALSE)
      endif()
    endif()
  endif()
  if (CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED)
    set(CMAKE_${lang}_LINK_DEPENDS_USE_LINKER TRUE)
  else()
    set(CMAKE_${lang}_LINK_DEPENDS_USE_LINKER FALSE)
  endif()

  # Due to GNU binutils ld bug when LTO is enabled (see GNU bug
  # `30568 <https://sourceware.org/bugzilla/show_bug.cgi?id=30568>`_),
  # deactivate this feature if the version is less than 2.41.
  if (CMAKE_${lang}_COMPILER_LINKER_ID
      AND CMAKE_${lang}_COMPILER_LINKER_ID STREQUAL "GNU"
      AND CMAKE_${lang}_COMPILER_LINKER_VERSION VERSION_LESS "2.41")
    set(CMAKE_${lang}_LINK_DEPENDS_USE_LINKER FALSE)
  endif()

  # Linker warning as error
  set(CMAKE_${lang}_LINK_OPTIONS_WARNING_AS_ERROR "LINKER:--fatal-warnings")

  return(PROPAGATE CMAKE_${lang}_LINKER_DEPFILE_FLAGS
    CMAKE_${lang}_LINKER_DEPFILE_FORMAT
    CMAKE_${lang}_LINKER_DEPFILE_SUPPORTED
    CMAKE_${lang}_LINK_DEPENDS_USE_LINKER
    CMAKE_${lang}_LINK_OPTIONS_WARNING_AS_ERROR)
endfunction()

endblock()
