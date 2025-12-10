# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
WriteBasicConfigVersionFile
---------------------------

.. deprecated:: 3.0

  Use the identical command :command:`write_basic_package_version_file()`
  from module :module:`CMakePackageConfigHelpers`.

.. code-block:: cmake

  WRITE_BASIC_CONFIG_VERSION_FILE(filename
    [VERSION major.minor.patch]
    COMPATIBILITY (AnyNewerVersion|SameMajorVersion|SameMinorVersion|ExactVersion)
    [ARCH_INDEPENDENT]
    )


#]=======================================================================]

function(WRITE_BASIC_CONFIG_VERSION_FILE _filename)

  set(options ARCH_INDEPENDENT )
  set(oneValueArgs VERSION COMPATIBILITY )
  set(multiValueArgs )

  cmake_parse_arguments(CVF "${options}" "${oneValueArgs}" "${multiValueArgs}"  ${ARGN})

  if(CVF_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown keywords given to WRITE_BASIC_CONFIG_VERSION_FILE(): \"${CVF_UNPARSED_ARGUMENTS}\"")
  endif()

  set(versionTemplateFile "${CMAKE_ROOT}/Modules/BasicConfigVersion-${CVF_COMPATIBILITY}.cmake.in")
  if(NOT EXISTS "${versionTemplateFile}")
    message(FATAL_ERROR "Bad COMPATIBILITY value used for WRITE_BASIC_CONFIG_VERSION_FILE(): \"${CVF_COMPATIBILITY}\"")
  endif()

  if("${CVF_VERSION}" STREQUAL "")
    if ("${PROJECT_VERSION}" STREQUAL "")
      message(FATAL_ERROR "No VERSION specified for WRITE_BASIC_CONFIG_VERSION_FILE()")
    else()
      set(CVF_VERSION "${PROJECT_VERSION}")
    endif()
  endif()

  if(NOT CVF_ARCH_INDEPENDENT)
    set(CVF_ARCH_INDEPENDENT_CHECK "
# if the installed or the using project don't have CMAKE_SIZEOF_VOID_P set, ignore it:
if(\"\${CMAKE_SIZEOF_VOID_P}\" STREQUAL \"\" OR \"${CMAKE_SIZEOF_VOID_P}\" STREQUAL \"\")
  return()
endif()

# check that the installed version has the same 32/64bit-ness as the one which is currently searching:
if(NOT CMAKE_SIZEOF_VOID_P STREQUAL \"${CMAKE_SIZEOF_VOID_P}\")
  math(EXPR installedBits \"${CMAKE_SIZEOF_VOID_P} * 8\")
  set(PACKAGE_VERSION \"\${PACKAGE_VERSION} (\${installedBits}bit)\")
  set(PACKAGE_VERSION_UNSUITABLE TRUE)
endif()")
  endif()

  configure_file("${versionTemplateFile}" "${_filename}" @ONLY)

endfunction()
