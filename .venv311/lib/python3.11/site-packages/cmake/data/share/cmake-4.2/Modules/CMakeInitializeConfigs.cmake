# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include_guard(GLOBAL)

# Initializes `<_PREFIX>_<CONFIG>` variables from the corresponding
# `<_PREFIX>_<CONFIG>_INIT`, for the configurations currently used.
function(cmake_initialize_per_config_variable _PREFIX _DOCSTRING)
  string(STRIP "${${_PREFIX}_INIT}" _INIT)
  set("${_PREFIX}" "${_INIT}"
    CACHE STRING "${_DOCSTRING} during all build types.")
  mark_as_advanced("${_PREFIX}")

  if (NOT CMAKE_NOT_USING_CONFIG_FLAGS)
    set(_CONFIGS Debug Release MinSizeRel RelWithDebInfo)

    get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (_GENERATOR_IS_MULTI_CONFIG)
      list(APPEND _CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
      if (NOT CMAKE_NO_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE_INIT}" CACHE STRING
          "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
      endif()
      list(APPEND _CONFIGS ${CMAKE_BUILD_TYPE})
    endif()

    list(REMOVE_DUPLICATES _CONFIGS)
    foreach(_BUILD_TYPE IN LISTS _CONFIGS)
      if (NOT "${_BUILD_TYPE}" STREQUAL "")
        string(TOUPPER "${_BUILD_TYPE}" _BUILD_TYPE)
        string(STRIP "${${_PREFIX}_${_BUILD_TYPE}_INIT}" _INIT)
        set("${_PREFIX}_${_BUILD_TYPE}" "${_INIT}"
          CACHE STRING "${_DOCSTRING} during ${_BUILD_TYPE} builds.")
        mark_as_advanced("${_PREFIX}_${_BUILD_TYPE}")
      endif()
    endforeach()
  endif()
endfunction()
