# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# Do NOT include this module directly into any of your code. It is used by
# the try_compile() implementation to work around a specific issue with
# conflicting flags when building for Apple platforms.
if(NOT APPLE)
  return()
endif()

function(__cmake_internal_workaround_headerpad_flag_conflict _LANG)

  # Until we can avoid hard-coding -Wl,-headerpad_max_install_names in the
  # linker flags, we need to remove it here for cases where we know it will
  # conflict with other flags, generate a warning and be ignored.
  set(regex "(^| )(-fembed-bitcode(-marker|=(all|bitcode|marker))?|-bundle_bitcode)($| )")
  set(remove_headerpad NO)

  # Check arbitrary flags that the user or project has set. These compiler
  # flags get added to the linker command line.
  if("${CMAKE_${_LANG}_FLAGS}" MATCHES "${regex}")
    set(remove_headerpad YES)
  endif()
  if(NOT remove_headerpad)
    get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(is_multi_config)
      # Only one of these config-specific variables will be set by try_compile()
      # and the rest will be unset, but we can't easily tell which one is set.
      # No harm to just add them all here, empty ones won't add flags to check.
      foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
        if("${CMAKE_${_LANG}_FLAGS_${config}}" MATCHES "${regex}")
          set(remove_headerpad YES)
          break()
        endif()
      endforeach()
    else()
      if("${CMAKE_${_LANG}_FLAGS_${CMAKE_BUILD_TYPE}}" MATCHES "${regex}")
        set(remove_headerpad YES)
      endif()
    endif()
  endif()

  # The try_compile() command passes compiler flags to check in a way that
  # results in them being added to add_definitions(). Those don't end up on
  # the linker command line, so we don't need to check them here.

  if(remove_headerpad)
    foreach(flag IN ITEMS
      CMAKE_${_LANG}_LINK_FLAGS
      CMAKE_SHARED_LIBRARY_CREATE_${_LANG}_FLAGS
      CMAKE_SHARED_MODULE_CREATE_${_LANG}_FLAGS)
      string(REPLACE "-Wl,-headerpad_max_install_names" "" ${flag} "${${flag}}")
      set(${flag} "${${flag}}" PARENT_SCOPE)
    endforeach()
  endif()
endfunction()

get_property(__enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
foreach(__lang IN LISTS __enabled_languages)
  __cmake_internal_workaround_headerpad_flag_conflict(${__lang})
endforeach()
unset(__lang)
unset(__enabled_languages)
