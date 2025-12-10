# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(NOT "${CPACK_EXTERNAL_REQUESTED_VERSIONS}" STREQUAL "")
  unset(_found_major)

  foreach(_req_version IN LISTS CPACK_EXTERNAL_REQUESTED_VERSIONS)
    if(_req_version MATCHES "^([0-9]+)\\.([0-9]+)$")
      set(_req_major "${CMAKE_MATCH_1}")
      set(_req_minor "${CMAKE_MATCH_2}")

      foreach(_known_version IN LISTS CPACK_EXTERNAL_KNOWN_VERSIONS)
        string(REGEX MATCH
          "^([0-9]+)\\.([0-9]+)$"
          _known_version_dummy
          "${_known_version}"
        )

        set(_known_major "${CMAKE_MATCH_1}")
        set(_known_minor "${CMAKE_MATCH_2}")

        if(_req_major EQUAL _known_major AND NOT _known_minor LESS _req_minor)
          set(_found_major "${_known_major}")
          set(_found_minor "${_known_minor}")
          break()
        endif()
      endforeach()

      if(DEFINED _found_major)
        break()
      endif()
    endif()
  endforeach()

  if(DEFINED _found_major)
    set(CPACK_EXTERNAL_SELECTED_MAJOR "${_found_major}")
    set(CPACK_EXTERNAL_SELECTED_MINOR "${_found_minor}")
    set(CPACK_EXTERNAL_SELECTED_VERSION "${_found_major}.${_found_minor}")
  else()
    message(FATAL_ERROR
      "Could not find a suitable version in CPACK_EXTERNAL_REQUESTED_VERSIONS"
    )
  endif()
else()
  list(GET CPACK_EXTERNAL_KNOWN_VERSIONS 0 CPACK_EXTERNAL_SELECTED_VERSION)
  string(REGEX MATCH
    "^([0-9]+)\\.([0-9]+)$"
    _dummy
    "${CPACK_EXTERNAL_SELECTED_VERSION}"
  )
  set(CPACK_EXTERNAL_SELECTED_MAJOR "${CMAKE_MATCH_1}")
  set(CPACK_EXTERNAL_SELECTED_MINOR "${CMAKE_MATCH_2}")
endif()
