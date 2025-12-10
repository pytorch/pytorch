# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCURL
--------

Finds the native curl installation (include directories and libraries) for
transferring data with URLS:

.. code-block:: cmake

  find_package(CURL [<version>] [COMPONENTS <components>...] [...])

.. versionadded:: 3.17
  If curl is built using its CMake-based build system, it will provide its own
  CMake Package Configuration file (``CURLConfig.cmake``) for use with the
  :command:`find_package` command in *config mode*.  By default, this module
  searches for this file and, if found, returns the results without further
  action.  If the upstream configuration file is not found, this module falls
  back to *module mode* and searches standard locations.

.. versionadded:: 3.13
  Debug and Release library variants are found separately.

Components
^^^^^^^^^^

.. versionadded:: 3.14

This module supports optional components to detect the protocols and features
available in the installed curl (these can vary based on the curl version)::

  Protocols: DICT FILE FTP FTPS GOPHER GOPHERS HTTP HTTPS IMAP IMAPS IPFS IPNS
             LDAP LDAPS MQTT POP3 POP3S RTMP RTMPS RTSP SCP SFTP SMB SMBS SMTP
             SMTPS TELNET TFTP WS WSS
  Features:  alt-svc asyn-rr AsynchDNS brotli CAcert Debug ECH gsasl GSS-API
             HSTS HTTP2 HTTP3 HTTPS-proxy HTTPSRR IDN IPv6 Kerberos Largefile
             libz MultiSSL NTLM NTLM_WB PSL SPNEGO SSL SSLS-EXPORT SSPI
             threadsafe TLS-SRP TrackMemory Unicode UnixSockets zstd

Components can be specified with the :command:`find_package` command as required
for curl to be considered found:

.. code-block:: cmake

  find_package(CURL [COMPONENTS <protocols>... <features>...])

Or to check for them optionally, allowing conditional handling in the code:

.. code-block:: cmake

  find_package(CURL [OPTIONAL_COMPONENTS <protocols>... <features>...])

Refer to the curl documentation for more information on supported protocols and
features.  Component names are case-sensitive and follow the upstream curl
naming conventions.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``CURL::libcurl``
  .. versionadded:: 3.12

  Target encapsulating the curl usage requirements, available if curl is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``CURL_FOUND``
  Boolean indicating whether (the requested version of) curl and all required
  components were found.

``CURL_VERSION``
  .. versionadded:: 4.0

  The version of curl found.  This supersedes ``CURL_VERSION_STRING``.

``CURL_<component>_FOUND``
  .. versionadded:: 3.14

  Boolean indicating whether the specified component (curl protocol or feature)
  was found.

``CURL_INCLUDE_DIRS``
  Include directories containing the ``curl/curl.h`` and other headers needed to
  use curl.

  .. note::

    When curl is found via *config mode*, this variable is available only with
    curl version 8.9 or newer.

``CURL_LIBRARIES``
  List of libraries needed to link against to use curl.

  .. note::

    When curl is found via *module mode*, this is a list of library file paths.
    In *config mode*, this variable is available only with curl version 8.9 or
    newer and contains a list of imported targets.

Hints
^^^^^

This module accepts the following variables:

``CURL_NO_CURL_CMAKE``
  .. versionadded:: 3.17

  Set this variable to ``TRUE`` to disable searching for curl via *config mode*.

``CURL_USE_STATIC_LIBS``
  .. versionadded:: 3.28

  Set this variable to ``TRUE`` to use static libraries.  This is meaningful
  only when curl is not found via *config mode*.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``CURL_VERSION_STRING``
  .. deprecated:: 4.0
    Superseded by ``CURL_VERSION``.

  The version of curl found.

Examples
^^^^^^^^

Finding the curl library and specifying the required minimum version:

.. code-block:: cmake

  find_package(CURL 7.61.0)

Finding the curl library and linking it to a project target:

.. code-block:: cmake

  find_package(CURL)
  target_link_libraries(project_target PRIVATE CURL::libcurl)

Using components to check if the found curl supports specific protocols or
features:

.. code-block:: cmake

  find_package(CURL OPTIONAL_COMPONENTS HTTPS SSL)

  if(CURL_HTTPS_FOUND)
    # curl supports the HTTPS protocol
  endif()

  if(CURL_SSL_FOUND)
    # curl has SSL feature enabled
  endif()
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

include(FindPackageHandleStandardArgs)

if(NOT CURL_NO_CURL_CMAKE)
  # do a find package call to specifically look for the CMake version
  # of curl
  find_package(CURL QUIET NO_MODULE)
  mark_as_advanced(CURL_DIR)

  # if we found the CURL cmake package then we are done, and
  # can print what we found and return.
  if(CURL_FOUND)
    find_package_handle_standard_args(CURL HANDLE_COMPONENTS CONFIG_MODE)
    # The upstream curl package sets CURL_VERSION, not CURL_VERSION_STRING.
    set(CURL_VERSION_STRING "${CURL_VERSION}")

    cmake_policy(POP)
    return()
  endif()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_CURL QUIET libcurl)
  if(PC_CURL_FOUND)
    pkg_get_variable(CURL_SUPPORTED_PROTOCOLS_STRING libcurl supported_protocols)
    string(REPLACE " " ";" CURL_SUPPORTED_PROTOCOLS "${CURL_SUPPORTED_PROTOCOLS_STRING}")
    pkg_get_variable(CURL_SUPPORTED_FEATURES_STRING libcurl supported_features)
    string(REPLACE " " ";" CURL_SUPPORTED_FEATURES "${CURL_SUPPORTED_FEATURES_STRING}")
  endif()
endif()

# Look for the header file.
find_path(CURL_INCLUDE_DIR
          NAMES curl/curl.h
          HINTS ${PC_CURL_INCLUDE_DIRS})
mark_as_advanced(CURL_INCLUDE_DIR)

if(NOT CURL_LIBRARY)
  # Look for the library (sorted from most current/relevant entry to least).
  find_library(CURL_LIBRARY_RELEASE NAMES
      curl
    # Windows MSVC prebuilts:
      curllib
      libcurl_imp
      curllib_static
    # Windows older "Win32 - MSVC" prebuilts (libcurl.lib, e.g. libcurl-7.15.5-win32-msvc.zip):
      libcurl
    # Some Windows prebuilt versions distribute `libcurl_a.lib` instead of `libcurl.lib`
      libcurl_a
      NAMES_PER_DIR
      HINTS ${PC_CURL_LIBRARY_DIRS}
  )
  mark_as_advanced(CURL_LIBRARY_RELEASE)

  find_library(CURL_LIBRARY_DEBUG NAMES
    # Windows MSVC CMake builds in debug configuration on vcpkg:
      libcurl-d_imp
      libcurl-d
      NAMES_PER_DIR
      HINTS ${PC_CURL_LIBRARY_DIRS}
  )
  mark_as_advanced(CURL_LIBRARY_DEBUG)

  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(CURL)
endif()

if(CURL_INCLUDE_DIR)
  foreach(_curl_version_header curlver.h curl.h)
    if(EXISTS "${CURL_INCLUDE_DIR}/curl/${_curl_version_header}")
      file(STRINGS "${CURL_INCLUDE_DIR}/curl/${_curl_version_header}" curl_version_str REGEX "^#define[\t ]+LIBCURL_VERSION[\t ]+\".*\"")

      string(REGEX REPLACE "^#define[\t ]+LIBCURL_VERSION[\t ]+\"([^\"]*)\".*" "\\1" CURL_VERSION "${curl_version_str}")
      set(CURL_VERSION_STRING "${CURL_VERSION}")
      unset(curl_version_str)
      break()
    endif()
  endforeach()
endif()

if(CURL_FIND_COMPONENTS)
  foreach(component IN LISTS CURL_FIND_COMPONENTS)
    set(CURL_${component}_FOUND FALSE)
  endforeach()

  if(NOT PC_CURL_FOUND)
    find_program(CURL_CONFIG_EXECUTABLE NAMES curl-config)
    if(CURL_CONFIG_EXECUTABLE)
      execute_process(COMMAND ${CURL_CONFIG_EXECUTABLE} --version
                      OUTPUT_VARIABLE CURL_CONFIG_VERSION_STRING
                      ERROR_QUIET
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      execute_process(COMMAND ${CURL_CONFIG_EXECUTABLE} --feature
                      OUTPUT_VARIABLE CURL_CONFIG_FEATURES_STRING
                      ERROR_QUIET
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REPLACE "\n" ";" CURL_SUPPORTED_FEATURES "${CURL_CONFIG_FEATURES_STRING}")
      execute_process(COMMAND ${CURL_CONFIG_EXECUTABLE} --protocols
                      OUTPUT_VARIABLE CURL_CONFIG_PROTOCOLS_STRING
                      ERROR_QUIET
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REPLACE "\n" ";" CURL_SUPPORTED_PROTOCOLS "${CURL_CONFIG_PROTOCOLS_STRING}")
    endif()
  endif()

  foreach(component IN LISTS CURL_FIND_COMPONENTS)
    list(FIND CURL_SUPPORTED_PROTOCOLS ${component} _found)

    if(NOT _found EQUAL -1)
      set(CURL_${component}_FOUND TRUE)
    else()
      list(FIND CURL_SUPPORTED_FEATURES ${component} _found)
      if(NOT _found EQUAL -1)
        set(CURL_${component}_FOUND TRUE)
      endif()
    endif()
  endforeach()
endif()

find_package_handle_standard_args(CURL
                                  REQUIRED_VARS CURL_LIBRARY CURL_INCLUDE_DIR
                                  VERSION_VAR CURL_VERSION
                                  HANDLE_COMPONENTS)

if(CURL_FOUND)
  set(CURL_LIBRARIES ${CURL_LIBRARY})
  set(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIR})

  if(NOT TARGET CURL::libcurl)
    add_library(CURL::libcurl UNKNOWN IMPORTED)
    set_target_properties(CURL::libcurl PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIRS}")

    if(CURL_USE_STATIC_LIBS)
      set_property(TARGET CURL::libcurl APPEND PROPERTY
                   INTERFACE_COMPILE_DEFINITIONS "CURL_STATICLIB")
    endif()

    if(EXISTS "${CURL_LIBRARY}")
      set_target_properties(CURL::libcurl PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${CURL_LIBRARY}")
    endif()
    if(CURL_LIBRARY_RELEASE)
      set_property(TARGET CURL::libcurl APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(CURL::libcurl PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION_RELEASE "${CURL_LIBRARY_RELEASE}")
    endif()
    if(CURL_LIBRARY_DEBUG)
      set_property(TARGET CURL::libcurl APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(CURL::libcurl PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION_DEBUG "${CURL_LIBRARY_DEBUG}")
    endif()

    if(PC_CURL_FOUND)
      if(PC_CURL_LINK_LIBRARIES)
        set_property(TARGET CURL::libcurl PROPERTY
                     INTERFACE_LINK_LIBRARIES "${PC_CURL_LINK_LIBRARIES}")
      endif()
      if(PC_CURL_LDFLAGS_OTHER)
        set_property(TARGET CURL::libcurl PROPERTY
                     INTERFACE_LINK_OPTIONS "${PC_CURL_LDFLAGS_OTHER}")
      endif()
      if(PC_CURL_CFLAGS_OTHER)
        set_property(TARGET CURL::libcurl PROPERTY
                     INTERFACE_COMPILE_OPTIONS "${PC_CURL_CFLAGS_OTHER}")
      endif()
    else()
      if(CURL_USE_STATIC_LIBS AND MSVC)
         set_target_properties(CURL::libcurl PROPERTIES
             INTERFACE_LINK_LIBRARIES "normaliz.lib;ws2_32.lib;wldap32.lib")
      endif()
    endif()

  endif()
endif()

cmake_policy(POP)
