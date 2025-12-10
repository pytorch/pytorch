# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindXMLRPC
----------

Finds the native XML-RPC library for C and C++:

.. code-block:: cmake

  find_package(XMLRPC [...] [COMPONENTS <components>...] [...])

XML-RPC is a standard network protocol that enables remote procedure calls
(RPC) between systems.  It encodes requests and responses in XML and uses
HTTP as the transport mechanism.

Components
^^^^^^^^^^

The XML-RPC C/C++ library consists of various features (modules) that provide
specific functionality.  The availability of these features depends on the
installed XML-RPC library version and system configuration.  Some features also
have dependencies on others.

To list the available features on a system, the ``xmlrpc-c-config`` command-line
utility can be used.

In CMake, these features can be specified as components with the
:command:`find_package` command:

.. code-block:: cmake

  find_package(XMLRPC [COMPONENTS <components>...])

Components may be:

``c++2``
  C++ wrapper API, replacing the legacy ``c++`` feature.
``c++``
  The legacy C++ wrapper API (superseded by ``c++2``).
``client``
  XML-RPC client functions (also available as the legacy libwww-based feature
  named ``libwww-client``).
``cgi-server``
  CGI-based server functions.
``abyss-server``
  Abyss-based server functions.
``pstream-server``
  The pstream-based server functions.
``server-util``
  Basic server functions (they are automatically included with ``*-server``
  features).
``abyss``
  Abyss HTTP server (not needed with ``abyss-server``).
``openssl``
  OpenSSL convenience functions.

If no components are specified, this module searches for XML-RPC library and
its include directories without additional features.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``XMLRPC_FOUND``
  Boolean indicating whether the XML-RPC library and all its requested
  components were found.
``XMLRPC_INCLUDE_DIRS``
  Include directories containing ``xmlrpc.h`` and other headers needed to use
  the XML-RPC library.
``XMLRPC_LIBRARIES``
  List of libraries needed for linking to XML-RPC library and its requested
  features.

Examples
^^^^^^^^

Finding XML-RPC library and its ``client`` feature, and conditionally
creating an interface :ref:`imported target <Imported Targets>` that
encapsulates its usage requirements for linking to a project target:

.. code-block:: cmake

  find_package(XMLRPC REQUIRED COMPONENTS client)

  if(XMLRPC_FOUND AND NOT TARGET XMLRPC::XMLRPC)
    add_library(XMLRPC::XMLRPC INTERFACE IMPORTED)
    set_target_properties(
      XMLRPC::XMLRPC
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${XMLRPC_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${XMLRPC_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE XMLRPC::XMLRPC)
#]=======================================================================]

# First find the config script from which to obtain other values.
find_program(XMLRPC_C_CONFIG NAMES xmlrpc-c-config)

# Check whether we found anything.
if(XMLRPC_C_CONFIG)
  set(XMLRPC_C_FOUND 1)
else()
  set(XMLRPC_C_FOUND 0)
endif()

# Lookup the include directories needed for the components requested.
if(XMLRPC_C_FOUND)
  execute_process(
    COMMAND ${XMLRPC_C_CONFIG} ${XMLRPC_FIND_COMPONENTS} --cflags
    OUTPUT_VARIABLE XMLRPC_C_CONFIG_CFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE XMLRPC_C_CONFIG_RESULT
    )

  # Parse the include flags.
  if("${XMLRPC_C_CONFIG_RESULT}" STREQUAL "0")
    # Convert the compile flags to a CMake list.
    string(REGEX REPLACE " +" ";"
      XMLRPC_C_CONFIG_CFLAGS "${XMLRPC_C_CONFIG_CFLAGS}")

    # Look for -I options.
    # FIXME: Use these as hints to a find_path call to find the headers.
    set(XMLRPC_INCLUDE_DIRS)
    foreach(flag ${XMLRPC_C_CONFIG_CFLAGS})
      if("${flag}" MATCHES "^-I(.+)")
        file(TO_CMAKE_PATH "${CMAKE_MATCH_1}" DIR)
        list(APPEND XMLRPC_INCLUDE_DIRS "${DIR}")
      endif()
    endforeach()
  else()
    message("Error running ${XMLRPC_C_CONFIG}: [${XMLRPC_C_CONFIG_RESULT}]")
    set(XMLRPC_C_FOUND 0)
  endif()
endif()

# Lookup the libraries needed for the components requested.
if(XMLRPC_C_FOUND)
  execute_process(
    COMMAND ${XMLRPC_C_CONFIG} ${XMLRPC_FIND_COMPONENTS} --libs
    OUTPUT_VARIABLE XMLRPC_C_CONFIG_LIBS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE XMLRPC_C_CONFIG_RESULT
    )

  # Parse the library names and directories.
  if("${XMLRPC_C_CONFIG_RESULT}" STREQUAL "0")
    string(REGEX REPLACE " +" ";"
      XMLRPC_C_CONFIG_LIBS "${XMLRPC_C_CONFIG_LIBS}")

    # Look for -L flags for directories and -l flags for library names.
    set(XMLRPC_LIBRARY_DIRS)
    set(XMLRPC_LIBRARY_NAMES)
    foreach(flag ${XMLRPC_C_CONFIG_LIBS})
      if("${flag}" MATCHES "^-L(.+)")
        file(TO_CMAKE_PATH "${CMAKE_MATCH_1}" DIR)
        list(APPEND XMLRPC_LIBRARY_DIRS "${DIR}")
      elseif("${flag}" MATCHES "^-l(.+)")
        list(APPEND XMLRPC_LIBRARY_NAMES "${CMAKE_MATCH_1}")
      endif()
    endforeach()

    # Search for each library needed using the directories given.
    foreach(name ${XMLRPC_LIBRARY_NAMES})
      # Look for this library.
      find_library(XMLRPC_${name}_LIBRARY
        NAMES ${name}
        HINTS ${XMLRPC_LIBRARY_DIRS}
        )
      mark_as_advanced(XMLRPC_${name}_LIBRARY)

      # If any library is not found then the whole package is not found.
      if(NOT XMLRPC_${name}_LIBRARY)
        set(XMLRPC_C_FOUND 0)
      endif()

      # Build an ordered list of all the libraries needed.
      set(XMLRPC_LIBRARIES ${XMLRPC_LIBRARIES} "${XMLRPC_${name}_LIBRARY}")
    endforeach()
  else()
    message("Error running ${XMLRPC_C_CONFIG}: [${XMLRPC_C_CONFIG_RESULT}]")
    set(XMLRPC_C_FOUND 0)
  endif()
endif()

# Report the results.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    XMLRPC
    REQUIRED_VARS XMLRPC_C_FOUND XMLRPC_LIBRARIES
    FAIL_MESSAGE "XMLRPC was not found. Make sure the entries XMLRPC_* are set.")
