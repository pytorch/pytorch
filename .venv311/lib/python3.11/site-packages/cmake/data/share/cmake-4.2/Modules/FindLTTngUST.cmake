# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLTTngUST
------------

.. versionadded:: 3.6

Finds the `LTTng <https://lttng.org/>`_ (Linux Trace Toolkit: next generation)
user space tracing library (LTTng-UST):

.. code-block:: cmake

  find_package(LTTngUST [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``LTTng::UST``
  Target providing the LTTng-UST library usage requirements.  This target is
  available only when LTTng-UST is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LTTngUST_FOUND``
  Boolean indicating whether the (requested version of) LTTng-UST library
  was found.

``LTTngUST_VERSION``
  .. versionadded:: 4.2

  The LTTng-UST version.

``LTTNGUST_HAS_TRACEF``
  ``TRUE`` if the ``tracef()`` API is available in the system's LTTng-UST.

``LTTNGUST_HAS_TRACELOG``
  ``TRUE`` if the ``tracelog()`` API is available in the system's LTTng-UST.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LTTNGUST_INCLUDE_DIRS``
  The LTTng-UST include directories.
``LTTNGUST_LIBRARIES``
  The libraries needed to use LTTng-UST.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``LTTNGUST_FOUND``
  .. deprecated:: 4.2
    Use ``LTTngUST_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) LTTng-UST library
  was found.

``LTTNGUST_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``LTTngUST_VERSION``.

  The LTTng-UST version.

Examples
^^^^^^^^

Finding the LTTng-UST library and linking it to a project target:

.. code-block:: cmake

  find_package(LTTugNST)
  target_link_libraries(project_target PRIVATE LTTng::UST)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(LTTNGUST_INCLUDE_DIRS NAMES lttng/tracepoint.h)
find_library(LTTNGUST_LIBRARIES NAMES lttng-ust)

if(LTTNGUST_INCLUDE_DIRS AND LTTNGUST_LIBRARIES)
  # find tracef() and tracelog() support
  set(LTTNGUST_HAS_TRACEF 0)
  set(LTTNGUST_HAS_TRACELOG 0)

  if(EXISTS "${LTTNGUST_INCLUDE_DIRS}/lttng/tracef.h")
    set(LTTNGUST_HAS_TRACEF TRUE)
  endif()

  if(EXISTS "${LTTNGUST_INCLUDE_DIRS}/lttng/tracelog.h")
    set(LTTNGUST_HAS_TRACELOG TRUE)
  endif()

  # get version
  set(lttngust_version_file "${LTTNGUST_INCLUDE_DIRS}/lttng/ust-version.h")

  if(EXISTS "${lttngust_version_file}")
    file(STRINGS "${lttngust_version_file}" lttngust_version_major_string
         REGEX "^[\t ]*#define[\t ]+LTTNG_UST_MAJOR_VERSION[\t ]+[0-9]+[\t ]*$")
    file(STRINGS "${lttngust_version_file}" lttngust_version_minor_string
         REGEX "^[\t ]*#define[\t ]+LTTNG_UST_MINOR_VERSION[\t ]+[0-9]+[\t ]*$")
    file(STRINGS "${lttngust_version_file}" lttngust_version_patch_string
         REGEX "^[\t ]*#define[\t ]+LTTNG_UST_PATCHLEVEL_VERSION[\t ]+[0-9]+[\t ]*$")
    string(REGEX REPLACE ".*[\t ]+([0-9]+).*" "\\1"
           lttngust_v_major "${lttngust_version_major_string}")
    string(REGEX REPLACE ".*[\t ]+([0-9]+).*" "\\1"
           lttngust_v_minor "${lttngust_version_minor_string}")
    string(REGEX REPLACE ".*[\t ]+([0-9]+).*" "\\1"
           lttngust_v_patch "${lttngust_version_patch_string}")
    set(LTTngUST_VERSION
        "${lttngust_v_major}.${lttngust_v_minor}.${lttngust_v_patch}")
    set(LTTNGUST_VERSION_STRING "${LTTngUST_VERSION}")
    unset(lttngust_version_major_string)
    unset(lttngust_version_minor_string)
    unset(lttngust_version_patch_string)
    unset(lttngust_v_major)
    unset(lttngust_v_minor)
    unset(lttngust_v_patch)
  endif()

  unset(lttngust_version_file)

  if(NOT TARGET LTTng::UST)
    add_library(LTTng::UST UNKNOWN IMPORTED)
    set_target_properties(LTTng::UST PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${LTTNGUST_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES ${CMAKE_DL_LIBS}
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      IMPORTED_LOCATION "${LTTNGUST_LIBRARIES}")
  endif()

  # add libdl to required libraries
  set(LTTNGUST_LIBRARIES ${LTTNGUST_LIBRARIES} ${CMAKE_DL_LIBS})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LTTngUST
                                  REQUIRED_VARS LTTNGUST_LIBRARIES
                                                LTTNGUST_INCLUDE_DIRS
                                  VERSION_VAR LTTngUST_VERSION)
mark_as_advanced(LTTNGUST_LIBRARIES LTTNGUST_INCLUDE_DIRS)

cmake_policy(POP)
