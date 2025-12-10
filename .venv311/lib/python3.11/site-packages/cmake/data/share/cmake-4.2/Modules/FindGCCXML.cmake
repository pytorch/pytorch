# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGCCXML
----------

.. versionchanged:: 4.1
  This module is available only if policy :policy:`CMP0188` is not set to ``NEW``.
  Port projects to search for CastXML by calling :command:`find_program` directly.

Finds the GCC-XML front-end executable:

.. code-block:: cmake

  find_package(GCCXML [...])

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GCCXML``
  The GCC-XML front-end executable.

Examples
^^^^^^^^

In new code, find CastXML with :command:`find_program` instead:

.. code-block:: cmake

  find_program(CASTXML_EXECUTABLE castxml)
#]=======================================================================]

cmake_policy(GET CMP0188 _FindGCCXML_CMP0188)
if(_FindGCCXML_CMP0188 STREQUAL "NEW")
  message(FATAL_ERROR "The FindGCCXML module has been removed by policy CMP0188.")
endif()

if(_FindGCCXML_testing)
  set(_FindGCCXML_included TRUE)
  return()
endif()

find_program(GCCXML
  NAMES gccxml
        ../GCC_XML/gccxml
  PATHS [HKEY_CURRENT_USER\\Software\\Kitware\\GCC_XML;loc]
  "$ENV{ProgramFiles}/GCC_XML"
  "C:/Program Files/GCC_XML"
)

mark_as_advanced(GCCXML)
