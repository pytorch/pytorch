# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindAVIFile
-----------

Finds `AVIFile <https://avifile.sourceforge.net/>`_ library and include paths:

.. code-block:: cmake

  find_package(AVIFile [...])

AVIFile is a set of libraries for i386 machines to use various AVI codecs.
Support is limited beyond Linux.  Windows provides native AVI support, and so
doesn't need this library.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``AVIFile_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether AVIFile was found.

``AVIFILE_LIBRARIES``
  The libraries to link against.

``AVIFILE_DEFINITIONS``
  Definitions to use when compiling.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``AVIFILE_INCLUDE_DIR``
  Directory containing ``avifile.h`` and other AVIFile headers.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``AVIFILE_FOUND``
  .. deprecated:: 4.2
    Use ``AVIFile_FOUND``, which has the same value.

  Boolean indicating whether AVIFile was found.

Examples
^^^^^^^^

Finding AVIFile and conditionally creating an interface :ref:`Imported Target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(AVIFile)

  if(AVIFile_FOUND AND NOT TARGET AVIFile::AVIFile)
    add_library(AVIFile::AVIFile INTERFACE IMPORTED)
    set_target_properties(
      AVIFile::AVIFile
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${AVIFILE_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${AVIFILE_LIBRARIES}"
        INTERFACE_COMPILE_DEFINITIONS "${AVIFILE_DEFINITIONS}"
    )
  endif()

  target_link_libraries(example PRIVATE AVIFile::AVIFile)
#]=======================================================================]

if (UNIX)

  find_path(AVIFILE_INCLUDE_DIR avifile.h PATH_SUFFIXES avifile/include include/avifile include/avifile-0.7)
  find_library(AVIFILE_AVIPLAY_LIBRARY aviplay aviplay-0.7 PATH_SUFFIXES avifile/lib)

endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  AVIFile
  REQUIRED_VARS AVIFILE_AVIPLAY_LIBRARY AVIFILE_INCLUDE_DIR
)

if (AVIFile_FOUND)
    set(AVIFILE_LIBRARIES ${AVIFILE_AVIPLAY_LIBRARY})
    set(AVIFILE_DEFINITIONS "")
endif()

mark_as_advanced(AVIFILE_INCLUDE_DIR AVIFILE_AVIPLAY_LIBRARY)
