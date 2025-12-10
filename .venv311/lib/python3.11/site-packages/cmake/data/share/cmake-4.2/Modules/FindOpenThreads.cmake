# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenThreads
---------------

Finds the OpenThreads C++ based threading library:

.. code-block:: cmake

  find_package(OpenThreads [...])

OpenThreads header files are intended to be included as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <OpenThreads/Thread>

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OpenThreads_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the OpenThreads library was found.

``OPENTHREADS_LIBRARY``
  Libraries needed to link against to use OpenThreads.  This provides either
  release (optimized) or debug library variant, which are found separately
  depending on the project's :ref:`Build Configurations`.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OPENTHREADS_INCLUDE_DIR``
  The directory containing the header files needed to use OpenThreads.

Hints
^^^^^

This module accepts the following variables:

``OPENTHREADS_DIR``
  An environment or CMake variable that can be set to help find an OpenThreads
  library installed in a custom location.  It should point to the installation
  destination that was used when configuring, building, and installing
  OpenThreads library: ``./configure --prefix=$OPENTHREADS_DIR``.

This module was originally introduced to support the
:module:`FindOpenSceneGraph` module and its components.  To simplify one-step
automated configuration and builds when the OpenSceneGraph package is developed
and distributed upstream, this module supports additional environment variables
to find dependencies in specific locations.  This approach is used by upstream
package over specifying ``-DVAR=value`` on the command line because it offers
better isolation from internal changes to the module and allows more flexibility
when specifying individual OSG components independently of the ``CMAKE_*_PATH``
variables.  Explicit ``-DVAR=value`` arguments can still override these settings
if needed.  Since OpenThreads is an optional standalone dependency of
OpenSceneGraph, this module also honors the following variables for convenience:

``OSG_DIR``
  May be set as an environment or CMake variable. Treated the same as
  ``OPENTHREADS_DIR``.

``OSGDIR``
  Environment variable treated the same as ``OPENTHREADS_DIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OPENTHREADS_FOUND``
  .. deprecated:: 4.2
    Use ``OpenThreads_FOUND``, which has the same value.

  Boolean indicating whether the OpenThreads library was found.

Examples
^^^^^^^^

Finding the OpenThreads library and creating an interface :ref:`imported target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(OpenThreads)

  if(OpenThreads_FOUND AND NOT TARGET OpenThreads::OpenThreads)
    add_library(OpenThreads::OpenThreads INTERFACE IMPORTED)
    set_target_properties(
      OpenThreads::OpenThreads
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENTHREADS_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OPENTHREADS_LIBRARY}"
    )
  endif()

  target_link_libraries(example PRIVATE OpenThreads::OpenThreads)
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)

find_path(OPENTHREADS_INCLUDE_DIR OpenThreads/Thread
    HINTS
        ENV OPENTHREADS_INCLUDE_DIR
        ENV OPENTHREADS_DIR
        ENV OSG_INCLUDE_DIR
        ENV OSG_DIR
        ENV OSGDIR
        ENV OpenThreads_ROOT
        ENV OSG_ROOT
        ${OPENTHREADS_DIR}
        ${OSG_DIR}
    PATH_SUFFIXES include
)

find_library(OPENTHREADS_LIBRARY_RELEASE
    NAMES OpenThreads OpenThreadsWin32
    HINTS
        ENV OPENTHREADS_LIBRARY_DIR
        ENV OPENTHREADS_DIR
        ENV OSG_LIBRARY_DIR
        ENV OSG_DIR
        ENV OSGDIR
        ENV OpenThreads_ROOT
        ENV OSG_ROOT
        ${OPENTHREADS_DIR}
        ${OSG_DIR}
    PATH_SUFFIXES lib
)

find_library(OPENTHREADS_LIBRARY_DEBUG
    NAMES OpenThreadsd OpenThreadsWin32d
    HINTS
        ENV OPENTHREADS_DEBUG_LIBRARY_DIR
        ENV OPENTHREADS_LIBRARY_DIR
        ENV OPENTHREADS_DIR
        ENV OSG_LIBRARY_DIR
        ENV OSG_DIR
        ENV OSGDIR
        ENV OpenThreads_ROOT
        ENV OSG_ROOT
        ${OPENTHREADS_DIR}
        ${OSG_DIR}
    PATH_SUFFIXES lib
)

select_library_configurations(OPENTHREADS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenThreads DEFAULT_MSG
    OPENTHREADS_LIBRARY OPENTHREADS_INCLUDE_DIR)
