CMAKE_DL_LIBS
-------------

This variable contains a name of the dynamic loading library, or a list of
system libraries needed to use the ``dlopen()`` and ``dlclose()`` functions.

Few examples of the values this variable is set to:

``dl``
  On most Unix-like systems.

``-lld``
  On AIX, prior to CMake 4.2.

``dld``
  On HP-UX.

""
  Empty string value or not set on systems that have ``dl*()`` functions
  either in the default library that is implicitly linked (e.g., BSD-like
  systems, Haiku, macOS, SunOS, etc.), or on systems that don't provide these
  functions (e.g., Windows).

Examples
^^^^^^^^

Example: Linking Dynamic Loading Library
""""""""""""""""""""""""""""""""""""""""

Using this variable in a project that uses dynamic loading functionality:

.. code-block:: cmake

  target_link_libraries(example PRIVATE ${CMAKE_DL_LIBS})

Example: Checking Symbols
"""""""""""""""""""""""""

Checking for symbols with the dynamic loading library linked during the check:

.. code-block:: cmake

  include(CheckSymbolExists)
  include(CMakePushCheckState)

  cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
    check_symbol_exists(dlopen "dlfcn.h" HAVE_DLOPEN)
  cmake_pop_check_state()
