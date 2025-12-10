CMAKE_MODULE_PATH
-----------------

:ref:`Semicolon-separated list <CMake Language Lists>` of directories,
represented using forward slashes, specifying a search path for CMake modules
to be loaded by the :command:`include` or :command:`find_package` commands
before checking the default modules that come with CMake. By default it is
empty. It is intended to be set by the project.

It's fairly common for a project to have a directory containing various
``*.cmake`` files to assist in development. Adding the directory to the
:variable:`CMAKE_MODULE_PATH` simplifies loading them. For example, a
project's top-level ``CMakeLists.txt`` file may contain:

.. code-block:: cmake

  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

  include(Foo) # Loads ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Foo.cmake

  find_package(Bar) # Loads ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindBar.cmake
