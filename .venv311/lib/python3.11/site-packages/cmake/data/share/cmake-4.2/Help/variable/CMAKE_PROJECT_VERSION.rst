CMAKE_PROJECT_VERSION
---------------------

.. versionadded:: 3.12

The version of the top level project.

This variable holds the version of the project as specified in the top
level CMakeLists.txt file by a :command:`project` command.  In the event that
the top level CMakeLists.txt contains multiple :command:`project` calls,
the most recently called one from that top level CMakeLists.txt will determine
the value that ``CMAKE_PROJECT_VERSION`` contains.  For example, consider
the following top level CMakeLists.txt:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.0)
  project(First VERSION 1.2.3)
  project(Second VERSION 3.4.5)
  add_subdirectory(sub)
  project(Third VERSION 6.7.8)

And ``sub/CMakeLists.txt`` with the following contents:

.. code-block:: cmake

  project(SubProj VERSION 1)
  message("CMAKE_PROJECT_VERSION = ${CMAKE_PROJECT_VERSION}")

The most recently seen :command:`project` command from the top level
CMakeLists.txt would be ``project(Second ...)``, so this will print::

  CMAKE_PROJECT_VERSION = 3.4.5

To obtain the version from the most recent call to :command:`project` in
the current directory scope or above, see the :variable:`PROJECT_VERSION`
variable.
