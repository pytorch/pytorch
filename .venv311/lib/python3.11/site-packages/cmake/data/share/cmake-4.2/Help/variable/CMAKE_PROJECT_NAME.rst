CMAKE_PROJECT_NAME
------------------

The name of the top level project.

This variable holds the name of the project as specified in the top
level CMakeLists.txt file by a :command:`project` command.  In the event that
the top level CMakeLists.txt contains multiple :command:`project` calls,
the most recently called one from that top level CMakeLists.txt will determine
the name that ``CMAKE_PROJECT_NAME`` contains.  For example, consider
the following top level CMakeLists.txt:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.0)
  project(First)
  project(Second)
  add_subdirectory(sub)
  project(Third)

And ``sub/CMakeLists.txt`` with the following contents:

.. code-block:: cmake

  project(SubProj)
  message("CMAKE_PROJECT_NAME = ${CMAKE_PROJECT_NAME}")

The most recently seen :command:`project` command from the top level
CMakeLists.txt would be ``project(Second)``, so this will print::

  CMAKE_PROJECT_NAME = Second

To obtain the name from the most recent call to :command:`project` in
the current directory scope or above, see the :variable:`PROJECT_NAME`
variable.
