CMAKE_PROJECT_DESCRIPTION
-------------------------

.. versionadded:: 3.9

The description of the top level project.

This variable holds the description of the project as specified in the top
level CMakeLists.txt file by a :command:`project` command.  In the event that
the top level CMakeLists.txt contains multiple :command:`project` calls,
the most recently called one from that top level CMakeLists.txt will determine
the value that ``CMAKE_PROJECT_DESCRIPTION`` contains.  For example, consider
the following top level CMakeLists.txt:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.0)
  project(First DESCRIPTION "I am First")
  project(Second DESCRIPTION "I am Second")
  add_subdirectory(sub)
  project(Third DESCRIPTION "I am Third")

And ``sub/CMakeLists.txt`` with the following contents:

.. code-block:: cmake

  project(SubProj DESCRIPTION "I am SubProj")
  message("CMAKE_PROJECT_DESCRIPTION = ${CMAKE_PROJECT_DESCRIPTION}")

The most recently seen :command:`project` command from the top level
CMakeLists.txt would be ``project(Second ...)``, so this will print::

  CMAKE_PROJECT_DESCRIPTION = I am Second

To obtain the description from the most recent call to :command:`project` in
the current directory scope or above, see the :variable:`PROJECT_DESCRIPTION`
variable.
