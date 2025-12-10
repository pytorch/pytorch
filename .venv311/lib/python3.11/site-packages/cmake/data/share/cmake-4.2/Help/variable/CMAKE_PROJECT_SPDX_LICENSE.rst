CMAKE_PROJECT_SPDX_LICENSE
--------------------------

.. versionadded:: 4.2

.. note::

  Experimental. Gated by ``CMAKE_EXPERIMENTAL_EXPORT_PACKAGE_INFO``.

The license(s) of the top level project.

This variable holds the license expression of the project as specified in the
top level CMakeLists.txt file by a :command:`project` command.  In the event
that the top level CMakeLists.txt contains multiple :command:`project` calls,
the most recently called one from that top level CMakeLists.txt will determine
the value that ``CMAKE_PROJECT_SPDX_LICENSE`` contains.  For example, consider
the following top level CMakeLists.txt:

.. code-block:: cmake

  cmake_minimum_required(VERSION 4.2)
  project(First SPDX_LICENSE "BSD-3-Clause")
  project(Second SPDX_LICENSE "BSD-3-Clause AND CC-BY-SA-4.0")
  add_subdirectory(sub)
  project(Third SPDX_LICENSE "BSD-3-Clause AND CC0-1.0")

And ``sub/CMakeLists.txt`` with the following contents:

.. code-block:: cmake

  project(SubProj SPDX_LICENSE Apache-2.0)
  message("CMAKE_PROJECT_SPDX_LICENSE = ${CMAKE_PROJECT_SPDX_LICENSE}")

The most recently seen :command:`project` command from the top level
CMakeLists.txt would be ``project(Second ...)``, so this will print::

  CMAKE_PROJECT_SPDX_LICENSE = BSD-3-Clause AND CC-BY-SA-4.0

To obtain the version from the most recent call to :command:`project` in
the current directory scope or above, see the :variable:`PROJECT_SPDX_LICENSE`
variable.
