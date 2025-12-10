VS_NO_SOLUTION_DEPLOY
---------------------

.. versionadded:: 3.15

Specify that the target should not be marked for deployment to a Windows CE
or Windows Phone device in the generated Visual Studio solution.

Be default, all EXE and shared library (DLL) targets are marked to deploy to
the target device in the generated Visual Studio solution.

Generator expressions are supported.

There are reasons one might want to exclude a target / generated project from
deployment:

- The library or executable may not be necessary in the primary deploy/debug
  scenario, and excluding from deployment saves time in the
  develop/download/debug cycle.
- There may be insufficient space on the target device to accommodate all of
  the build products.
- Visual Studio 2013 requires a target device IP address be entered for each
  target marked for deployment.  For large numbers of targets, this can be
  tedious.
  NOTE: Visual Studio *will* deploy all project dependencies of a project
  tagged for deployment to the IP address configured for that project even
  if those dependencies are not tagged for deployment.


Example 1
^^^^^^^^^

This shows setting the variable for the target foo.

.. code-block:: cmake

  add_library(foo SHARED foo.cpp)
  set_property(TARGET foo PROPERTY VS_NO_SOLUTION_DEPLOY ON)

Example 2
^^^^^^^^^

This shows setting the variable for the Release configuration only.

.. code-block:: cmake

  add_library(foo SHARED foo.cpp)
  set_property(TARGET foo PROPERTY VS_NO_SOLUTION_DEPLOY "$<CONFIG:Release>")
