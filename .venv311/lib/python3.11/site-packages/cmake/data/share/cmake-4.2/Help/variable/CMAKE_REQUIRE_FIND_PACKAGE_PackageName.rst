CMAKE_REQUIRE_FIND_PACKAGE_<PackageName>
----------------------------------------

.. versionadded:: 3.22

Variable for making :command:`find_package` call ``REQUIRED``.

Every non-``REQUIRED`` :command:`find_package` call in a project can be
turned into ``REQUIRED`` by setting the variable
``CMAKE_REQUIRE_FIND_PACKAGE_<PackageName>`` to ``TRUE``.
This can be used to assert assumptions about build environment and to
ensure the build will fail early if they do not hold.

Note that setting this variable to true breaks some commonly used patterns.
Multiple calls to :command:`find_package` are sometimes used to obtain a
different search order to the default.
For example, projects can force checking a known path for a particular
package first before searching any of the other default search paths:

.. code:: cmake

  find_package(something PATHS /some/local/path NO_DEFAULT_PATH)
  find_package(something)

In the above, the first call looks for the ``something`` package in a specific
directory.  If ``CMAKE_REQUIRE_FIND_PACKAGE_something`` is set to true, then
this first call must succeed, otherwise a fatal error occurs.  The second call
never gets a chance to provide a fall-back to using the default search
locations.

A similar pattern is used even by some of CMake's own Find modules to search
for a config package first:

.. code:: cmake

  find_package(something CONFIG QUIET)
  if(NOT something_FOUND)
    # Fall back to searching using typical Find module logic...
  endif()

Again, if ``CMAKE_REQUIRE_FIND_PACKAGE_something`` is true, the first call
must succeed.  It effectively means a config package must be found for the
dependency, and the Find module logic is never used.

See also the :variable:`CMAKE_DISABLE_FIND_PACKAGE_<PackageName>` variable.
