CMAKE_DISABLE_FIND_PACKAGE_<PackageName>
----------------------------------------

Variable for disabling :command:`find_package` calls.

Every non-``REQUIRED`` :command:`find_package` call in a project can be
disabled by setting the variable
``CMAKE_DISABLE_FIND_PACKAGE_<PackageName>`` to ``TRUE``.
This can be used to build a project without an optional package,
although that package is installed.

This switch should be used during the initial CMake run.  Otherwise if
the package has already been found in a previous CMake run, the
variables which have been stored in the cache will still be there.  In
that case it is recommended to remove the cache variables for this
package from the cache using the cache editor or :option:`cmake -U`.

Note that this variable can lead to inconsistent results within the project.
Consider the case where a dependency is requested via :command:`find_package`
from two different places within the project.  If the first call does not
have the ``REQUIRED`` keyword, it will not find the dependency when
``CMAKE_DISABLE_FIND_PACKAGE_<PackageName>`` is set to true for that
dependency.  The project will proceed under the assumption that the dependency
isn't available.  If the second call elsewhere in the project *does* have the
``REQUIRED`` keyword, it can succeed.  Two different parts of the same project
have then seen opposite results for the same dependency.

See also the :variable:`CMAKE_REQUIRE_FIND_PACKAGE_<PackageName>` variable.
