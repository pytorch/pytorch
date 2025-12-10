CMAKE_FIND_PACKAGE_WARN_NO_MODULE
---------------------------------

Tell :command:`find_package` to warn if called without an explicit mode.

If :command:`find_package` is called without an explicit mode option
(``MODULE``, ``CONFIG``, or ``NO_MODULE``) and no ``Find<pkg>.cmake`` module
is in :variable:`CMAKE_MODULE_PATH` then CMake implicitly assumes that the
caller intends to search for a package configuration file.  If no package
configuration file is found then the wording of the failure message
must account for both the case that the package is really missing and
the case that the project has a bug and failed to provide the intended
Find module.  If instead the caller specifies an explicit mode option
then the failure message can be more specific.

Set ``CMAKE_FIND_PACKAGE_WARN_NO_MODULE`` to ``TRUE`` to tell
:command:`find_package` to warn when it implicitly assumes Config mode.  This
helps developers enforce use of an explicit mode in all calls to
:command:`find_package` within a project.

This variable has no effect if :variable:`CMAKE_FIND_PACKAGE_PREFER_CONFIG` is
set to ``TRUE``.
