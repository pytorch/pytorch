DISABLED_FEATURES
-----------------

List of features which are disabled during the CMake run.

List of features which are disabled during the CMake run.  By default
it contains the names of all packages which were not found.  This is
determined using the ``<NAME>_FOUND`` variables.  Packages which are
searched ``QUIET`` are not listed.  A project can add its own features to
this list.  This property is used by the macros in
:module:`FeatureSummary` module.
