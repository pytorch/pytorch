CMAKE_POLICY_DEFAULT_CMP<NNNN>
------------------------------

Default for CMake Policy ``CMP<NNNN>`` when it is otherwise left unset.

Commands :command:`cmake_minimum_required(VERSION)` and
:command:`cmake_policy(VERSION)` by default leave policies introduced after
the given version unset.  Set ``CMAKE_POLICY_DEFAULT_CMP<NNNN>`` to ``OLD``
or ``NEW`` to specify the default for policy ``CMP<NNNN>``, where ``<NNNN>``
is the policy number.

This variable should not be set by a project in CMake code as a way to
set its own policies; use :command:`cmake_policy(SET)` instead.  This
variable is meant to externally set policies for which a project has
not itself been updated:

* Users running CMake may set this variable in the cache
  (e.g. ``-DCMAKE_POLICY_DEFAULT_CMP<NNNN>=<OLD|NEW>``).  Set it to ``OLD``
  to quiet a policy warning while using old behavior or to ``NEW`` to
  try building the project with new behavior.

* Projects may set this variable before a call to :command:`add_subdirectory`
  that adds a third-party project in order to set its policies without
  modifying third-party code.

See :variable:`CMAKE_POLICY_VERSION_MINIMUM` set policies to ``NEW``
based on the version of CMake that introduced them.
