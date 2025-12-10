SYMBOLIC
--------

.. versionadded:: 4.2

Read-only indication of whether a target is ``SYMBOLIC``.

Symbolic targets are created by calls to
:command:`add_library(INTERFACE SYMBOLIC) <add_library(INTERFACE-SYMBOLIC)>`.
They are useful for packages to represent additional **components** or
**feature selectors** that consumers can request via ``find_package()``.
