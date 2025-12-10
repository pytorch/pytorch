CMAKE_LINK_LIBRARY_USING_<FEATURE>
----------------------------------

.. versionadded:: 3.24

This variable defines how to link a library or framework for the specified
``<FEATURE>`` when a :genex:`LINK_LIBRARY` generator expression is used.
Both of the following conditions must be met for this variable to have any
effect:

* The associated :variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED`
  variable must be set to true.

* There is no language-specific definition for the same ``<FEATURE>``.
  This means :variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED`
  cannot be true for the link language used by the target for which the
  :genex:`LINK_LIBRARY` generator expression is evaluated.

.. include:: include/CMAKE_LINK_LIBRARY_USING_FEATURE.rst
