CMAKE_LINK_GROUP_USING_<FEATURE>
--------------------------------

.. versionadded:: 3.24

This variable defines how to link a group of libraries for the specified
``<FEATURE>`` when a :genex:`LINK_GROUP` generator expression is used.
Both of the following conditions must be met for this variable to have any
effect:

* The associated :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>_SUPPORTED`
  variable must be set to true.

* There is no language-specific definition for the same ``<FEATURE>``.
  This means :variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>_SUPPORTED`
  cannot be true for the link language used by the target for which the
  :genex:`LINK_GROUP` generator expression is evaluated.

The :variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>` variable should be
defined instead for features that are dependent on the link language.

.. include:: include/CMAKE_LINK_GROUP_USING_FEATURE.rst
