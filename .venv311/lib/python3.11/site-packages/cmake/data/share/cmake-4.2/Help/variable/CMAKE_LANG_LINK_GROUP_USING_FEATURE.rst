CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>
---------------------------------------

.. versionadded:: 3.24

This variable defines how to link a group of libraries for the specified
``<FEATURE>`` when a :genex:`LINK_GROUP` generator expression is used and
the link language for the target is ``<LANG>``.
For this variable to have any effect, the associated
:variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>_SUPPORTED` variable
must be set to true.

The :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>` variable should be defined
instead for features that are independent of the link language.

.. include:: include/CMAKE_LINK_GROUP_USING_FEATURE.rst
