CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>_SUPPORTED
-------------------------------------------------

.. versionadded:: 3.24

This variable specifies whether the ``<FEATURE>`` is supported for the link
language ``<LANG>``.  If this variable is true, then the ``<FEATURE>`` must
be defined by :variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>`, and the
more generic :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>_SUPPORTED` and
:variable:`CMAKE_LINK_GROUP_USING_<FEATURE>` variables are not used.

If ``CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>_SUPPORTED`` is false or is not
set, then the :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>_SUPPORTED` variable
will determine whether ``<FEATURE>`` is deemed to be supported.
