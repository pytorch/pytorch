CMAKE_LINK_GROUP_USING_<FEATURE>_SUPPORTED
------------------------------------------

.. versionadded:: 3.24

This variable specifies whether the ``<FEATURE>`` is supported regardless of
the link language.  If this variable is true, then the ``<FEATURE>`` must
be defined by :variable:`CMAKE_LINK_GROUP_USING_<FEATURE>`.

Note that this variable has no effect if
:variable:`CMAKE_<LANG>_LINK_GROUP_USING_<FEATURE>_SUPPORTED` is true for
the link language of the target.
