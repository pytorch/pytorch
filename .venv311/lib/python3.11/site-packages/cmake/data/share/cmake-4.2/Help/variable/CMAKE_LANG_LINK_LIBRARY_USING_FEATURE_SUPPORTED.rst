CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED
---------------------------------------------------

.. versionadded:: 3.24

Set to ``TRUE`` if the ``<FEATURE>``, as defined by variable
:variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>`, is supported for the
linker language ``<LANG>``.

.. note::

  This variable is evaluated before the more generic variable
  :variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED`.
