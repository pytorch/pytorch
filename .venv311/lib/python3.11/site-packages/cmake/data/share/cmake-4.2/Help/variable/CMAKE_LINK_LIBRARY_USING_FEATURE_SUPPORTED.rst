CMAKE_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED
--------------------------------------------

.. versionadded:: 3.24

Set to ``TRUE`` if the ``<FEATURE>``, as defined by variable
:variable:`CMAKE_LINK_LIBRARY_USING_<FEATURE>`, is supported regardless the
linker language.

.. note::

  This variable is evaluated if, and only if, the variable
  :variable:`CMAKE_<LANG>_LINK_LIBRARY_USING_<FEATURE>_SUPPORTED` is not
  defined.
