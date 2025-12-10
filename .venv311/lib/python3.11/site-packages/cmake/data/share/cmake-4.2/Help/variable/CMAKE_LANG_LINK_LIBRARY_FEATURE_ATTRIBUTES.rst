CMAKE_<LANG>_LINK_LIBRARY_<FEATURE>_ATTRIBUTES
----------------------------------------------

.. versionadded:: 3.30

This variable defines the semantics of the specified link library ``<FEATURE>``
when linking with the link language ``<LANG>``. It takes precedence over
:variable:`CMAKE_LINK_LIBRARY_<FEATURE>_ATTRIBUTES` if that variable is also
defined for the same ``<FEATURE>``, but otherwise has similar effects.
See :variable:`CMAKE_LINK_LIBRARY_<FEATURE>_ATTRIBUTES` for further details.
