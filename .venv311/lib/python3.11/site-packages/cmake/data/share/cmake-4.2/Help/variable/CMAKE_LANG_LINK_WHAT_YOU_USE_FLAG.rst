CMAKE_<LANG>_LINK_WHAT_YOU_USE_FLAG
-----------------------------------

.. versionadded:: 3.22

Linker flag used by :prop_tgt:`LINK_WHAT_YOU_USE` to tell the linker to
link all shared libraries specified on the command line even if none
of their symbols is needed.  This is an implementation detail used so
that the command in :variable:`CMAKE_LINK_WHAT_YOU_USE_CHECK` can check
the binary for unnecessarily-linked shared libraries.

.. note::

  Do not rely on this abstraction to intentionally link to
  shared libraries whose symbols are not needed.
