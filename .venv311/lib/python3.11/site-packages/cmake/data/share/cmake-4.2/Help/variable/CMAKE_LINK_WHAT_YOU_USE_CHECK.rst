CMAKE_LINK_WHAT_YOU_USE_CHECK
-----------------------------

.. versionadded:: 3.22

Command executed by :prop_tgt:`LINK_WHAT_YOU_USE` after the linker to
check for unnecessarily-linked shared libraries.
This check is currently only defined on ``ELF`` platforms with value
``ldd -u -r``.

See also :variable:`CMAKE_<LANG>_LINK_WHAT_YOU_USE_FLAG` variables.
