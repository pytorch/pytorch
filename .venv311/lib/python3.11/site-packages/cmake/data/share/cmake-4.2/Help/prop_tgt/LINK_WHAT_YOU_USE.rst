LINK_WHAT_YOU_USE
-----------------

.. versionadded:: 3.7

This is a boolean option that, when set to ``TRUE``, adds a link-time check
to print a list of shared libraries that are being linked but provide no symbols
used by the target.  This is intended as a lint.

The flag specified by :variable:`CMAKE_<LANG>_LINK_WHAT_YOU_USE_FLAG` will
be passed to the linker so that all libraries specified on the command line
will be linked into the target.  Then the command specified by
:variable:`CMAKE_LINK_WHAT_YOU_USE_CHECK` will run after the target is linked
to check the binary for unnecessarily-linked shared libraries.

.. note::

  For now, it is only supported for ``ELF`` platforms and is only applicable to
  executable and shared or module library targets. This property will be
  ignored for any other targets and configurations.

This property is initialized by the value of
the :variable:`CMAKE_LINK_WHAT_YOU_USE` variable if it is set
when a target is created.
