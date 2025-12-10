Option De-duplication
^^^^^^^^^^^^^^^^^^^^^

The final set of options used for a target is constructed by
accumulating options from the current target and the usage requirements of
its dependencies.  The set of options is de-duplicated to avoid repetition.

.. versionadded:: 3.12
  While beneficial for individual options, the de-duplication step can break
  up option groups.  For example, ``-option A -option B`` becomes
  ``-option A B``.  One may specify a group of options using shell-like
  quoting along with a ``SHELL:`` prefix.  The ``SHELL:`` prefix is dropped,
  and the rest of the option string is parsed using the
  :command:`separate_arguments` ``UNIX_COMMAND`` mode. For example,
  ``"SHELL:-option A" "SHELL:-option B"`` becomes ``-option A -option B``.
