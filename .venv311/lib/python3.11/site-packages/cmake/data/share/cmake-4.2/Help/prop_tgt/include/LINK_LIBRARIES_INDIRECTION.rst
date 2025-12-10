.. note::
  A call to :command:`target_link_libraries(<target> ...)` may update this
  property on ``<target>``.  If ``<target>`` was not created in the same
  directory as the call then :command:`target_link_libraries` will wrap each
  entry with the form ``::@(directory-id);...;::@``, where the ``::@`` is
  literal and the ``(directory-id)`` is unspecified.
  This tells the generators that the named libraries must be looked up in
  the scope of the caller rather than in the scope in which the
  ``<target>`` was created.  Valid directory ids are stripped on export
  by the :command:`install(EXPORT)` and :command:`export` commands.
