ALIAS_GLOBAL
------------

.. versionadded:: 3.18

Read-only property indicating of whether an :ref:`ALIAS target <Alias Targets>`
is globally visible.

The boolean value of this property is ``TRUE`` for aliases to
:ref:`IMPORTED targets <Imported Targets>` created
with the ``GLOBAL`` options to :command:`add_executable()` or
:command:`add_library()`, ``FALSE`` otherwise. It is undefined for
targets built within the project.

.. note::

  Promoting an :ref:`IMPORTED target <Imported Targets>` from ``LOCAL``
  to ``GLOBAL`` scope by changing the value or :prop_tgt:`IMPORTED_GLOBAL`
  target property do not change the scope of local aliases.
