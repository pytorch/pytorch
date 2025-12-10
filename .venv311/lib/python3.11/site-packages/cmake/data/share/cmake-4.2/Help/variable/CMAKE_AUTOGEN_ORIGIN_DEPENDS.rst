CMAKE_AUTOGEN_ORIGIN_DEPENDS
----------------------------

.. versionadded:: 3.14

Switch for forwarding origin target dependencies to the corresponding
:ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` targets.

  .. note::

    If Qt 5.15 or later is used and the generator is either :generator:`Ninja`
    or :ref:`Makefile Generators`, additional target dependencies are added to
    the :ref:`<ORIGIN>_autogen_timestamp_deps <<ORIGIN>_autogen_timestamp_deps>`
    target instead of the :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target.

This variable is used to initialize the :prop_tgt:`AUTOGEN_ORIGIN_DEPENDS`
property on all the targets.  See that target property for additional
information.

By default ``CMAKE_AUTOGEN_ORIGIN_DEPENDS`` is ``ON``.
