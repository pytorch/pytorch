AUTOGEN_ORIGIN_DEPENDS
----------------------

.. versionadded:: 3.14

Switch for forwarding origin target dependencies to the corresponding
:ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target.

  .. note::

    If Qt 5.15 or later is used and the generator is either :generator:`Ninja`
    or :ref:`Makefile Generators`, origin target dependencies are forwarded to
    the :ref:`<ORIGIN>_autogen_timestamp_deps <<ORIGIN>_autogen_timestamp_deps>`
    target instead of :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`.

Targets which have their :prop_tgt:`AUTOMOC` or :prop_tgt:`AUTOUIC` property
``ON`` have a corresponding :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target
which generates ``moc`` and ``uic`` files.
As this :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target is created at
generate-time, it is not possible to define dependencies of it using
e.g.  :command:`add_dependencies`.  Instead the ``AUTOGEN_ORIGIN_DEPENDS``
target property decides whether the origin target dependencies should be
forwarded to the :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target or not.

By default ``AUTOGEN_ORIGIN_DEPENDS`` is initialized from
:variable:`CMAKE_AUTOGEN_ORIGIN_DEPENDS` which is ``ON`` by default.

In total the dependencies of the :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`
target are composed from

- forwarded origin target dependencies
  (enabled by default via ``AUTOGEN_ORIGIN_DEPENDS``)
- additional user defined dependencies from :prop_tgt:`AUTOGEN_TARGET_DEPENDS`

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

.. note::

    Disabling ``AUTOGEN_ORIGIN_DEPENDS`` is useful to avoid building of
    origin target dependencies when building the
    :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target only.
    This is especially interesting when a
    :variable:`global autogen target <CMAKE_GLOBAL_AUTOGEN_TARGET>` is enabled.

    When the :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target doesn't require
    all the origin target's dependencies, and ``AUTOGEN_ORIGIN_DEPENDS`` is
    disabled, it might be necessary to extend :prop_tgt:`AUTOGEN_TARGET_DEPENDS`
    to add missing dependencies.
