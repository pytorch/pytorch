CMAKE_GLOBAL_AUTOGEN_TARGET
---------------------------

.. versionadded:: 3.14

Switch to enable generation of a global ``autogen`` target.

When ``CMAKE_GLOBAL_AUTOGEN_TARGET`` is enabled, a custom target
``autogen`` is generated.  This target depends on all :prop_tgt:`AUTOMOC` and
:prop_tgt:`AUTOUIC` generated :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>`
targets in the project.  By building the global ``autogen`` target, all
:prop_tgt:`AUTOMOC` and :prop_tgt:`AUTOUIC` files in the project will be
generated.

The name of the global ``autogen`` target can be changed by setting
:variable:`CMAKE_GLOBAL_AUTOGEN_TARGET_NAME`.

By default ``CMAKE_GLOBAL_AUTOGEN_TARGET`` is unset.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

.. note::

    :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` targets by default inherit their
    origin target's dependencies. This might result in unintended dependency
    target builds when only :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` targets
    are built.  A solution is to disable :prop_tgt:`AUTOGEN_ORIGIN_DEPENDS` on
    the respective origin targets.
