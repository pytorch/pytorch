CMAKE_EXPORT_PACKAGE_REGISTRY
-----------------------------

.. versionadded:: 3.15

Enables the :command:`export(PACKAGE)` command when :policy:`CMP0090`
is set to ``NEW``.

The :command:`export(PACKAGE)` command does nothing by default.  In some cases
it is desirable to write to the user package registry, so the
``CMAKE_EXPORT_PACKAGE_REGISTRY`` variable may be set to enable it.

If :policy:`CMP0090` is *not* set to ``NEW`` this variable does nothing, and
the :variable:`CMAKE_EXPORT_NO_PACKAGE_REGISTRY` variable controls the behavior
instead.

See also :ref:`Disabling the Package Registry`.
