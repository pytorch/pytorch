CMAKE_EXPORT_NO_PACKAGE_REGISTRY
--------------------------------

.. versionadded:: 3.1

Disable the :command:`export(PACKAGE)` command when :policy:`CMP0090`
is not set to ``NEW``.

In some cases, for example for packaging and for system wide
installations, it is not desirable to write the user package registry.
If the ``CMAKE_EXPORT_NO_PACKAGE_REGISTRY`` variable is enabled,
the :command:`export(PACKAGE)` command will do nothing.

If :policy:`CMP0090` is set to ``NEW`` this variable does nothing, and the
:variable:`CMAKE_EXPORT_PACKAGE_REGISTRY` variable controls the behavior
instead.

See also :ref:`Disabling the Package Registry`.
