AUTOMOC_MACRO_NAMES
-------------------

.. versionadded:: 3.10

A :ref:`semicolon-separated list <CMake Language Lists>` of macro names used by
:prop_tgt:`AUTOMOC` to determine if a C++ file needs to be processed by ``moc``.

This property is only used if the :prop_tgt:`AUTOMOC` property is ``ON``
for this target.

When running :prop_tgt:`AUTOMOC`, CMake searches for the strings listed in
``AUTOMOC_MACRO_NAMES`` in C++ source and header files.
If any of the strings is found

- as the first non space string on a new line or
- as the first non space string after a ``{`` on a new line,

then the file will be processed by ``moc``.

By default ``AUTOMOC_MACRO_NAMES`` is initialized from
:variable:`CMAKE_AUTOMOC_MACRO_NAMES`.

See also the :prop_tgt:`INTERFACE_AUTOMOC_MACRO_NAMES` target property.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

Example
^^^^^^^

In this case the ``Q_OBJECT`` macro is hidden inside another macro
called ``CUSTOM_MACRO``.  To let CMake know that source files that contain
``CUSTOM_MACRO`` need to be ``moc`` processed, we call:

.. code-block:: cmake

  set_property(TARGET tgt APPEND PROPERTY AUTOMOC_MACRO_NAMES "CUSTOM_MACRO")
