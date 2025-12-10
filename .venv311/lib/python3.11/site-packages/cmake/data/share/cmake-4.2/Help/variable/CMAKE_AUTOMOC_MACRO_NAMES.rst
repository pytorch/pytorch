CMAKE_AUTOMOC_MACRO_NAMES
----------------------------

.. versionadded:: 3.10

:ref:`Semicolon-separated list <CMake Language Lists>` list of macro names used by
:variable:`CMAKE_AUTOMOC` to determine if a C++ file needs to be
processed by ``moc``.

This variable is used to initialize the :prop_tgt:`AUTOMOC_MACRO_NAMES`
property on all the targets. See that target property for additional
information.

The default value is ``Q_OBJECT;Q_GADGET;Q_NAMESPACE;Q_NAMESPACE_EXPORT``.

Example
^^^^^^^
Let CMake know that source files that contain ``CUSTOM_MACRO`` must be ``moc``
processed as well:

.. code-block:: cmake

  set(CMAKE_AUTOMOC ON)
  list(APPEND CMAKE_AUTOMOC_MACRO_NAMES "CUSTOM_MACRO")
