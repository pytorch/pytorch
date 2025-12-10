AUTOMOC_PATH_PREFIX
-------------------

.. versionadded:: 3.16

When this property is ``ON``, CMake will generate the ``-p`` path prefix
option for ``moc`` on :prop_tgt:`AUTOMOC` enabled Qt targets.

To generate the path prefix, CMake tests if the header compiled by ``moc``
is in any of the target
:command:`include directories <target_include_directories>`.  If so, CMake will
compute the relative path accordingly.  If the header is not in the
:command:`include directories <target_include_directories>`, CMake will omit
the ``-p`` path prefix option.  ``moc`` usually generates a
relative include path in that case.

``AUTOMOC_PATH_PREFIX`` is initialized from the variable
:variable:`CMAKE_AUTOMOC_PATH_PREFIX`, which is ``OFF`` by default.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

Reproducible builds
^^^^^^^^^^^^^^^^^^^

For reproducible builds it is recommended to keep headers that are ``moc``
compiled in one of the target
:command:`include directories <target_include_directories>` and set
``AUTOMOC_PATH_PREFIX`` to ``ON``.  This ensures that:

- ``moc`` output files are identical on different build setups,
- ``moc`` output files will compile correctly when the source and/or
  build directory is a symbolic link.
