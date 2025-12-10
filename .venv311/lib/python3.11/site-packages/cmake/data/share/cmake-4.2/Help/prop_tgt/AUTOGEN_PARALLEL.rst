AUTOGEN_PARALLEL
----------------

.. versionadded:: 3.11

Number of parallel ``moc`` or ``uic`` processes to start when using
:prop_tgt:`AUTOMOC` and :prop_tgt:`AUTOUIC`.

The custom :ref:`<ORIGIN>_autogen <<ORIGIN>_autogen>` target starts a number
of threads.  Each thread parses a source file and on demand starts a ``moc``
or ``uic`` process. ``AUTOGEN_PARALLEL`` controls how many parallel threads
(and therefore ``moc`` or ``uic`` processes) are started.

- An empty (or unset) value or the string ``AUTO`` sets the number of
  threads/processes to the number of physical CPUs on the host system.
- A positive non zero integer value sets the exact thread/process count.
- Otherwise a single thread/process is started.

By default ``AUTOGEN_PARALLEL`` is initialized from
:variable:`CMAKE_AUTOGEN_PARALLEL`.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
