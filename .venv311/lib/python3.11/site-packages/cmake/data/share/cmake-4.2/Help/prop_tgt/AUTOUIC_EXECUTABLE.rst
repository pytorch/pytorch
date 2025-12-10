AUTOUIC_EXECUTABLE
------------------

.. versionadded:: 3.14

``AUTOUIC_EXECUTABLE`` is file path pointing to the ``uic``
executable to use for :prop_tgt:`AUTOUIC` enabled files. Setting
this property will make CMake skip the automatic detection of the
``uic`` binary as well as the sanity-tests normally run to ensure
that the binary is available and working as expected.

Usually this property does not need to be set. Only consider this
property if auto-detection of ``uic`` can not work -- e.g. because
you are building the ``uic`` binary as part of your project.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
