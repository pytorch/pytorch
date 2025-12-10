CMAKE_GLOBAL_AUTORCC_TARGET_NAME
--------------------------------

.. versionadded:: 3.14

Change the name of the global ``autorcc`` target.

When :variable:`CMAKE_GLOBAL_AUTORCC_TARGET` is enabled, a global custom target
named ``autorcc`` is created.  ``CMAKE_GLOBAL_AUTORCC_TARGET_NAME``
allows to set a different name for that target.

By default ``CMAKE_GLOBAL_AUTORCC_TARGET_NAME`` is unset.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
