CMAKE_GLOBAL_AUTORCC_TARGET
---------------------------

.. versionadded:: 3.14

Switch to enable generation of a global ``autorcc`` target.

When ``CMAKE_GLOBAL_AUTORCC_TARGET`` is enabled, a custom target
``autorcc`` is generated. This target depends on all :prop_tgt:`AUTORCC`
generated ``<ORIGIN>_arcc_<QRC>`` targets in the project.
By building the global ``autorcc`` target, all :prop_tgt:`AUTORCC`
files in the project will be generated.

The name of the global ``autorcc`` target can be changed by setting
:variable:`CMAKE_GLOBAL_AUTORCC_TARGET_NAME`.

By default ``CMAKE_GLOBAL_AUTORCC_TARGET`` is unset.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
