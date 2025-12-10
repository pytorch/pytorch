AUTOGEN_BUILD_DIR
-----------------

.. versionadded:: 3.9

Directory where :prop_tgt:`AUTOMOC`, :prop_tgt:`AUTOUIC` and :prop_tgt:`AUTORCC`
generate files for the target.

The directory is created on demand and automatically added to the
:prop_tgt:`ADDITIONAL_CLEAN_FILES` target property.

When unset or empty the directory ``<dir>/<target-name>_autogen`` is used where
``<dir>`` is :variable:`CMAKE_CURRENT_BINARY_DIR` and ``<target-name>``
is :prop_tgt:`NAME`.

By default ``AUTOGEN_BUILD_DIR`` is unset.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.
