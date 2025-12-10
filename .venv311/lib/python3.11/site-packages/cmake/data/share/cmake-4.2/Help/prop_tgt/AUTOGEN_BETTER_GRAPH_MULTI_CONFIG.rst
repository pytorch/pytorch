AUTOGEN_BETTER_GRAPH_MULTI_CONFIG
---------------------------------

.. versionadded:: 3.29

``AUTOGEN_BETTER_GRAPH_MULTI_CONFIG`` is a boolean property that can be set
on a target to have better dependency graph for multi-configuration generators.
When this property is enabled, ``CMake`` will generate more per-config targets.
Thus, the dependency graph will be more accurate for multi-configuration
generators and some recompilations will be avoided.

If the Qt version is 6.8 or newer, this property is enabled by default.
If the Qt version is older than 6.8, this property is disabled by default.
Consult the Qt documentation to check if the property can be enabled for older
Qt versions.

See the :manual:`cmake-qt(7)` manual for more information on using CMake
with Qt.

This property is initialized by the
:variable:`CMAKE_AUTOGEN_BETTER_GRAPH_MULTI_CONFIG` variable if it is set when
a target is created.
