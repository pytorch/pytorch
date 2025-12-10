CMAKE_CROSS_CONFIGS
-------------------

.. versionadded:: 3.17

Specifies a :ref:`semicolon-separated list <CMake Language Lists>` of
configurations available from all ``build-<Config>.ninja`` files in the
:generator:`Ninja Multi-Config` generator.  This variable activates
cross-config mode. Targets from each config specified in this variable can be
built from any ``build-<Config>.ninja`` file. Custom commands will use the
configuration native to ``build-<Config>.ninja``. If it is set to ``all``, all
configurations from :variable:`CMAKE_CONFIGURATION_TYPES` are cross-configs. If
it is not specified, or empty, each ``build-<Config>.ninja`` file will only
contain build rules for its own configuration.

The value of this variable must be a subset of
:variable:`CMAKE_CONFIGURATION_TYPES`.
