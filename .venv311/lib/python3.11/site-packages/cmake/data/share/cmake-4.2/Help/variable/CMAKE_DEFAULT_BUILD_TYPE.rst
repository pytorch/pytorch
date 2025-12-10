CMAKE_DEFAULT_BUILD_TYPE
------------------------

.. versionadded:: 3.17

Specifies the configuration to use by default in a ``build.ninja`` file in the
:generator:`Ninja Multi-Config` generator. If this variable is specified,
``build.ninja`` uses build rules from ``build-<Config>.ninja`` by default. All
custom commands are executed with this configuration. If the variable is not
specified, the first item from :variable:`CMAKE_CONFIGURATION_TYPES` is used
instead.

The value of this variable must be one of the items from
:variable:`CMAKE_CONFIGURATION_TYPES`.
