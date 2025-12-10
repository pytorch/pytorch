GENERATOR_IS_MULTI_CONFIG
-------------------------

.. versionadded:: 3.9

Read-only property that is true on multi-configuration generators.

True when using a multi-configuration generator such as:

* :generator:`Ninja Multi-Config`
* :ref:`Visual Studio Generators`
* :generator:`Xcode`

Multi-config generators use :variable:`CMAKE_CONFIGURATION_TYPES`
as the set of configurations and ignore :variable:`CMAKE_BUILD_TYPE`.
