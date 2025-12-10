CMAKE_DEFAULT_CONFIGS
---------------------

.. versionadded:: 3.17

Specifies a :ref:`semicolon-separated list <CMake Language Lists>` of configurations
to build for a target in ``build.ninja`` if no ``:<Config>`` suffix is specified in
the :generator:`Ninja Multi-Config` generator. If it is set to ``all``, all
configurations from :variable:`CMAKE_CROSS_CONFIGS` are used. If it is not
specified, it defaults to :variable:`CMAKE_DEFAULT_BUILD_TYPE`.

For example, if you set :variable:`CMAKE_DEFAULT_BUILD_TYPE` to ``Release``,
but set ``CMAKE_DEFAULT_CONFIGS`` to ``Debug`` or ``all``, all
``<target>`` aliases in ``build.ninja`` will resolve to ``<target>:Debug`` or
``<target>:all``, but custom commands will still use the ``Release``
configuration.

The value of this variable must be a subset of :variable:`CMAKE_CROSS_CONFIGS`
or be the same as :variable:`CMAKE_DEFAULT_BUILD_TYPE`. It must not be
specified if :variable:`CMAKE_DEFAULT_BUILD_TYPE` or
:variable:`CMAKE_CROSS_CONFIGS` is not used.
