FASTBUILD_CACHING
-----------------

.. versionadded:: 4.2

A target property that controls whether caching is enabled for the given
target in the generated ``fbuild.bff``.

If set to ``OFF``, the :generator:`FASTBuild` generator disables caching
features for this target. This is useful for targets that are known to be
unreliably cached or not worth caching.

Example:

.. code-block:: cmake

  set_property(TARGET my_target PROPERTY FASTBUILD_CACHING OFF)

Defaults to ``ON``.
