FASTBUILD_DISTRIBUTION
----------------------

.. versionadded:: 4.2

A target property that controls whether distribution is enabled for the given
target in the generated ``fbuild.bff``.

If set to ``OFF``, the :generator:`FASTBuild` generator disables distributed
compilation for this target. This can be helpful for targets that are fast to
build locally or are incompatible with distributed execution.

Example:

.. code-block:: cmake

  set_property(TARGET my_target PROPERTY FASTBUILD_DISTRIBUTION OFF)

Defaults to ``ON``.
