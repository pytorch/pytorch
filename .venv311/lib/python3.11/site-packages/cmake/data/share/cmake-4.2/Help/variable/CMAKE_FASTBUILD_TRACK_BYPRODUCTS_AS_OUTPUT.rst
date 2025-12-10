CMAKE_FASTBUILD_TRACK_BYPRODUCTS_AS_OUTPUT
------------------------------------------

.. versionadded:: 4.2

By default, custom commands declaring only ``BYPRODUCTS`` will always
run unconditionally.
You can use this variable to make FASTBuild rerun the command only when its
inputs have changed or the byproduct file is missing.

.. note::

   When this variable is ``OFF`` (the default), ``BYPRODUCTS`` are treated
   similarly to how Ninja handles them — as opaque side effects — and the
   generator emits ``ExecAlways`` nodes to ensure they always run.

Defaults to ``OFF``.
