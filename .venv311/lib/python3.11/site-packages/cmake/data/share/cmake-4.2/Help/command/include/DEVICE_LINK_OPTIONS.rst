Host And Device Specific Link Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.18
  When a device link step is involved, which is controlled by
  :prop_tgt:`CUDA_SEPARABLE_COMPILATION` and
  :prop_tgt:`CUDA_RESOLVE_DEVICE_SYMBOLS` properties and policy :policy:`CMP0105`,
  the raw options will be delivered to the host and device link steps (wrapped in
  ``-Xcompiler`` or equivalent for device link). Options wrapped with
  :genex:`$<DEVICE_LINK:...>` generator expression will be used
  only for the device link step. Options wrapped with :genex:`$<HOST_LINK:...>`
  generator expression will be used only for the host link step.
