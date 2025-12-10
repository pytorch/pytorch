CMAKE_VS_GLOBALS
----------------

.. versionadded:: 3.13

List of ``Key=Value`` records to be set per target as target properties
:prop_tgt:`VS_GLOBAL_<variable>` with ``variable=Key`` and value ``Value``.

For example:

.. code-block:: cmake

  set(CMAKE_VS_GLOBALS
    "DefaultLanguage=en-US"
    "MinimumVisualStudioVersion=14.0"
    )

will set properties ``VS_GLOBAL_DefaultLanguage`` to ``en-US`` and
``VS_GLOBAL_MinimumVisualStudioVersion`` to ``14.0`` for all targets
(except for ``INTERFACE`` libraries).

This variable is meant to be set by a
:variable:`toolchain file <CMAKE_TOOLCHAIN_FILE>`.
