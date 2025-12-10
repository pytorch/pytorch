VS_SHADER_DISABLE_OPTIMIZATIONS
-------------------------------

.. versionadded:: 3.11

Disable compiler optimizations for an ``.hlsl`` source file.  This adds the
``-Od`` flag to the command line for the FxCompiler tool.  Specify the value
``true`` for this property to disable compiler optimizations.
