VS_SHADER_ENABLE_DEBUG
----------------------

.. versionadded:: 3.11

Enable debugging information for an ``.hlsl`` source file.  This adds the
``-Zi`` flag to the command line for the FxCompiler tool.  Specify the value
``true`` to generate debugging information for the compiled shader.
