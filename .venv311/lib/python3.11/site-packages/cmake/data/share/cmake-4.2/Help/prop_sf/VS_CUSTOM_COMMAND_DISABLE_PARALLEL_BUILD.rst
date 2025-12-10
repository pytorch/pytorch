VS_CUSTOM_COMMAND_DISABLE_PARALLEL_BUILD
----------------------------------------

.. versionadded:: 4.0

A boolean property that disables parallel building for the source file in
Visual Studio if it is built via :command:`add_custom_command` and is the
``MAIN_DEPENDENCY`` input for the custom command.
See policy :policy:`CMP0147`.
