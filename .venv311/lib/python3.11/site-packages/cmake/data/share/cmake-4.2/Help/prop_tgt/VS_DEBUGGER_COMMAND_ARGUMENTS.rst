VS_DEBUGGER_COMMAND_ARGUMENTS
-----------------------------

.. versionadded:: 3.13

Sets the local debugger command line arguments for Visual Studio C++ targets.
The property value may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
This is defined in ``<LocalDebuggerCommandArguments>`` in the Visual Studio
project file.  This property is initialized by the value of the variable
:variable:`CMAKE_VS_DEBUGGER_COMMAND_ARGUMENTS` if it is set when a target is
created.

This property only works for :ref:`Visual Studio Generators`;
it is ignored on other generators.
