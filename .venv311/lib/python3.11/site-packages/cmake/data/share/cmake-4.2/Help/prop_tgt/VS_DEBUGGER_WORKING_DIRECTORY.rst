VS_DEBUGGER_WORKING_DIRECTORY
-----------------------------

.. versionadded:: 3.8

Sets the local debugger working directory for Visual Studio C++ targets.
The property value may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
This is defined in ``<LocalDebuggerWorkingDirectory>`` in the Visual Studio
project file.  This property is initialized by the value of the variable
:variable:`CMAKE_VS_DEBUGGER_WORKING_DIRECTORY` if it is set when a target is
created.

This property only works for :ref:`Visual Studio Generators`;
it is ignored on other generators.

See also :prop_tgt:`DEBUGGER_WORKING_DIRECTORY`.
