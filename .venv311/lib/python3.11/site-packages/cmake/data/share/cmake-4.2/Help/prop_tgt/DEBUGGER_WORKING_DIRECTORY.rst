DEBUGGER_WORKING_DIRECTORY
--------------------------

.. versionadded:: 4.0

Sets the local debugger working directory for C++ targets.
The property value may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
This property is initialized by the value of the variable
:variable:`CMAKE_DEBUGGER_WORKING_DIRECTORY` if it is set when a target is
created.

If the :prop_tgt:`VS_DEBUGGER_WORKING_DIRECTORY` property is also set, it will
take precedence over ``DEBUGGER_WORKING_DIRECTORY`` when using one of the
:ref:`Visual Studio Generators`.

Similarly, if :prop_tgt:`XCODE_SCHEME_WORKING_DIRECTORY` is set, it will
override ``DEBUGGER_WORKING_DIRECTORY`` when using the :generator:`Xcode`
generator.
