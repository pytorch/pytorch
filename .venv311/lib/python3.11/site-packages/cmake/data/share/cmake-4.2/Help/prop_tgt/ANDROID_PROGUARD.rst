ANDROID_PROGUARD
----------------

.. versionadded:: 3.4

When this property is set to true that enables the ProGuard tool to shrink,
optimize, and obfuscate the code by removing unused code and renaming
classes, fields, and methods with semantically obscure names.
This property is initialized by the value of the
:variable:`CMAKE_ANDROID_PROGUARD` variable if it is set
when a target is created.
