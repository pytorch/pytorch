ANDROID_NATIVE_LIB_DEPENDENCIES
-------------------------------

.. versionadded:: 3.4

Set the Android property that specifies the .so dependencies.
This is a string property.

This property is initialized by the value of the
:variable:`CMAKE_ANDROID_NATIVE_LIB_DEPENDENCIES` variable if it is set
when a target is created.

Contents of ``ANDROID_NATIVE_LIB_DEPENDENCIES`` may use
"generator expressions" with the syntax ``$<...>``. See the
:manual:`cmake-generator-expressions(7)` manual for
available expressions.
