ANDROID_PROGUARD_CONFIG_PATH
----------------------------

.. versionadded:: 3.4

Set the Android property that specifies the location of the ProGuard
config file. Leave empty to use the default one.
This a string property that contains the path to ProGuard config file.
This property is initialized by the value of the
:variable:`CMAKE_ANDROID_PROGUARD_CONFIG_PATH` variable if it is set
when a target is created.
