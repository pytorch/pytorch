ANDROID_JAR_DIRECTORIES
-----------------------

.. versionadded:: 3.4

Set the Android property that specifies directories to search for
the JAR libraries.

This a string property that contains the directory paths separated by
semicolons. This property is initialized by the value of the
:variable:`CMAKE_ANDROID_JAR_DIRECTORIES` variable if it is set when
a target is created.

Contents of ``ANDROID_JAR_DIRECTORIES`` may use "generator expressions"
with the syntax ``$<...>``.  See the :manual:`cmake-generator-expressions(7)`
manual for available expressions.
