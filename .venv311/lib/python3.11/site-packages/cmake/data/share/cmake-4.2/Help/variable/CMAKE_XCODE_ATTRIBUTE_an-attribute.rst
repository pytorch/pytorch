CMAKE_XCODE_ATTRIBUTE_<an-attribute>
------------------------------------

.. versionadded:: 3.1

Set Xcode target attributes directly.

Tell the :generator:`Xcode` generator to set ``<an-attribute>`` to a given
value in the generated Xcode project.  Ignored on other generators.

This offers low-level control over the generated Xcode project file.
It is meant as a last resort for specifying settings that CMake does
not otherwise have a way to control.  Although this can override a
setting CMake normally produces on its own, doing so bypasses CMake's
model of the project and can break things.

See the :prop_tgt:`XCODE_ATTRIBUTE_<an-attribute>` target property
to set attributes on a specific target.

Contents of ``CMAKE_XCODE_ATTRIBUTE_<an-attribute>`` may use
"generator expressions" with the syntax ``$<...>``.  See the
:manual:`cmake-generator-expressions(7)` manual for available
expressions.  See the :manual:`cmake-buildsystem(7)` manual
for more on defining buildsystem properties.
