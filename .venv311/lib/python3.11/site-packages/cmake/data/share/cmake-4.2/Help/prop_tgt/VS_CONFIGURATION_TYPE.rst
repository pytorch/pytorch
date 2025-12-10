VS_CONFIGURATION_TYPE
---------------------

.. versionadded:: 3.6

Visual Studio project configuration type.

Sets the ``ConfigurationType`` attribute for a generated Visual Studio project.
The property value may use
:manual:`generator expressions <cmake-generator-expressions(7)>`.
If this property is set, it overrides the default setting that is based on the
target type (e.g. ``StaticLibrary``, ``Application``, ...).

Supported on :ref:`Visual Studio Generators` for VS 2010 and higher.
