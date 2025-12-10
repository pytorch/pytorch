CMAKE_VS_PLATFORM_NAME_DEFAULT
------------------------------

.. versionadded:: 3.14.3

Default for the Visual Studio target platform name for the current generator
without considering the value of the :variable:`CMAKE_GENERATOR_PLATFORM`
variable.  For :ref:`Visual Studio Generators` for VS 2017 and below this is
always ``Win32``.  For VS 2019 and above this is based on the host platform.

See also the :variable:`CMAKE_VS_PLATFORM_NAME` variable.
