Visual Studio 10 2010
---------------------

Removed.  This once generated Visual Studio 10 2010 project files, but
the generator has been removed since CMake 3.25.  It is still possible
to build with the VS 10 2010 toolset by also installing VS 2017 (or above)
and using the :generator:`Visual Studio 15 2017` (or above) generator with
:variable:`CMAKE_GENERATOR_TOOLSET` set to ``v100``,
or by using the :generator:`NMake Makefiles` generator.
