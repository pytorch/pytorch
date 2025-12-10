Visual Studio 11 2012
---------------------

Removed.  This once generated Visual Studio 11 2012 project files, but
the generator has been removed since CMake 3.28.  It is still possible
to build with the VS 11 2012 toolset by also installing VS 2017 (or above)
and using the :generator:`Visual Studio 15 2017` (or above) generator with
:variable:`CMAKE_GENERATOR_TOOLSET` set to ``v110``,
or by using the :generator:`NMake Makefiles` generator.
