Visual Studio 12 2013
---------------------

Removed.  This once generated Visual Studio 12 2013 project files, but
the generator has been removed since CMake 3.31.  It is still possible
to build with the VS 12 2013 toolset by also installing VS 2017 (or above)
and using the :generator:`Visual Studio 15 2017` (or above) generator with
:variable:`CMAKE_GENERATOR_TOOLSET` set to ``v120``,
or by using the :generator:`NMake Makefiles` generator.
