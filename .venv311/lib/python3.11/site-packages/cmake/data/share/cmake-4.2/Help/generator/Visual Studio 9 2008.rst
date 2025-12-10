Visual Studio 9 2008
--------------------

Removed.  This once generated Visual Studio 9 2008 project files, but
the generator has been removed since CMake 3.30.  It is still possible
to build with the VS 9 2008 toolset by also installing VS 10 2010 and
VS 2017 (or above) and using the :generator:`Visual Studio 15 2017`
generator (or above) with :variable:`CMAKE_GENERATOR_TOOLSET` set to ``v90``,
or by using the :generator:`NMake Makefiles` generator.
