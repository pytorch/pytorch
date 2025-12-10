CMAKE_VS_PLATFORM_TOOLSET
-------------------------

Visual Studio Platform Toolset name.

VS 10 and above use MSBuild under the hood and support multiple
compiler toolchains.  CMake may specify a toolset explicitly, such as
``v110`` for VS 11 or ``Windows7.1SDK`` for 64-bit support in VS 10
Express.  CMake provides the name of the chosen toolset in this
variable.

See the :variable:`CMAKE_GENERATOR_TOOLSET` variable for details.
