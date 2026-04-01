Slang "Hello World" Example
===========================

The goal of this example is to demonstrate an almost minimal application that uses Slang for shading.

The `shaders.slang` file contains simple vertex and fragment shader entry points. The shader code should compile as either Slang or HLSL code (that is, this example does not show off any new Slang language features).

The `main.cpp` file contains the C++ application code, showing how to use the Slang API to load and compile the shader code to DirectX shader bytecode (DXBC).
The application perform rendering using the D3D11 API, through a platform and graphics API abstraction layer that is implemented in `tools/gfx`.
Note that this abstraction layer is *not* required in order to work with Slang, and it is just there to help us write example and test applications more conveniently.

This example is not necessarily representative of best practices for integrating Slang into a production engine; the goal is merely to use the minimum amount of code possible to demonstrate a complete applicaiton that uses Slang.
