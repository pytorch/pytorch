Slang "CPU Hello World" Example
===============================

The goal of this example is to demonstrate an almost minimal application that uses Slang to produce and use a kernel that is run on CPU.

The `shader.slang` file contains a compute shader entry point. The shader code should compile as either Slang or HLSL code (that is, this example does not show off any new Slang language features).

The `main.cpp` file contains the C++ application code, showing how to use the Slang API to load and compile the shader code to produce and execute CPU code.

This example is not necessarily representative of best practices for integrating Slang into a production engine; the goal is merely to use the minimum amount of code possible to demonstrate a complete application that uses Slang.
