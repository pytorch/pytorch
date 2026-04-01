Slang "Ray Tracing" Example
===========================

The goal of this example is to demonstrate how to use hardware ray-tracing in Slang.

The `shaders.slang` file contains a compute shader that traces primary rays from camera and shade intersections with basic lighting + ray-traced shadows. The file also defines a vertex and a fragment shader entry point for displaying the ray-traced image produced by the compute shader.

The `main.cpp` file contains the C++ application code, showing how to use the Slang API to load and compile the shader code, and how to use a graphics API abstraction layer implemented in `tools/gfx` to initiate hardware ray-tracing.
Note that this abstraction layer is *not* required in order to work with Slang, and it is just there to help us write example and test applications more conveniently.