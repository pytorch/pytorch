Slang Examples
==============

This directory contains small example programs showing how to use the Slang language, compiler, and API.

* The [`hello-world`](hello-world/) example shows a minimal example of using Slang shader code more or less like HLSL.

* The [`shader-object`](shader-object/) example shows how Slang's support for interface types can be used to implement shader specialization with simpler logic than preprocessor-based techniques.

* The [`gpu-printing`](gpu-printing/) example shows how Slang's support for string literals can be used to implement a cross-API "GPU `printf`" solution

Most of the examples presented here use a software layer called `gfx` (exposed via `slang-gfx.h`) to abstract over the differences between various target APIs/platforms (D3D11, D3D12, OpenGL, Vulkan, CUDA, and CPU).
Using `gfx` is not a requirement for using Slang, but it provides a concrete example of how tight integration of Slang's features into a GPU abstraction layer can provide for a clean and usable application programming model.