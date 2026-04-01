Slang
=====
![CI Status](https://github.com/shader-slang/slang/actions/workflows/ci.yml/badge.svg?branch=master)
![CTS Status](https://github.com/shader-slang/slang/actions/workflows/vk-gl-cts-nightly.yml/badge.svg)

Slang is a shading language that makes it easier to build and maintain large shader codebases in a modular and extensible fashion, while also maintaining the highest possible performance on modern GPUs and graphics APIs.
Slang is based on years of collaboration between researchers at NVIDIA, Carnegie Mellon University, Stanford, MIT, UCSD and the University of Washington.


Why Slang?
---------------

The Slang shading language is designed to enable real-time graphics developers to work with large-scale, high-performance shader code.

### Write Shaders Once, Run Anywhere

The Slang compiler can generate code for a wide variety of targets: D3D12, Vulkan, Metal, D3D11, OpenGL, CUDA, and even generate code to run on a CPU. For textual targets, such as Metal Shading Language (MSL) and CUDA, Slang produces readable code that preserves original identifier names, as well as the type and call structure, making it easier to debug.

### Access the Latest GPU Features

Slang code is highly portable, but can still leverage unique platform capabilities, including the latest features in Direct3D and Vulkan. For example, developers can make full use of [pointers](https://shader-slang.com/slang/user-guide/convenience-features.html#pointers-limited) when generating SPIR-V.
Slang's [capability system](https://shader-slang.com/slang/user-guide/capabilities.html) helps applications manage feature set differences across target platforms by ensuring code only uses available features during the type-checking step, before generating final code. Additionally, Slang provides [flexible interop](https://shader-slang.com/slang/user-guide/a1-04-interop.html) features to enable directly embedding target code or SPIR-V into generated shaders.

### Leverage Neural Graphics with Automatic Differentiation

Slang can [automatically generate both forward and backward derivative propagation code](https://shader-slang.com/slang/user-guide/autodiff.html) for complex functions that involve arbitrary control flow and dynamic dispatch. This allows existing rendering codebases to easily become differentiable, or for Slang to serve as the kernel language in a PyTorch-driven machine learning framework via [`slangtorch`](https://shader-slang.com/slang/user-guide/a1-02-slangpy.html).

### Scalable Software Development with Modules

Slang provides a [module system](https://shader-slang.com/slang/user-guide/modules.html) that enables logical organization of code for separate compilation. Slang modules can be independently compiled offline to a custom IR (with optional obfuscation) and then linked at runtime to generate code in formats such as DXIL or SPIR-V.

### Code Specialization that Works with Modules

Slang supports [generics and interfaces](https://shader-slang.com/slang/user-guide/interfaces-generics.html) (a.k.a. type traits/protocols), allowing for clear expression of shader specialization without the need for preprocessor techniques or string-pasting. Unlike C++ templates, Slang's generics are pre-checked and don't produce cascading error messages that are difficult to diagnose. The same generic shader can be specialized for a variety of different types to produce specialized code ahead of time, or on the fly, entirely under application control.

### Easy On-ramp for HLSL and GLSL Codebases

Slang's syntax is similar to HLSL, and most existing HLSL code can be compiled with the Slang compiler out-of-the-box, or with just minor modifications. This allows existing shader codebases to immediately benefit from Slang without requiring a complete rewrite or port.

Slang provides a compatibility module that enables the use of most GLSL intrinsic functions and GLSL's parameter binding syntax.

### Comprehensive Tooling Support

Slang comes with full support of IntelliSense editing features in Visual Studio Code and Visual Studio through the Language Server Protocol.
Full debugging capabilities are also available through RenderDoc and SPIR-V based tools.

Getting Started
---------------

The fastest way to get started using Slang in your own development is to use a pre-built binary package, available through GitHub [releases](https://github.com/shader-slang/slang/releases).
Slang binaries are also included in the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) since version 1.3.296.0.

There are packages built for x86_64 and aarch64 Windows, Linux and macOS.
Each binary release includes the command-line `slangc` compiler, a shared library for the compiler, and the `slang.h` header.

See the user-guide for info on using the `slangc` command-line tool: [Slang Command Line Usage](
https://shader-slang.com/slang/user-guide/compiling.html#command-line-compilation-with-slangc).

If you want to try out the Slang language without installing anything, a fast and simple way is to use the [Slang Playground](https://shader-slang.com/slang-playground). The playground allows you to compile Slang code to a variety of targets, and even run some simple shaders directly within the browser. The playground loads Slang compiler to your browser and runs all compilation locally. No data will be sent to any servers.

If you would like to build Slang from source, please consult the [build instructions](docs/building.md).

Documentation
-------------

The Slang project provides a variety of different [documentation](docs/), but most users would be well served starting with the [User's Guide](https://shader-slang.github.io/slang/user-guide/).

For developers writing Slang code, the [Slang Core Module Reference](https://shader-slang.com/stdlib-reference/) provides detailed documentation on Slang's built-in types and functions.

We also provide a few [examples](examples/) of how to integrate Slang into a rendering application.

These examples use a graphics layer that we include with Slang called "GFX" which is an abstraction library of various graphics APIs (D3D11, D2D12, OpenGL, Vulkan, CUDA, and the CPU) to support cross-platform applications using GPU graphics and compute capabilities. 
If you'd like to learn more about GFX, see the [GFX User Guide](https://shader-slang.com/slang/gfx-user-guide/index.html).

Additionally, we recommend checking out [Vulkan Mini Examples](https://github.com/nvpro-samples/vk_mini_samples/) for more examples of using Slang's language features available on Vulkan, such as pointers and the ray tracing intrinsics.

Contributing
------------

If you'd like to contribute to the project, we are excited to have your input.
The following guidelines should be observed by contributors:

* Please follow the contributor [Code of Conduct](CODE_OF_CONDUCT.md).
* Bugs reports and feature requests should go through the GitHub issue tracker
* Changes should ideally come in as small pull requests on top of `master`, coming from your own personal fork of the project
* Large features that will involve multiple contributors or a long development time should be discussed in issues, and broken down into smaller pieces that can be implemented and checked in in stages

[Contribution guide](CONTRIBUTING.md) describes the workflow for contributors at more detail.

Limitations and Support
-----------------------

### Platform support

The Slang compiler and libraries can be built on the following platforms:

|  Windows  |   Linux   |   MacOS   |  WebAssembly |
|:---------:|:---------:|:---------:|:------------:|
| supported | supported | supported | experimental |

Both `x86_64` and `aarch64` architectures are supported on Windows, Linux and MacOS platforms.

### Target support

Slang can compile shader code to the following targets:

|    Target   |                                         Status                                        |                          Output Formats                          |
|:-----------:|:-------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| Direct3D 11 |    [supported](https://shader-slang.com/slang/user-guide/targets.html#direct3d-11)    |                               HLSL                               |
| Direct3D 12 |    [supported](https://shader-slang.com/slang/user-guide/targets.html#direct3d-12)    |                               HLSL                               |
|    Vulkan   |       [supported](https://shader-slang.com/slang/user-guide/targets.html#vulkan)      |                            SPIRV, GLSL                           |
|    Metal    |     [experimental*](https://shader-slang.com/slang/user-guide/targets.html#metal)     |                      Metal Shading Language                      |
|    WebGPU   |                                     experimental**                                    |                               WGSL                               |
|     CUDA    |   [supported](https://shader-slang.com/slang/user-guide/targets.html#cuda-and-optix)  |                        C++ (compute only)                        |
|    Optix    | [experimental](https://shader-slang.com/slang/user-guide/targets.html#cuda-and-optix) |                             C++ (WIP)                            |
|     CPU     |   [experimental](https://shader-slang.com/slang/user-guide/targets.html#cpu-compute)  | C++ (kernel), C++ (host), standalone executable, dynamic library |

> *Slang currently supports generating vertex, fragment, compute, task and mesh
> shaders for Metal.

> **WGSL support is still work in-progress.

For greater detail, see the [Supported Compilation
Targets](https://shader-slang.com/slang/user-guide/targets.html) section of the
[User Guide](https://shader-slang.github.io/slang/user-guide/)

The Slang project has been used for production applications and large shader
codebases, but it is still under active development. Support is currently
focused on the platforms (Windows, Linux) and target APIs (Direct3D 12, Vulkan)
where Slang is used most heavily. Users who are looking for support on other
platforms or APIs should coordinate with the development team via the issue
tracker to make sure that their use cases can be supported.

License
-------

The Slang code itself is under the Apache 2.0 with LLVM Exception license (see [LICENSE](LICENSE)).

Builds of the core Slang tools depend on the following projects, either automatically or optionally, which may have their own licenses:

* [`glslang`](https://github.com/KhronosGroup/glslang) (BSD)
* [`lz4`](https://github.com/lz4/lz4) (BSD)
* [`miniz`](https://github.com/richgel999/miniz) (MIT)
* [`spirv-headers`](https://github.com/KhronosGroup/SPIRV-Headers) (Modified MIT)
* [`spirv-tools`](https://github.com/KhronosGroup/SPIRV-Tools) (Apache 2.0)
* [`ankerl::unordered_dense::{map, set}`](https://github.com/martinus/unordered_dense) (MIT)

Slang releases may include [LLVM](https://github.com/llvm/llvm-project) under the license:

* [`llvm`](https://llvm.org/docs/DeveloperPolicy.html#new-llvm-project-license-framework) (Apache 2.0 License with LLVM exceptions)

Some of the tests and example programs that build with Slang use the following projects, which may have their own licenses:

* [`glm`](https://github.com/g-truc/glm) (MIT)
* `stb_image` and `stb_image_write` from the [`stb`](https://github.com/nothings/stb) collection of single-file libraries (Public Domain)
* [`tinyobjloader`](https://github.com/tinyobjloader/tinyobjloader) (MIT)
