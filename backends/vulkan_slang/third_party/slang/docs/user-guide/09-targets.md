---
layout: user-guide
permalink: /user-guide/targets
---

# Supported Compilation Targets

This chapter provides a brief overview of the compilation targets supported by Slang, and their different capabilities.

## Background and Terminology

### Code Formats

When Slang compiles for a target platform one of the most important distinctions is the _format_ of code for that platform.
For a native CPU target, the format is typically the executable machine-code format for the processor family (for example, x86-64).
In contrast, GPUs are typically programmed through APIs that abstract over multiple GPU processor families and versions.
GPU APIs usually define an _intermediate language_ that sits between a high-level-language compiler like Slang and GPU-specific compilers that live in drivers for the API.

### Pipelines and Stages

GPU code execution occurs in the context of a _pipeline_.
A pipeline comprises one or more _stages_ and dataflow connections between them.
Some stages are _programmable_ and run a user-defined _kernel_ that has been compiled from a language like Slang, while others are _fixed-function_ and can only be configured, rather than programmed, by the user.
Slang supports three different pipelines.

#### Rasterization

The _rasterization_ pipeline is the original GPU rendering pipeline.
On current GPUs, the simplest rasterization pipelines have two programmable stages: a `vertex` stage and a `fragment` stage.
The rasterization pipeline is named after its most important fixed-function stage: the rasterizer, which determines the pixels covered by a geometric primitive, and emits _fragments_ covering those pixels, to be shaded.

#### Compute

The _compute_ pipeline is a simple pipeline with only one stage: a programmable `compute` stage.
As a result of being a single-stage pipeline the compute pipeline doesn't need to deal with many issues around inter-stage dataflow that other pipelines do.

#### Ray Tracing

A _ray tracing_ pipeline has multiple stages pertaining to the life cycle of a ray being traced through a scene of geometric primitives.
These can include an `intersection` stage to compute whether a ray intersects a geometry primitive, a `miss` stage that runs when a ray does not intersect any geometric object in a scene, etc.

Note that some platforms support types and operations related to ray tracing that can run outside of the context of a dedicated ray tracing pipeline.
Just as applications can do computation outside of the dedicated compute pipeline, the use of ray tracing does not necessarily mean that a ray tracing pipeline is being used.

### Shader Parameter Bindings

The kernels that execute within a pipeline typically has access to four different kinds of data:

- _Varying inputs_ coming from the system or from a preceding pipeline stage

- _Varying outputs_ which will be passed along to the system or to a following pipeline stage

- _Temporaries_ which are scratch memory or registers used by each invocation of the kernel and then dismissed on exit.

- _Shader parameters_ (sometimes also called _uniform parameters_), which provide access to data from outside the pipeline dataflow

The first three of these kinds of data are largely handled by the implementation of a pipeline.
In contrast, an application programmer typically needs to manually prepare shader parameters, using the appropriate mechanisms and rules for each target platform.

On platforms that provide a CPU-like "flat" memory model with a single virtual address space, and where any kind of data can be stored at any address, passing shader parameters can be almost trivial.
Current graphics APIs provide far more complicated and less uniform mechanisms for passing shader parameters.

A high-level language compiler like Slang handles the task of _binding_ each user-defined shader parameter to one or more of the parameter-passing resources defined by a target platform.
For example, the Slang compiler might bind a global `Texture2D` parameter called `gDiffuse` to the `t1` register defined by the Direct3D 11 API.

An application is responsible for passing the argument data for a parameter using the using the corresponding platform-specific resource it was bound to.
For example, an application should set the texture they want to use for `gDiffuse` to the `t1` register using Direct3D 11 API calls.

#### Slots

Historically, most graphics APIs have used a model where shader parameters are passed using a number of API-defined _slots_.
Each slot can store a single argument value of an allowed type.
Depending on the platform slots might be called "registers," "locations," "bindings," "texture units," or other similar names.

Slots almost exclusively use opaque types: textures, buffers, etc.
On platforms that use slots for passing shader parameters, value of ordinary types like `float` or `int` need to be stored into a buffer, and then that buffer is passed via an appropriate slot.

Although many graphics APIs use slots as an abstraction, the details vary greatly across APIs.
Different APIs define different kinds of slots, and the types of arguments that may be stored in those slots vary.
For example, one API might use two different kinds of slots for textures and buffers, while another uses a single kind of slot for both.
On some APIs each pipeline stage gets is own dedicated slots, while on others slots are shared across all stages in a pipeline.

#### Blocks

Newer graphics APIs typically provide a system for grouping related shader parameters into re-usable _blocks_.
Blocks might be referred to as "descriptor tables," "descriptor sets," or "argument buffers."
Each block comprises one or more slots (often called "descriptors") that can be used to bind textures, buffers, etc.

Blocks are in turn set into appropriate slots provided by a pipeline.
Because a block can contain many different slots for textures or buffers, switching a pipeline argument from one block to another can effectively swap out a large number of shader parameters in one operation.
Thus, while blocks introduce a level of indirection to parameter setting, then can also enable greater efficiency when parameters are grouped into blocks according to frequency of change.

#### Root Constants

Most recent graphics APIs also allow for a small amount of ordinary data (meaning types like `float` and `int` but not opaque types like buffers or textures) to be passed to the pipeline as _root constants_ (also called "push constants").

Using root constants can eliminate some overheads from passing parameters of ordinary types via buffers.
Passing a single `float` using a root constant rather than a buffer obviously eliminates a level of indirection.
More importantly, though, using a root constant can avoid application code having to allocate and manage the lifetime of a buffer in a concurrent CPU/GPU program.

## Direct3D 11

Direct3D 11 (D3D11) is an older graphics API, but remains popular because it is much simpler to learn and use than some more recent APIs.
In this section we will give an overview of the relevant features of D3D11 when used as a target platform for Slang.
Subsequent sections about other APIs may describe them by comparison to D3D11.

D3D11 kernels must be compiled to the DirectX Bytecode (DXBC) intermediate language.
A DXBC binary includes a hash/checksum computed using an undocumented algorithm, and the runtime API rejects kernels without a valid checksum.
The only supported way to generate DXBC is by compiling HLSL using the fxc compiler.

### Pipelines

D3D11 exposes two pipelines: rasterization and compute.

The D3D11 rasterization pipeline can include up to five programmable stages, although most of them are optional:

- The `vertex` stage (VS) transforms vertex data loaded from memory

- The optional `hull` stage (HS) typically sets up and computes desired tessellation levels for a higher-order primitive

- The optional `domain` stage (DS) evaluates a higher-order surface at domain locations chosen by a fixed-function tessellator

- The optional `geometry` stage (GS) receives as input a primitive and can produce zero or more new primitives as output

- The optional `fragment` stage transforms fragments produced by the fixed-function rasterizer, determining the values for those fragments that will be merged with values in zero or more render targets. The fragment stage is sometimes called a "pixel" stage (PS), even when it does not process pixels.

### Parameter Passing

Shader parameters are passed to each D3D11 stage via slots.
Each stage has its own slots of the following types:

- **Constant buffers** are used for passing relatively small (4KB or less) amounts of data that will be read by GPU code. Constant buffers are passed via `b` registers.

- **Shader resource views** (SRVs) include most textures, buffers, and other opaque resource types there are read or sampled by GPU code. SRVs use `t` registers.

- **Unordered access views** (UAVs) include textures, buffers, and other opaque resource types used for write or read-write operations in GPU code. UAVs use `u` registers.

- **Samplers** are used to pass opaque texture-sampling state, and use `s` registers.

In addition, the D3D11 pipeline provides _vertex buffer_ slots and a single _index buffer_ slot to be used as the source vertex and index data that defines primitives.
User-defined varying vertex shader inputs are bound to _vertex attribute_ slots (referred to as "input elements" in D3D11) which define how data from vertex buffers should be fetched to provide values for vertex attributes.

The D3D11 rasterization pipeline also provides a mechanism for specifying _render target views_ (RTVs) and _depth-stencil views_ (DSVs) that provide the backing storage for the pixels in a framebuffer.
User-defined fragment shader varying outputs (with `SV_Target` binding semantics) are bound to RTV slots.

One notable detail of the D3D11 API is that the slots for fragment-stage UAVs and RTVs overlap.
For example, a fragment kernel cannot use both `u0` and `SV_Target0` at once.

## Direct3D 12

Direct3D 12 (D3D12) is the current major version of the Direct3D API.

D3D12 kernels must be compiled to the DirectX Intermediate Language (DXIL).
DXIL is a layered encoding based off of LLVM bitcode; it introduces additional formatting rules and constraints which are loosely documented.
A DXIL binary may be signed, and the runtime API only accepts appropriately signed binaries (unless a developer mode is enabled on the host machine).
A DXIL validator `dxil.dll` is included in SDK releases, and this validator can sign binaries that pass validation.
While DXIL can in principle be generated from multiple compiler front-ends, support for other compilers is not prioritized.

### Pipelines

D3D12 includes rasterization and compute pipelines similar to those in D3D11.
Revisions to D3D12 have added additional stages to the rasterization pipeline, as well as a ray-tracing pipeline.

#### Mesh Shaders

> #### Note
>
> The Slang system does not currently support mesh shaders.

The D3D12 rasterization pipeline provides alternative geometry processing stages that may be used as an alternative to the `vertex`, `hull`, `domain`, and `geometry` stages:

- The `mesh` stage runs groups of threads which are responsible cooperating to produce both the vertex and index data for a _meshlet_ a bounded-size chunk of geometry.

- The optional `amplification` stage precedes the mesh stage and is responsible for determining how many mesh shader invocations should be run.

Compared to the D3D11 pipeline without tessellation (hull and domain shaders), a mesh shader is kind of like a combined/generalized vertex and geometry shader.

Compared to the D3D11 pipeline with tessellation, an amplification shader is kind of like a combined/generalized vertex and hull shader, while a mesh shader is kind of like a combined/generalized domain and geometry shader.

#### Ray Tracing

The DirectX Ray Tracing (DXR) feature added a ray tracing pipeline to D3D12.
The D3D12 ray tracing pipeline exposes the following programmable stages:

- The ray generation (`raygeneration`) stage is similar to a compute stage, but can trace zero or more rays and make use of the results of those traces.

- The `intersection` stage runs kernels to compute whether a ray intersects a user-defined primitive type. The system also includes a default intersector that handles triangle meshes.

- The so-called any-hit (`anyhit`) stage runs on _candidate_ hits where a ray has intersected some geometry, but the hit must be either accepted or rejected by application logic. Note that the any-hit stage does not necessarily run on _all_ hits, because configuration options on both scene geometry and rays can lead to these checks being bypassed.

- The closest-hit (`closesthit`) stage runs a single _accepted_ hit for a ray; under typical circumstances this will be the closest hit to the origin of the ray. A typical closest-hit shader might compute the apparent color of a surface, similar to a typical fragment shader.

- The `miss` stage runs for rays that do not find or accept any hits in a scene. A typical miss shader might return a background color or sample an environment map.

- The `callable` stage allows user-defined kernels to be invoked like subroutines in the context of the ray tracing pipeline.

Compared to existing rasterization and compute pipelines, an important difference in the design of the D3D12 ray tracing pipeline is that multiple kernels can be loaded into the pipeline for each of the programming stages.
The specific closest-hit, miss, or other kernel that runs for a given hit or ray is determined by indexing into an appropriate _shader table_, which is effectively an array of kernels.
The indexing into a shader table can depend on many factors including the type of ray, the type of geometry hit, etc.

Note that DXR version 1.1 adds ray tracing types and operations that can be used outside of the dedicated ray tracing pipeline.
These new mechanisms have less visible impact for a programmer using or integrating Slang.

### Parameter Passing

The mechanisms for parameter passing in D3D12 differ greatly from D3D11.
Most opaque types (texture, resources, samplers) must be set into blocks (D3D12 calls blocks "descriptor tables").
Each pipeline supports a fixed amount of storage for "root parameters," and allows those root parameters to be configured as root constants, slots for blocks, or slots for a limited number of opaque types (primarily just flat buffers).

Shader parameters are still grouped and bound to registers as in D3D11; for example, a `Texture2D` parameter is considered as an SRV and uses a `t` register.
D3D12 additionally associates binds shader parameters to "spaces" which are expressed similarly to registers (e.g., `space2`), but represent an orthogonal "axis" of binding.

While shader parameters are bound registers and spaces, those registers and spaces do not directly correspond to slots provided by the D3D12 API the way registers do in D3D11.
Instead, the configuration of the root parameters and the correspondence of registers/spaces to root parameters, blocks, and/or slots are defined by a _pipeline layout_ that D3D12 calls a "root signature."

Unlike D3D11, all of the stages in a D3D12 pipeline share the same root parameters.
A D3D12 pipeline layout can specify that certain root parameters or certain slots within blocks will only be accessed by a subset of stages, and can map the _same_ register/space pair to different parameters/blocks/slots as long as this is done for disjoint subset of stages.

#### Ray Tracing Specifics

The D3D12 ray tracing pipeline adds a new mechanism for passing shader parameters.
In addition to allowing shader parameters to be passed to the entire pipeline via root parameters, each shader table entry provides storage space for passing argument data specific to that entry.

Similar to the use of a pipeline layout (root signature) to configure the use of root parameters, each kernel used within shader entries must be configured with a "local root signature" that defines how the storage space in the shader table entry is to be used.
Shader parameters are still bound to registers and spaces as for non-ray-tracing code, and the local root signature simply allows those same registers/spaces to be associated with locations in a shader table entry.

One important detail is that some shader table entries are associated with a kernel for a single stage (e.g., a single miss shader), while other shader table entries are associated with a "hit group" consisting of up to one each of an intersection, any-hit, and closest-hit kernel.
Because multiple kernels in a hit group share the same shader table entry, they also share the configured slots in that entry for binding root constants, blocks, etc.

## Vulkan

Vulkan is a cross-platform GPU API for graphics and compute with a detailed specification produced by a multi-vendor standards body.
In contrast with OpenGL, Vulkan focuses on providing explicit control over as many aspects of GPU work as possible.
In contrast with OpenCL, Vulkan focuses first and foremost on the needs of real-time graphics developers.

Vulkan requires kernels to be compiled to the SPIR-V intermediate language.
SPIR-V is a simple and extensible binary program format with a detailed specification; it is largely unrelated to earlier "SPIR" formats that were LLVM-based and loosely specified.
The SPIR-V format does not require signing or hashing, and is explicitly designed to allow many different tools to produce and manipulate the format.
Drivers that consume SPIR-V are expected to perform validation at load time.
Some choices in the SPIR-V encoding are heavily influenced by specific design choices in the GLSL language, and may require non-GLSL compilers to transform code to match GLSL idioms.

### Pipelines

Vulkan includes rasterization, compute, and ray tracing pipelines with the same set of stages as described for D3D12 above.

### Parameter Passing

Like D3D12, Vulkan uses blocks (called "descriptor sets") to organize groups of bindings for opaque types (textures, buffers, samplers).
Similar to D3D12, a Vulkan pipeline supports a limited number of slots for passing blocks to the pipeline, and these slots are shared across all stages.
Vulkan also supports a limited number of bytes reserved for passing root constants (called "push constants").
Vulkan uses pipeline layouts to describe configurations of usage for blocks and root constants.

High-level-language shader parameters are bound to a combination of a "binding" and a "set" for Vulkan, which are superficially similar to the registers and spaces of D3D12.
Unlike D3D12, however, bindings and sets in Vulkan directly correspond to the API-provided parameter-passing mechanism.
The set index of a parameter indicates the zero-based index of a slot where a block must be passed, and the binding index is the zero-based index of a particular opaque value set into the block.
A shader parameter that will be passed using root constants (rather than via blocks) must be bound to a root-constant offset as part of compilation.

Unlike D3D12, where SRVs, UAVs, etc. use distinct classes of registers, all opaque-type shader parameters use the same index space of bindings.
That is, a buffer and a texture both using `binding=2` in `set=3` for Vulkan will alias the same slot in the same block.

The Vulkan ray tracing pipeline also uses a shader table, and also forms hit groups similar to D3D12.
Unlike D3D12, each shader table entry in Vulkan can only be used to pass ordinary values (akin to root constants), and cannot be configured for binding of opaque types or blocks.

## OpenGL

> #### Note
>
> Slang has only limited support for compiling code for OpenGL.

OpenGL has existed for many years, and predates programmable GPU pipelines of the kind this chapter discusses; we will focus solely on use of OpenGL as an API for programmable GPU pipelines.

OpenGL is a cross-platform GPU API for graphics and compute with a detailed specification produced by a multi-vendor standard body.
In contrast with Vulkan, OpenGL provides many convenience and safety features that can simplify GPU programming.

OpenGL allows kernels to be loaded as SPIR-V binaries, vendor-specific binaries, or using GLSL source code.
Loading shaders as GLSL source code is the most widely supported of these options, such that GLSL is the _de facto_ intermediate language of OpenGL.

### Pipelines

OpenGL supports rasterization and compute pipelines with the same stages as described for D3D11.
The OpenGL rasterization pipeline also supports the same mesh shader stages that are supported by D3D12.

### Parameter Passing

OpenGL uses slots for binding.
There are distinct kinds of slots for buffers and textures/images, and each set of slots is shared by all pipeline stages.

High-level-language shader parameters are bounding to a "binding" index for OpenGL.
The binding index of a parameter is the zero-based index of the slot (of the appropriate kind) that must be used to pass an argument value.

Note that while OpenGL and Vulkan both use binding indices for shader parameters like textures, the semantics of those are different because OpenGL uses distinct slots for passing buffers and textures.
For OpenGL it is legal to have a texture that uses `binding=2` and a buffer that uses `binding=2` in the same kernel, because those are indices of distinct kinds of slots, while this scenario would typically be invalid for Vulkan.

## Metal

> #### Note
>
> Slang support for Metal is a work in progress.

Metal is Apple's proprietary graphics and compute API for iOS and macOS
platforms. It provides a modern, low-overhead architecture similar to Direct3D
12 and Vulkan.

Metal kernels must be compiled to the Metal Shading Language (MSL), which is
based on C++14 with additional GPU-specific features and constraints. Unlike
some other APIs, Metal does not use an intermediate representation - MSL source
code is compiled directly to platform-specific binaries by Apple's compiler.

### Pipelines

Metal supports rasterization, compute, and ray tracing pipelines.

> #### Note
>
> Ray-tracing support for Metal is a work in progress.

The Metal rasterization pipeline includes the following programmable stages:

- The vertex stage outputs vertex data

- The optional mesh stage allows groups of threads to cooperatively generate geometry

- The optional task stage can be used to control mesh shader invocations

- The optional tessellation stages (kernel, post-tessellation vertex) enable hardware tessellation

- The fragment stage processes fragments produced by the rasterizer

### Parameter Passing

Metal uses a combination of slots and blocks for parameter passing:

- Resources (buffers, textures, samplers) are bound to slots using explicit
  binding indices

- Argument buffers (similar to descriptor tables/sets in other APIs) can group
  multiple resources together

- Each resource type (buffer, texture, sampler) has its own independent binding
  space

- Arguments within argument buffers are referenced by offset rather than
  explicit bindings

Unlike some other APIs, Metal:

- Does not support arrays of buffers as of version 3.1
- Shares binding slots across all pipeline stages
- Uses argument buffers that can contain nested resources without consuming additional binding slots

The Metal ray tracing pipeline follows similar parameter passing conventions to
the rasterization and compute pipelines, while adding intersection,
closest-hit, and miss stages comparable to those in Direct3D 12 and Vulkan.

## CUDA and OptiX

> #### Note
>
> Slang support for OptiX is a work in progress.

CUDA C/C++ is a language for expressing heterogeneous CPU and GPU code with a simple interface to invoking GPU compute work.
OptiX is a ray tracing API that uses CUDA C++ as the language for expressing shader code.
We focus here on OptiX version 7 and up.

CUDA and OptiX allow kernels to be loaded as GPU-specific binaries, or using the PTX intermediate language.

### Pipelines

CUDA supports a compute pipeline that is similar to D3D12 or Vulkan, with additional features.

OptiX introduced the style of ray tracing pipeline adopted by D3D12 and Vulkan, and thus uses the same basic stages.

The CUDA system does not currently expose a rasterization pipeline.

### Parameter Passing

Unlike most of the GPU APIs discussed so far, CUDA supports a "flat" memory model with a single virtual address space for all GPU data.
Textures, buffers, etc. are not opaque types, but can instead sit in the same memory as ordinary data like `float`s or `int`s.

With a flat memory model, a distinct notion of a slot or block is not needed.
A slot is just an ordinary memory location that happens to be used to store a value of texture, buffer, or other resource type.
A block is just an ordinary memory buffer that happens to be filled with values of texture/buffer/etc. type.

CUDA provides two parameter-passing mechanisms for the compute pipeline.
First, when invoking a compute kernel, the application passes a limited number of bytes of parameter data that act as root constants.
Second, each loaded module of GPU code may contain pre-allocated "constant memory" storage which can be initialized from the host and then read by GPU code.
Because types like blocks or textures are not special in CUDA, either of these mechanisms can be utilized to pass any kind of data including references to pointer-based data structures stored in the GPU virtual address space.
The use of "slots" or "blocks" or "root constants" is a matter of application policy instead of API mechanism.

OptiX supports use of constant memory storage for ray tracing pipelines, where all the stages in a ray tracing pipeline share that storage.
OptiX uses a shader table for managing kernels and hit groups, and allows kernels to access the bytes of their shader table entry via a pointer.
Similar to the compute pipeline, application code can layer many different policies on top of these mechanisms.

## CPU Compute

> #### Note
>
> Slang's support for CPU compute is functional, but not feature- or performance-complete.
> Backwards-incompatible changes to this target may come in future versions of Slang.

For the purposes of Slang, different CPU-based host platforms are largely the same.
All support binary code in a native machine-code format.
All CPU platforms Slang supports use a flat memory model with a single virtual address space, where any data type can be stored at any virtual address.

Note that this section considers CPU-based platforms only as targets for kernel compilation; using a CPU as a target for scalar "host" code is an advanced target beyond the scope of this document.

### Pipelines

Slang's CPU compute target supports only a compute pipeline.

### Parameter Passing

Because CPU target support flexible pointer-based addressing and large low-latency caches, a compute kernel can simply be passed a small fixed number of pointers and be relied upon to load parameter values of any types via indirection through those pointers.

## WebGPU

> #### Note
>
> Slang support for WebGPU is work in progress.

WebGPU is a graphics and compute API.
It is similar in spirit to modern APIs, like Metal, Direct3D 12 and Vulkan, but with concessions to portability and privacy.

WebGPU is available both in browsers as a JavaScript API, and natively as a C/C++ API.
[Dawn](https://github.com/google/dawn), is a native WebGPU implementation used by the Chrome browser.

By combining Slang, [Dawn](https://github.com/google/dawn) and [Emscripten](https://emscripten.org/),
an application can easily target any native API, and the web, with a single codebase consisting of C++ and Slang code.

WebGPU shader modules are created from WGSL (WebGPU Shading Language) source files.
WebGPU does not use an intermediate representation - WGSL code is compiled to backend-specific code by
compilers provided by the WebGPU implementation.

### Pipelines

WebGPU supports render and compute pipelines.

The WebGPU render pipeline includes the following programmable stages:

- The vertex stage outputs vertex data

- The fragment stage outputs fragments

### Parameter Passing

WebGPU uses groups of bindings called bind groups to bind things like textures, buffers and samplers.
Bind group objects are passed as arguments when encoding bind group setting commands.

There is a notion of equivalence for bind groups, and a notion of equivalence for pipelines defined in
terms of bind group equivalence.
This equivalence allows an application to save some bind group setting commands, when switching between
pipelines, if bindings are grouped together appropriately.

Which bindings are grouped together can be controlled using Slang's `ParameterBlock` generic type.

## Summary

This chapter has reviewed the main target platforms supported by the Slang compiler and runtime system.
A key point to take away is that there is great variation in the capabilities of these systems.
Even superficially similar graphics APIs have complicated differences in their parameter-passing mechanisms that must be accounted for by application programmers and GPU compilers.
