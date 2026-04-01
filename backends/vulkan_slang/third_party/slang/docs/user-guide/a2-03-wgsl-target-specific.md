---
layout: user-guide
permalink: /user-guide/wgsl-target-specific
---

WGSL-Specific Functionalities
=============================

This chapter provides information for WGSL (WebGPU Shading Language)-specific functionalities and behaviors.


System-Value semantics
----------------------

The system-value semantics are translated to the following WGSL code.

| SV semantic name | WGSL code |
|--|--|
| SV_Barycentrics | *Not supported* |
| SV_ClipDistance<N> | *Not supported* |
| SV_CullDistance<N> | *Not supported* |
| SV_Coverage | `@builtin(sample_mask)` |
| SV_CullPrimitive | *Not supported* |
| SV_Depth | `@builtin(frag_depth)` |
| SV_DepthGreaterEqual | *Not supported* |
| SV_DepthLessEqual | *Not supported* |
| SV_DispatchThreadID | `@builtin(global_invocation_id)` |
| SV_DomainLocation | *Not supported* |
| SV_GSInstanceID | *Not supported* |
| SV_GroupID | `@builtin(workgroup_id)` |
| SV_GroupIndex | `@builtin(local_invocation_index)` |
| SV_GroupThreadID | `@builtin(local_invocation_id)` |
| SV_InnerCoverage | *Not supported* |
| SV_InsideTessFactor | *Not supported* |
| SV_InstanceID | `@builtin(instance_index)` |
| SV_IntersectionAttributes | *Not supported* |
| SV_IsFrontFace | `@builtin(front_facing)` |
| SV_OutputControlPointID | *Not supported* |
| SV_PointSize | *Not supported* |
| SV_Position | `@builtin(position)` |
| SV_PrimitiveID | *Not supported* |
| SV_RenderTargetArrayIndex | *Not supported* |
| SV_SampleIndex | `@builtin(sample_index)` |
| SV_ShadingRate | *Not supported* |
| SV_StartVertexLocation | *Not supported* |
| SV_StartInstanceLocation | *Not supported* |
| SV_StencilRef | *Not supported* |
| SV_Target<N> | *Not supported* |
| SV_TessFactor | *Not supported* |
| SV_VertexID | `@builtin(vertex_index)` |
| SV_ViewID | *Not supported* |
| SV_ViewportArrayIndex | *Not supported* |


Supported HLSL features when targeting WGSL
-------------------------------------------

The following table lists Slang's support for various HLSL feature sets, when targeting WGSL.

| Feature set | Supported |
| -- | -- |
| ray tracing | No |
| inline ray tracing | No |
| mesh shader | No |
| tessellation shader | No |
| geometry shader | No |
| wave intrinsics | No |
| barriers | Yes |
| atomics | Yes |


Supported atomic types
----------------------

The following table shows what is supported when targeting WGSL:

|              |  32-bit integer | 64-bit integer  |      32-bit float     |  64-bit float    |   16-bit float   |
|--------------|-----------------|-----------------|-----------------------|------------------|------------------|
| Supported?   |   Yes           |     No          |    No                 |       No         |      No          |


ConstantBuffer, (RW/RasterizerOrdered)StructuredBuffer, (RW/RasterizerOrdered)ByteAddressBuffer
-----------------------------------------------------------------------------------------------

ConstantBuffer translates to the `uniform` address space with `read` access mode in WGSL.
ByteAddressBuffer and RWByteAddressBuffer translate to `array<u32>` in the `storage` address space, with the `read` and `read_write` access modes in WGSL, respectively.
StructuredBuffer and RWStructuredBuffer with struct type T translate to `array<T>` in the `storage` address space, with with the `read` and `read_write` access modes in WGSL, respectively.


Specialization Constants
------------------------

Specialization constants are not supported when targeting WGSL, at the moment.
They should map to 'override declarations' in WGSL, however this is not yet implemented.


Interlocked operations
----------------------

The InterlockedAdd, InterlockedAnd, etc... functions are not supported when targeting WGSL.
Instead, operations on [`Atomic<T>`](https://shader-slang.com/stdlib-reference/types/atomic-0/index) types should be used.


Entry Point Parameter Handling
------------------------------

Slang performs several transformations on entry point parameters when targeting WGSL:

- Struct parameters and returned structs are flattened to eliminate nested structures.
- System value semantics are translated to WGSL built-ins. (See the `@builtin` attribute, and the table above.)
- Parameters without semantics are given automatic location indices. (See the `@location` attribute.)


Parameter blocks
----------------

Each `ParameterBlock` is assigned its own bind group in WGSL.


Write-only Textures
---------------

Many image formats supported by WebGPU can only be accessed in compute shader as a write-only image.
Use `WTexture2D` type (similar to `RWTexture2D`) to write to an image when possible.
The write-only texture types are also supported when targeting HLSL/GLSL/SPIR-V/Metal and CUDA.


Pointers
--------

`out` and `inout` parameters in Slang are translated to pointer-typed parameters in WGSL.
At callsites, a pointer value is formed and passed as argument using the `&` operator in WGSL.

Since WGSL cannot form pointers to fields of structs (or fields of fields of structs, etc...), the described transformation cannot be done in a direct way when a function argument expression is an "access chain" like `myStruct.myField` or `myStruct.myStructField.someField`.
In those cases, the argument is copied to a local variable, the address of the local variable is passed to the function, and then the local
variable is written back to the struct field after the function call.

Address Space Assignment
------------------------

WGSL requires explicit address space qualifiers. Slang automatically assigns appropriate address spaces:

| Variable Type         | WGSL Address Space  |
| --------------------- | ------------------- |
| Local Variables       | `function`          |
| Global Variables      | `private`           |
| Uniform Buffers       | `uniform`           |
| RW/Structured Buffers | `storage`           |
| Group Shared          | `workgroup`         |
| Parameter Blocks      | `uniform`           |


Matrix type translation
-----------------------

A m-row-by-n-column matrix in Slang, represented as float`m`x`n` or matrix<T, m, n>, is translated to `mat[n]x[m]` in WGSL, i.e. a matrix with `n` columns and `m` rows.
The rationale for this inversion of terminology is the same as [the rationale for SPIR-V](a2-01-spirv-target-specific.md#matrix-type-translation).
Since the WGSL matrix multiplication convention is the normal one, where inner products of rows of the matrix on the left are taken with columns of the matrix on the right, the order of matrix products is also reversed in WGSL. This is relying on the fact that the transpose of a matrix product equals the product of the transposed matrix operands in reverse order.

## Explicit Parameter Binding

The `[vk::binding(index,set)]` attribute is respected when emitting WGSL code, and will translate to `@binding(index) @group(set)` in WGSL.

If the `[vk::binding()]` attribute is not specified but a `:register()` semantic is present, Slang will derive the binding from the `register` semantic the same way as the SPIR-V and GLSL backends.

The `[vk::location(N)]` attributes on stage input/output parameters are respected.

## Specialization Constants

Specialization constants declared with the `[SpecializationConstant]` or `[vk::constant_id]` attribute will be translated into a global `override` declaration when generating WGSL source.
For example:

```csharp
[vk::constant_id(7)]
const int a = 2;
```

Translates to:

```wgsl
@id(7) override a : i32 = 2;
```