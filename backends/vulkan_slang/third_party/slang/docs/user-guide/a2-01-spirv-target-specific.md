---
layout: user-guide
permalink: /user-guide/spirv-target-specific
---

SPIR-V-Specific Functionalities
===============================

This chapter provides information for SPIR-V specific functionalities and behaviors.

Experimental support for the older versions of SPIR-V
-----------------------------------------------------

Slang's SPIR-V backend is stable when emitting SPIR-V 1.3 and later, however, support for SPIR-V 1.0, 1.1 and 1.2 is still experimental.
When targeting the older SPIR-V profiles, Slang may produce SPIR-V that uses the instructions and keywords that were introduced in the later versions of SPIR-V.


Combined texture sampler
------------------------
Slang supports Combined texture sampler such as `Sampler2D`.
Slang emits SPIR-V code with `OpTypeSampledImage` instruction.

For SPIR-V targets, explicit bindings may be provided through a single `vk::binding` decoration.
```
[[vk::binding(1,2)]]
Sampler2D explicitBindingSampler;
```

For other targets (HLSL or others) where combined texture samplers are _not_ supported intrinsically, they are emulated by Slang using separate objects for Texture and Sampler.
For explicit binding on such targets, you can specify two different register numbers for each: one for the texture register and another for the sampler register.
```
Sampler2D explicitBindingSampler : register(t4): register(s3);
```


System-Value semantics
----------------

The system-value semantics are translated to the following SPIR-V code.

| SV semantic name              | SPIR-V Code                       |
|-------------------------------|-----------------------------------|
| `SV_Barycentrics`             | `BuiltIn BaryCoordKHR`            |
| `SV_ClipDistance<N>`          | `BuiltIn ClipDistance`            |
| `SV_CullDistance<N>`          | `BuiltIn CullDistance`            |
| `SV_Coverage`                 | `BuiltIn SampleMask`              |
| `SV_CullPrimitive`            | `BuiltIn CullPrimitiveEXT`        |
| `SV_Depth`                    | `BuiltIn FragDepth`               |
| `SV_DepthGreaterEqual`        | `BuiltIn FragDepth`               |
| `SV_DepthLessEqual`           | `BuiltIn FragDepth`               |
| `SV_DispatchThreadID`         | `BuiltIn GlobalInvocationId`      |
| `SV_DomainLocation`           | `BuiltIn TessCoord`               |
| `SV_DrawIndex`<sup>*</sup>    | `Builtin DrawIndex`               |
| `SV_GSInstanceID`             | `BuiltIn InvocationId`            |
| `SV_GroupID`                  | `BuiltIn WorkgroupId`             |
| `SV_GroupIndex`               | `BuiltIn LocalInvocationIndex`    |
| `SV_GroupThreadID`            | `BuiltIn LocalInvocationId`       |
| `SV_InnerCoverage`            | `BuiltIn FullyCoveredEXT`         |
| `SV_InsideTessFactor`         | `BuiltIn TessLevelInner`          |
| `SV_InstanceID`               | `BuiltIn InstanceIndex`           |
| `SV_IntersectionAttributes`   | *Not supported*                   |
| `SV_IsFrontFace`              | `BuiltIn FrontFacing`             |
| `SV_OutputControlPointID`     | `BuiltIn InvocationId`            |
| `SV_PointSize`<sup>*</sup>    | `BuiltIn PointSize`               |
| `SV_PointCoord`<sup>*</sup>   | `BuiltIn PointCoord`              |
| `SV_Position`                 | `BuiltIn Position/FragCoord`      |
| `SV_PrimitiveID`              | `BuiltIn PrimitiveId`             |
| `SV_RenderTargetArrayIndex`   | `BuiltIn Layer`                   |
| `SV_SampleIndex`              | `BuiltIn SampleId`                |
| `SV_ShadingRate`              | `BuiltIn PrimitiveShadingRateKHR` |
| `SV_StartVertexLocation`      | `BuiltIn BaseVertex`              |
| `SV_StartInstanceLocation`    | `BuiltIn BaseInstance`            |
| `SV_StencilRef`               | `BuiltIn FragStencilRefEXT`       |
| `SV_Target<N>`                | `Location`                        |
| `SV_TessFactor`               | `BuiltIn TessLevelOuter`          |
| `SV_VertexID`                 | `BuiltIn VertexIndex`             |
| `SV_ViewID`                   | `BuiltIn ViewIndex`               |
| `SV_ViewportArrayIndex`       | `BuiltIn ViewportIndex`           |

*Note* that `SV_DrawIndex`, `SV_PointSize` and `SV_PointCoord` are Slang-specific semantics that are not defined in HLSL.


Behavior of `discard` after SPIR-V 1.6
--------------------------------------

`discard` is translated to OpKill in SPIR-V 1.5 and earlier. But it is translated to OpDemoteToHelperInvocation in SPIR-V 1.6.
You can use OpDemoteToHelperInvocation by explicitly specifying the capability, "SPV_EXT_demote_to_helper_invocation".

As an example, the following command-line arguments can control the behavior of `discard` when targeting SPIR-V.
```
slangc.exe test.slang -target spir-v -profile spir-v_1_5 # emits OpKill 
slangc.exe test.slang -target spir-v -profile spir-v_1_6 # emits OpDemoteToHelperInvocation 
slangc.exe test.slang -target spir-v -capability SPV_EXT_demote_to_helper_invocation -profile spir-v_1_5 # emits OpDemoteToHelperInvocation 
```


Supported HLSL features when targeting SPIR-V
---------------------------------------------

Slang supports the following HLSL feature sets when targeting SPIR-V.
 - ray tracing,
 - inline ray tracing,
 - mesh shader,
 - tessellation shader,
 - geometry shader,
 - wave intrinsics,
 - barriers,
 - atomics,
 - and more


Unsupported GLSL keywords when targeting SPIR-V
-----------------------------------------------

Slang doesn't support the following Precision qualifiers in Vulkan.
 - lowp : RelaxedPrecision, on storage variable and operation
 - mediump : RelaxedPrecision, on storage variable and operation
 - highp : 32-bit, same as int or float

Slang ignores the keywords above and all of them are treated as `highp`.


Supported atomic types for each target
--------------------------------------
Shader Model 6.2 introduced [16-bit scalar types](https://github.com/microsoft/DirectXShaderCompiler/wiki/16-Bit-Scalar-Types) such as `float16` and `int16_t`, but they didn't come with any atomic operations.
Shader Model 6.6 introduced [atomic operations for 64-bit integer types and bitwise atomic operations for 32-bit float type](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Int64_and_Float_Atomics.html), but 16-bit integer types and 16-bit float types are not a part of it.

[GLSL 4.3](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.30.pdf) introduced atomic operations for 32-bit integer types.
GLSL 4.4 with [GL_EXT_shader_atomic_int64](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_atomic_int64.txt) can use atomic operations for 64-bit integer types.
GLSL 4.6 with [GLSL_EXT_shader_atomic_float](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_shader_atomic_float.txt) can use atomic operations for 32-bit float type.
GLSL 4.6 with [GLSL_EXT_shader_atomic_float2](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_shader_atomic_float2.txt) can use atomic operations for 16-bit float type.

SPIR-V 1.5 with [SPV_EXT_shader_atomic_float_add](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/EXT/SPV_EXT_shader_atomic_float_add.asciidoc) and [SPV_EXT_shader_atomic_float_min_max](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/EXT/SPV_EXT_shader_atomic_float_min_max.asciidoc) can use atomic operations for 32-bit float type and 64-bit float type.
SPIR-V 1.5 with [SPV_EXT_shader_atomic_float16_add](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/EXT/SPV_EXT_shader_atomic_float16_add.asciidoc) can use atomic operations for 16-bit float type

|        |  32-bit integer | 64-bit integer  |      32-bit float     |  64-bit float    |   16-bit float   |
|--------|-----------------|-----------------|-----------------------|------------------|------------------|
| HLSL   |   Yes (SM5.0)   |   Yes (SM6.6)   | Only bit-wise (SM6.6) |       No         |      No          |
| GLSL   |   Yes (GL4.3)   | Yes (GL4.4+ext) |    Yes (GL4.6+ext)    | Yes (GL4.6+ext)  | Yes (GL4.6+ext)  |
| SPIR-V |   Yes           |     Yes         |    Yes (SPV1.5+ext)   | Yes (SPV1.5+ext) | Yes (SPV1.5+ext) |


ConstantBuffer, StructuredBuffer and ByteAddressBuffer
-----------------------------------------------------------------------------------------------

Each member in a `ConstantBuffer` will be emitted as `uniform` parameter in a uniform block.
StructuredBuffer and ByteAddressBuffer are translated to a shader storage buffer with `readonly` access.
RWStructuredBuffer and RWByteAddressBuffer are translated to a shader storage buffer with `read-write` access.
RasterizerOrderedStructuredBuffer and RasterizerOrderedByteAddressBuffer will use an extension, `SPV_EXT_fragment_shader_interlock`.

If you need to apply a different buffer layout for individual `ConstantBuffer` or `StructuredBuffer`, you can specify the layout as a second generic argument. E.g., `ConstantBuffer<T, Std430DataLayout>`, `StructuredBuffer<T, Std140DataLayout>`, `StructuredBuffer<T, Std430DataLayout>` or `StructuredBuffer<T, ScalarDataLayout>`.

Note that there are compiler options, "-fvk-use-scalar-layout" / "-force-glsl-scalar-layout" and "-fvk-use-dx-layout".
These options do the same but they are applied globally.


ParameterBlock for SPIR-V target
--------------------------------

`ParameterBlock` is a Slang generic type for binding uniform parameters.
In contrast to `ConstantBuffer`, a `ParameterBlock<T>` introduces a new descriptor set ID for resource/sampler handles defined in the element type `T`.

`ParameterBlock` is designed specifically for D3D12/Vulkan/Metal/WebGPU, so that parameters defined in `T` can be placed into an independent descriptor table/descriptor set/argument buffer/binding group.

For example, when targeting Vulkan, when a ParameterBlock doesn't contain nested parameter block fields, it will always map to a single descriptor set, with a dedicated set number and every resources is placed into the set with binding index starting from 0. This allows the user application to create and pre-populate the descriptor set and reuse it during command encoding, without explicitly specifying the binding index for each individual parameter.

When both ordinary data fields and resource typed fields exist in a parameter block, all ordinary data fields will be grouped together into a uniform buffer and appear as a binding 0 of the resulting descriptor set.


Push Constants
---------------------

By default, a `uniform` parameter defined in the parameter list of an entrypoint function is translated to a push constant in SPIRV, if the type of the parameter is ordinary data type (no resources/textures).
All `uniform` parameters defined in global scope are grouped together and placed in a default constant buffer. You can make a global uniform parameter laid out as a push constant by using the `[vk::push_constant]` attribute
on the uniform parameter. All push constants follow the std430 layout by default.

Specialization Constants
------------------------

You can specify a global constant to translate into a SPIRV specialization constant with the `[SpecializationConstant]` attribute.
For example:
```csharp
[SpecializationConstant]
const int myConst = 1; // Maps to a SPIRV specialization constant
```

By default, Slang will automatically assign `constant_id` number for specialization constants. If you wish to explicitly specify them, use `[vk::constant_id]` attribute:
```csharp
[vk::constant_id(1)]
const int myConst = 1;
```

Alternatively, the GLSL `layout` syntax is also supported by Slang:
```glsl
layout(constant_id = 1) const int MyConst = 1;
```

SPIR-V specific Attributes 
--------------------------

DXC supports a few attributes and command-line arguments for targeting SPIR-V. Similar to DXC, Slang supports a few of the attributes as following:

### [[vk::binding(binding: int, set: int = 0)]]
Similar to `binding` layout qualifier in Vulkan. It specifies the uniform buffer binding point, and the descriptor set for Vulkan.

### [[vk::location(X)]]
Same as `location` layout qualifier in Vulkan. For vertex shader inputs, it specifies the number of the vertex attribute from which input values are taken. For inputs of all other shader types, the location specifies a vector number that can be used to match against outputs from a previous shader stage.

### [[vk::index(Y)]]
Same as `index` layout qualifier in Vulkan. It is valid only when used with [[location(X)]]. For fragment shader outputs, the location and index specify the color output number and index receiving the values of the output. For outputs of all other shader stages, the location specifies a vector number that can be used to match against inputs in a subsequent shader stage.

### [[vk::input_attachment_index(i)]]
Same as `input_attachment_index` layout qualifier in Vulkan. It selects which subpass input is being read from. It is valid only when used on subpassInput type uniform variables.

### [[vk::push_constant]]
Same as `push_constant` layout qualifier in Vulkan. It is applicable only to a uniform block and it will be copied to a special memory location where GPU may have a more direct access to.

### [vk::image_format(format : String)]
Same as `[[vk::image_format("XX")]]` layout qualifier in DXC. Vulkan/GLSL allows the format string to be specified without the keyword, `image_format`.  Consider the following Slang code, as an example,
```csharp
[vk::image_format("r32f")] RWTexture2D<float> typicalTexture;
```
It will generate the following GLSL,
> layout(r32f) uniform image2D typicalTexture_0;

Or it will generate the following SPIR-V code,
> %18 = OpTypeImage %float 2D 2 0 0 2 R32f

### [vk::shader_record]
Same as `shaderRecordEXT` layout qualifier in [GL_EXT_ray_tracing extension](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_ray_tracing.txt).
It can be used on a buffer block that represents a buffer within a shader record as defined in the Ray Tracing API.


Multiple entry points support
-----------------------------

To use multiple entry points, you will need to use a compiler option, `-fvk-use-entrypoint-name`.

Because GLSL requires the entry point to be named, "main", a GLSL shader can have only one entry point.
The default behavior of Slang is to rename all entry points to "main" when targeting SPIR-V.

When there are more than one entry point, the default behavior will prevent a shader from having more than one entry point.
To generate a valid SPIR-V with multiple entry points, use `-fvk-use-entrypoint-name` compiler option to disable the renaming behavior and preserve the entry point names.


Global memory pointers
------------------------------

Slang supports global memory pointers when targeting SPIRV. See [an example and explanation](convenience-features.html#pointers-limited).

`float4*` in user code will be translated to a pointer in PhysicalStorageBuffer storage class in SPIRV.
When a slang module uses a pointer type, the resulting SPIRV will be using the SpvAddressingModelPhysicalStorageBuffer64 addressing mode. Modules without use of pointers will use SpvAddressingModelLogical addressing mode.


Matrix type translation
-----------------------

A m-row-by-n-column matrix in Slang, represented as float`m`x`n` or matrix<T, m, n>, is translated to OpTypeMatrix (OpTypeVector(T, n), m) in SPIRV. Note that in SPIR-V terminology, this type is referred to a m-column-by-n-row matrix.

The swap of row and column terminology may seem to be confusing at first, but this is the only translation without needing extra operations that may have negative performance consequences. For example, consider the following Slang code:
```
float3x4 v;
for (int i = 0; i < 3; ++i)
{
  for (int j = 0; j < 4; ++j)
  {
    v[i][j] = i * 4 + j;
  }
}
```
The Slang shader above can iterate each element of a `float3x4` matrix. This is similar to how a multi-dimensional array is handled in C and HLSL. When a matrix type is `float3x4`, the first dimension indexing, `i`, corresponds to the first value specified in the matrix type `3`. And the second dimension indexing, `j`, corresponds to the second value specified in the matrix type `4`.

A matrix in Slang can be also seen as an array of a vector type. And the following code is same as above.
```
float3x4 v;
for (int i = 0; i < 3; ++i)
{
  v[i] = float4(0, 1, 2, 3);
  v[i] += i * 4;
}
```

For the given example above, when targeting SPIR-V, Slang emits a matrix that consists of three vectors each of which has four elements,
```
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4 ; <= float4 type
%mat3v4float = OpTypeMatrix %v4float 3 ; <= three of float4
```

An alternative way to emit SPIR-V code is to emit four vectors and each vector has three elements. Slang doesn't do this but this is a more direct translation because SPIR-V spec defines OpTypeMatrix to take "Column Count" not row.
```
; NOT SLANG EMITTED CODE
%v3float = OpTypeVector %float 3 ; <= float3 type
%mat4v3float = OpTypeMatrix %v3float 4 ; <= four of float3
```
However, this results in a more complicated access pattern to the elements in a matrix, because `v[i]` will no longer correspond to a vector natively when emitted to SPIR-V.

Another way to put, Slang treats column as row and row as column when targeting GLSL or SPIR-V. This is same to [how DXC handles a matrix when emitting SPIR-V](https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#appendix-a-matrix-representation).

Due to the swap of row and column in terminology, the matrix multiplication needs to be performed little differently. Slang translates a matrix multiplication, `mul(mat1, mat2)`, to `transpose(mul(transpose(mat2), transpose(mat1)))` when targeting SPIR-V.

Note that the matrix translation explained above is orthogonal to the memory layout of a matrix. The memory layout is related to how CPU places matrix values in the memory and how GPU reads them. It is like how `std140` or `std430` works. DXC by default uses `column_major` memory layout and Slang uses row-major memory layout. For more information about the matrix memory layout, please see [a1-01-matrix-layout](a1-01-matrix-layout.md).


Legalization
------------

Legalization is a process where Slang applies slightly different approach to translate the input Slang shader to the target.
This process allows Slang shaders to be written in a syntax that SPIR-V may not be able to achieve natively.

Slang allows to use opaque resource types as members of a struct. These members will be hoisted out of struct types and become global variables.

Slang allows functions that return any resource types as return type or `out` parameter as long as things are statically resolvable.

Slang allows functions that return arrays. These functions will be converted to return the array via an out parameter in SPIRV.

Slang allows putting scalar/vector/matrix/array types directly as element type of a constant buffer or structured buffers. Such element types will be wrapped in a struct type when emitting to SPIRV.

When RasterizerOrder resources are used, the order of the rasterization is guaranteed by the instructions from `SPV_EXT_fragment_shader_interlock` extension.

A `StructuredBuffer` with a primitive type such as `StructuredBuffer<int> v` is translated to a buffer with a struct that has the primitive type, which is more like `struct Temp { int v; }; StructuredBuffer<Temp> v;`. It is because, SPIR-V requires buffer variables to be declared within a named buffer block.

When `pervertex` keyword is used, the given type for the varying input will be translated into an array of the given type whose element size is 3. It is because each triangle consists of three vertices.


Tessellation
------------

In HLSL and Slang, Hull shader requires two functions: a Hull shader and patch function.
A typical example of a Hull shader will look like the following.
```
// Hull Shader (HS)
[domain("quad")]
[patchconstantfunc("constants")]
HS_OUT main(InputPatch<VS_OUT, 4> patch, uint i : SV_OutputControlPointID)
{
  ...
}
HSC_OUT constants(InputPatch<VS_OUT, 4> patch)
{
  ...
}
```

When targeting SPIR-V, the patch function is merged as a part of the Hull shader, because SPIR-V doesn't have a same concept as `patchconstantfunc`.
The function used for `patchconstantfunc` should be called only once for each patch.

As an example, the Hull shader above will be emitted as following,
```
void main() {
    ...
    main(patch, gl_InvocationID);
    barrier(); // OpControlBarrier
    if (gl_InvocationID == 0)
    {
        constants(path);
    }
}
```

This behavior is same to [how DXC translates Hull shader from HLSL to SPIR-V](https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#patch-constant-function).


SPIR-V specific Compiler options
--------------------------------

The following compiler options are specific to SPIR-V.

### -emit-spirv-directly
Generate SPIR-V output directly (default)
It cannot be used with -emit-spirv-via-glsl

### -emit-spirv-via-glsl
Generate SPIR-V output by compiling to glsl source first, then use glslang compiler to produce SPIRV from the glsl.
It cannot be used with -emit-spirv-directly

### -g
Include debug information in the generated code, where possible.
When targeting SPIR-V, this option emits [SPIR-V NonSemantic Shader DebugInfo Instructions](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc).

### -O<optimization-level>
Set the optimization level.
Under `-O0` option, Slang will not perform extensive inlining for all functions calls, instead it will preserve the call graph as much as possible to help with understanding the SPIRV structure and diagnosing any downstream toolchain issues.

### -fvk-{b|s|t|u}-shift <N> <space>
For example '-fvk-b-shift <N> <space>' shifts by N the inferred binding
numbers for all resources in 'b' registers of space <space>. For a resource attached with :register(bX, <space>)
but not [vk::binding(...)], sets its Vulkan descriptor set to <space> and binding number to X + N. If you need to
shift the inferred binding numbers for more than one space, provide more than one such option. If more than one
such option is provided for the same space, the last one takes effect. If you need to shift the inferred binding
numbers for all sets, use 'all' as <space>.

For more information, see the following pages:
 - [DXC description](https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#implicit-binding-number-assignment)
 - [GLSL wiki](https://github.com/KhronosGroup/glslang/wiki/HLSL-FAQ#auto-mapped-binding-numbers)

### -fvk-bind-globals <N> <descriptor-set>
Places the $Globals cbuffer at descriptor set <descriptor-set> and binding <N>.
It lets you specify the descriptor for the source at a certain register.

For more information, see the following pages:
 - [DXC description](https://github.com/Microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst#hlsl-global-variables-and-vulkan-binding)

### -fvk-use-scalar-layout, -force-glsl-scalar-layout
Make data accessed through ConstantBuffer, ParameterBlock, StructuredBuffer, ByteAddressBuffer and general pointers follow the 'scalar' layout when targeting GLSL or SPIRV.

### -fvk-use-gl-layout
Use std430 layout instead of D3D buffer layout for raw buffer load/stores.

### -fvk-use-dx-layout
Pack members using FXCs member packing rules when targeting GLSL or SPIRV.

### -fvk-use-entrypoint-name
Uses the entrypoint name from the source instead of 'main' in the spirv output.

### -fspv-reflect
Include reflection decorations in the resulting SPIR-V for shader parameters.

### -spirv-core-grammar
A path to a specific spirv.core.grammar.json to use when generating SPIR-V output


