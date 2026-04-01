---
layout: user-guide
permalink: /user-guide/metal-target-specific
---

# Metal-Specific Functionalities

This chapter provides information for Metal-specific functionalities and
behaviors in Slang.

## Entry Point Parameter Handling

Slang performs several transformations on entry point parameters when targeting Metal:

- Struct parameters are flattened to eliminate nested structures
- Input parameters with varying inputs are packed into a single struct
- System value semantics are translated to Metal attributes
- Parameters without semantics are given automatic attribute indices

## System-Value semantics

The system-value semantics are translated to the following Metal attributes:

| SV semantic name            | Metal attribute                                      |
| --------------------------- | ---------------------------------------------------- |
| `SV_Position`               | `[[position]]`                                       |
| `SV_Coverage`               | `[[sample_mask]]`                                    |
| `SV_Depth`                  | `[[depth(any)]]`                                     |
| `SV_DepthGreaterEqual`      | `[[depth(greater)]]`                                 |
| `SV_DepthLessEqual`         | `[[depth(less)]]`                                    |
| `SV_DispatchThreadID`       | `[[thread_position_in_grid]]`                        |
| `SV_GroupID`                | `[[threadgroup_position_in_grid]]`                   |
| `SV_GroupThreadID`          | `[[thread_position_in_threadgroup]]`                 |
| `SV_GroupIndex`             | Calculated from `SV_GroupThreadID` and group extents |
| `SV_InstanceID`             | `[[instance_id]]`                                    |
| `SV_IsFrontFace`            | `[[front_facing]]`                                   |
| `SV_PointSize`              | `[[point_size]]`                                     |
| `SV_PointCoord`             | `[[point_coord]]`                                    |
| `SV_PrimitiveID`            | `[[primitive_id]]`                                   |
| `SV_RenderTargetArrayIndex` | `[[render_target_array_index]]`                      |
| `SV_SampleIndex`            | `[[sample_id]]`                                      |
| `SV_Target<N>`              | `[[color(N)]]`                                       |
| `SV_VertexID`               | `[[vertex_id]]`                                      |
| `SV_ViewportArrayIndex`     | `[[viewport_array_index]]`                           |
| `SV_StartVertexLocation`    | `[[base_vertex]]`                                    |
| `SV_StartInstanceLocation`  | `[[base_instance]]`                                  |

Custom semantics are mapped to user attributes:

- `[[user(SEMANTIC_NAME)]]` For non-system value semantics
- `[[user(SEMANTIC_NAME_INDEX)]]` When semantic has an index

## Interpolation Modifiers

Slang maps interpolation modifiers to Metal's interpolation attributes:

| Slang Interpolation | Metal Attribute             |
| ------------------- | --------------------------- |
| `nointerpolation`   | `[[flat]]`                  |
| `noperspective`     | `[[center_no_perspective]]` |
| `linear`            | `[[sample_no_perspective]]` |
| `sample`            | `[[sample_perspective]]`    |
| `centroid`          | `[[center_perspective]]`    |

## Resource Types

Resource types are translated with appropriate Metal qualifiers:

| Slang Type            | Metal Translation  |
| --------------------- | ------------------ |
| `Texture2D`           | `texture2d`        |
| `RWTexture2D`         | `texture2d`        |
| `ByteAddressBuffer`   | `uint32_t device*` |
| `StructuredBuffer<T>` | `device* T`        |
| `ConstantBuffer<T>`   | `constant* T`      |

| Slang Type                        | Metal Translation                     |
| --------------------------------- | ------------------------------------- |
| `Texture1D`                       | `texture1d`                           |
| `Texture1DArray`                  | `texture1d_array`                     |
| `RWTexture1D`                     | `texture1d`                           |
| `RWTexture1DArray`                | `texture1d_array`                     |
| `Texture2D`                       | `texture2d`                           |
| `Texture2DArray`                  | `texture2d_array`                     |
| `RWTexture2D`                     | `texture2d`                           |
| `RWTexture2DArray`                | `texture2d_array`                     |
| `Texture3D`                       | `texture3d`                           |
| `RWTexture3D`                     | `texture3d`                           |
| `TextureCube`                     | `texturecube`                         |
| `TextureCubeArray`                | `texturecube_array`                   |
| `Buffer<T>`                       | `device* T`                           |
| `RWBuffer<T>`                     | `device* T`                           |
| `ByteAddressBuffer`               | `device* uint32_t`                    |
| `RWByteAddressBuffer`             | `device* uint32_t`                    |
| `StructuredBuffer<T>`             | `device* T`                           |
| `RWStructuredBuffer<T>`           | `device* T`                           |
| `AppendStructuredBuffer<T>`       | `device* T`                           |
| `ConsumeStructuredBuffer<T>`      | `device* T`                           |
| `ConstantBuffer<T>`               | `constant* T`                         |
| `SamplerState`                    | `sampler`                             |
| `SamplerComparisonState`          | `sampler`                             |
| `RaytracingAccelerationStructure` | `(Not supported)`                     |
| `RasterizerOrderedTexture2D`      | `texture2d [[raster_order_group(0)]]` |
| `RasterizerOrderedBuffer<T>`      | `device* T [[raster_order_group(0)]]` |

Raster-ordered access resources receive the `[[raster_order_group(0)]]`
attribute, for example `texture2d<float, access::read_write> tex
[[raster_order_group(0)]]`.

## Array Types

Array types in Metal are declared using the array template:

| Slang Type          | Metal Translation          |
| ------------------- | -------------------------- |
| `ElementType[Size]` | `array<ElementType, Size>` |

## Matrix Layout

Metal exclusively uses column-major matrix layout. Slang automatically handles
the translation of matrix operations to maintain correct semantics:

- Matrix multiplication is transformed to account for layout differences
- Matrix types are declared as `matrix<T, Columns, Rows>`, for example
  `float3x4` is represented as `matrix<float, 3, 4>`

## Mesh Shader Support

Mesh shaders can be targeted using the following types and syntax. The same as task/mesh shaders generally in Slang.

```slang
[outputtopology("triangle")]
[numthreads(12, 1, 1)]
void meshMain(
    in uint tig: SV_GroupIndex,
    in payload MeshPayload meshPayload,
    OutputVertices<Vertex, MAX_VERTS> verts,
    OutputIndices<uint3, MAX_PRIMS> triangles,
    OutputPrimitives<Primitive, MAX_PRIMS> primitives
    )
```

## Header Inclusions and Namespace

When targeting Metal, Slang automatically includes the following headers, these
are available to any intrinsic code.

```cpp
#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
```

## Parameter blocks and Argument Buffers

`ParameterBlock` values are translated into _Argument Buffers_ potentially
containing nested resources. For example, this Slang code...

```slang
struct MyParameters
{
    int x;
    int y;
    StructuredBuffer<float> buffer1;
    RWStructuredBuffer<uint3> buffer2;
}

ParameterBlock<MyParameters> gObj;

void main(){ ... gObj ... }
```

... results in this Metal output:

```cpp
struct MyParameters
{
    int x;
    int y;
    float device* buffer1;
    uint3 device* buffer2;
};

[[kernel]] void main(MyParameters constant* gObj [[buffer(1)]])
```

## Struct Parameter Flattening

When targeting Metal, top-level nested struct parameters are automatically
flattened. For example:

```slang
struct NestedStruct
{
    float2 uv;
};
struct InputStruct
{
    float4 position;
    float3 normal;
    NestedStruct nested;
};
```

Will be flattened to:

```cpp
struct InputStruct
{
    float4 position;
    float3 normal;
    float2 uv;
};
```

## Return Value Handling

Non-struct return values from entry points are automatically wrapped in a
struct with appropriate semantics. For example:

```slang
float4 main() : SV_Target
{
    return float4(1,2,3,4);
}
```

becomes:

```c++
struct FragmentOutput
{
    float4 value : SV_Target;
};
FragmentOutput main()
{
    return { float4(1,2,3,4) };
}
```

## Value Type Conversion

Metal enforces strict type requirements for certain operations. Slang
automatically performs the following conversions:

- Vector size expansion (e.g., `float2` to `float4`), for example when the user
  specified `float2` but the semantic type in Metal is `float4`.
- Image store value expansion to 4-components

For example:

```slang
RWTexture2D<float2> tex;
tex[coord] = float2(1,2);  // Automatically expanded to float4(1,2,0,0)
```

## Conservative Rasterization

Since Metal doesn't support conservative rasterization, SV_InnerCoverage is always false.

## Address Space Assignment

Metal requires explicit address space qualifiers. Slang automatically assigns appropriate address spaces:

| Variable Type         | Metal Address Space |
| --------------------- | ------------------- |
| Local Variables       | `thread`            |
| Global Variables      | `device`            |
| Uniform Buffers       | `constant`          |
| RW/Structured Buffers | `device`            |
| Group Shared          | `threadgroup`       |
| Parameter Blocks      | `constant`          |

## Explicit Parameter Binding

The HLSL `:register()` semantic is respected when emitting Metal code.

Since Metal does not differentiate between a constant buffer, a shader resource (read-only) buffer and an unordered access buffer, Slang will map `register(tN)`, `register(uN)` and `register(bN)` to `[[buffer(N)]]` when such `register` semantic is declared on a buffer-typed parameter.

`spaceN` specifiers inside `register` semantics are ignored.

The `[vk::location(N)]` attributes on stage input/output parameters are respected.

## Specialization Constants

Specialization constants declared with the `[SpecializationConstant]` or `[vk::constant_id]` attribute will be translated into a `function_constant` when generating Metal source.
For example:

```csharp
[vk::constant_id(7)]
const int a = 2;
```

Translates to:

```metal
constant int fc_a_0 [[function_constant(7)]];
constant int a_0 = is_function_constant_defined(fc_a_0) ? fc_a_0 : 2;
```
