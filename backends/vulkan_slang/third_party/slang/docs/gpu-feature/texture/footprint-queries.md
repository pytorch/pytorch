Texture Footprint Queries
=========================

Slang supports querying the *footprint* of a texture sampling operation: the texels that would be accessed when performing that operation.
This feature is supported on Vulkan via the `GL_NV_shader_texture_footprint` extension, and on D3D12 via the `NvFootprint*` functions exposed by NVAPI.

# Background

There are many GPU rendering techniques that involve generating a texture (e.g., by rendering to it) and then sampling from that texture in a 3D rendering pass, such that it is difficult to predict *a priori* which parts of the texture will be accessed, or not.
As one example, consider rendering a shadow map that will be accessed when shading a g-buffer.
Depending on the geometry that was rendered into the g-buffer, and the occlusion that might exist, some parts of the shadow map might not be needed at all.

In principle, an application could use a compute pass on the g-buffer to compute, for each pixel, the part of the shadow-map texture that it will access - its footprint.
The application could then aggregate these footprints into a stencil mask or other data structure that could be used to optimize the rendering pass that generates the shadow map.

Unfortunately, it is almost impossible for applications to accurately and reliably predict the texel data that particular sampling operations will require, once non-trivial texture filtering modes are considered.
Sampling operations support a wide variety of state that affects the lookup and filtering of texels. For example:

* When bilinear filtering is enabled, a sampling operation typically accesses the four texels closest to the sampling location and blends them.

* When trilinear filtering is enabled, a sampling operation may access texels at two different mip levels.

* When anisotropic filtering is enabled, a sampling operation may take up to N *taps* (where N is the maximum supported degree of anisotropy), each of which may itself access a neighborhood of texels to produce a filtered value for that tap.

* When sampling a cube map, a sampling operation may straddle the "seam" between two or even three cube faces.

Texture footprint queries are intended to solve this problem by providing application developers with a primitive that can query the footprint of a texture sampling operation using the exact same sampler state and texture coordinates that will be used when sampling the texture later.

# Slang Shader API

Rather than exactly mirror the Vulkan GLSL extension or the NVAPI functions, the Slang core module provides a single common interface that can map to either of those implementations.

## Basics

A typical 2D texture sampling operation is performed using the `Sample()` method on `Texture2D`:

```hlsl
Texture2D<float4> texture = ...;
SamplerState sampler = ...;
float2 coords = ...;

// Sample a 2D texture
float4 color = texture.Sample(
    sampler, coords);
```

To query the footprint that would be accessed by this operation, we can use an operation like:

```hlsl
uint granularity = ...;
TextureFootprint2D footprint = texture.queryFootprintCoarse(granularity,
    sampler, coords);
```

Note that the same arguments used to call `Sample` above are here passed to `queryFootprint` in the exact same order.
The returned `footprint` encodes a conservative footprint of the texels that would be accessed by the equivalent `Sample` operation above.

Texture footprints are encoded in terms of blocks of texels, and the size of those blocks determined the *granularity* of the footprint.
The `granularity` argument to `queryFootprintCoarse` above indicates the granularity of blocks that the application requests.

In cases where a filtering operation might access two mip levels - one coarse and one fine - a footprint query only returns information about one of the two levels.
The application selects between these options by calling either `queryFootprintCoarse` or `queryFootprintFine`.

## Variations

A wide range of footprint queries are provided, corresponding to various cases of texture sampling operations with different parameters.
For 2D textures, the following functions are supported:

```hlsl
TextureFootprint2D Texture2D.queryFootprintCoarse(
    uint granularity, SamplerState sampler, float2 coords);
TextureFootprint2D Texture2D.queryFootprintFine(
    uint granularity, SamplerState sampler, float2 coords);
TextureFootprint2D Texture2D.queryFootprintCoarseBias(
    uint granularity, SamplerState sampler, float2 coords,
    float lodBias);
TextureFootprint2D Texture2D.queryFootprintFineBias(
    uint granularity, SamplerState sampler, float2 coords,
    float lodBias);
TextureFootprint2D Texture2D.queryFootprintCoarseLevel(
    uint granularity, SamplerState sampler, float2 coords,
    float lod);
TextureFootprint2D Texture2D.queryFootprintFineLevel(
    uint granularity, SamplerState sampler, float2 coords,
    float lod);
TextureFootprint2D Texture2D.queryFootprintCoarseGrad(
    uint granularity, SamplerState sampler, float2 coords,
    float2 dx, float2 dy);
TextureFootprint2D Texture2D.queryFootprintFineGrad(
    uint granularity, SamplerState sampler, float2 coords,
    float2 dx, float2 dy);

// Vulkan-only:
TextureFootprint2D Texture2D.queryFootprintCoarseClamp(
    uint granularity, SamplerState sampler, float2 coords,
    float lodClamp);
TextureFootprint2D Texture2D.queryFootprintFineClamp(
    uint granularity, SamplerState sampler, float2 coords,
    float lodClamp);
TextureFootprint2D Texture2D.queryFootprintCoarseBiasClamp(
    uint granularity, SamplerState sampler, float2 coords,
    float lodBias,
    float lodClamp);
TextureFootprint2D Texture2D.queryFootprintFineBiasClamp(
    uint granularity, SamplerState sampler, float2 coords,
    float lodBias,
    float lodClamp);
TextureFootprint2D Texture2D.queryFootprintCoarseGradClamp(
    uint granularity, SamplerState sampler, float2 coords,
    float2 dx, float2 dy,
    float lodClamp);
TextureFootprint2D Texture2D.queryFootprintFineGradClamp(
    uint granularity, SamplerState sampler, float2 coords,
    float2 dx, float2 dy,
    float lodClamp);
```

For 3D textures, the following functions are supported:

```hlsl
TextureFootprint3D Texture3D.queryFootprintCoarse(
    uint granularity, SamplerState sampler, float3 coords);
TextureFootprint3D Texture3D.queryFootprintFine(
    uint granularity, SamplerState sampler, float3 coords);
TextureFootprint3D Texture3D.queryFootprintCoarseBias(
    uint granularity, SamplerState sampler, float3 coords,
    float lodBias);
TextureFootprint3D Texture3D.queryFootprintFineBias(
    uint granularity, SamplerState sampler, float3 coords,
    float lodBias);
TextureFootprint3D Texture3D.queryFootprintCoarseLevel(
    uint granularity, SamplerState sampler, float3 coords,
    float lod);
TextureFootprint3D Texture3D.queryFootprintFineLevel(
    uint granularity, SamplerState sampler, float3 coords,
    float lod);

// Vulkan-only:
TextureFootprint3D Texture3D.queryFootprintCoarseClamp(
    uint granularity, SamplerState sampler, float3 coords,
    float lodClamp);
TextureFootprint3D Texture3D.queryFootprintFineClamp(
    uint granularity, SamplerState sampler, float3 coords,
    float lodClamp);
TextureFootprint3D Texture3D.queryFootprintCoarseBiasClamp(
    uint granularity, SamplerState sampler, float3 coords,
    float lodBias,
    float lodClamp);
TextureFootprint3D Texture3D.queryFootprintFineBiasClamp(
    uint granularity, SamplerState sampler, float3 coords,
    float lodBias,
    float lodClamp);
```

## Footprint Types

Footprint queries on 2D and 3D textures return values of type `TextureFootprint2D` and `TextureFootprint3D`, respectively, which are built-in `struct`s defined in the Slang core module:

```
struct TextureFootprint2D
{
    typealias Anchor        = uint2;
    typealias Offset        = uint2;
    typealias Mask          = uint2;
    typealias LOD           = uint;
    typealias Granularity   = uint;

    property anchor         : Anchor        { get; }
    property offset         : Offset        { get; }
    property mask           : Mask          { get; }
    property lod            : LOD           { get; }
    property granularity    : Granularity   { get; }
    property isSingleLevel  : bool          { get; }
}

struct TextureFootprint3D
{
    typealias Anchor        = uint3;
    typealias Offset        = uint3;
    typealias Mask          = uint2;
    typealias LOD           = uint;
    typealias Granularity   = uint;

    property anchor         : Anchor        { get; }
    property offset         : Offset        { get; }
    property mask           : Mask          { get; }
    property lod            : LOD           { get; }
    property granularity    : Granularity   { get; }
    property isSingleLevel  : bool          { get; }
}
```

A footprint is encoded in terms of *texel groups*, where the `granularity` determines the size of those groups.
When possible, the returned footprint will match the granularity passed into the query operation, but a larger granularity may be selected in cases where the footprint is too large to encode at the requested granularity.

The `anchor` property specifies an anchor point in the texture, in the vicinity of the footprint. Its components are in multiples of 8 texel groups.

The `offset` property specifies how the bits in `mask` map to texel groups in the vicinity of the `anchor` point.

The `mask` property is a 64-bit bitfield (encoded as a `uint2`), where each bit represents footprint coverage of one texel group, within a 8x8 (for 2D textures) or 4x4x4 neighborhood of texel groups.

The `lod` property indicates the mipmap level that would be accessed by the sampling operation.

The `isSingleLevel` property indicates if the sampling operation is known to access only a single mip level.
Note that this property will always be `false` when using the D3D/NVAPI path.
