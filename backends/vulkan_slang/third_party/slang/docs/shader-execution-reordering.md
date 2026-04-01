Shader Execution Reordering (SER)
=================================

Slang provides preliminary support for Shader Execution Reordering (SER). The API hasn't been finalized and may change in the future.

The feature is available on D3D12 via [NVAPI](nvapi-support.md) and on Vulkan through the [GL_NV_shader_invocation_reorder](https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GLSL_NV_shader_invocation_reorder.txt) extension.

## Vulkan

SER as implemented on Vulkan has extra limitations on usage. On D3D via NvAPI `HitObject` variables are like regular variables. They can be assigned, passed to functions and so forth. Using `GL_NV_shader_invocation_reorder` on Vulkan, this isn't the case and `HitObject` variables are special and act is if their introduction allocates a single unique entry. One implication of this is there are limitations on Vulkan around HitObject with flow control, and assignment to HitObject variables. 

TODO: Examples and discussion around these limitation.

## Links

* [SER white paper for NVAPI](https://developer.nvidia.com/sites/default/files/akamai/gameworks/ser-whitepaper.pdf)

# Preliminary API

The API is preliminary and based on the NvAPI SER interface. It may change with future Slang versions.

## Free Functions

* [ReorderThread](#reorder-thread)

--------------------------------------------------------------------------------
# `struct HitObject`

## Description

Immutable data type representing a ray hit or a miss. Can be used to invoke hit or miss shading,
or as a key in ReorderThread. Created by one of several methods described below. HitObject
and its related functions are available in raytracing shader types only.

## Methods

* [TraceRay](#trace-ray)
* [TraceMotionRay](#trace-motion-ray)
* [MakeMiss](#make-miss)
* [MakeHit](#make-hit)
* [MakeMotionHit](#make-motion-hit)
* [MakeMotionMiss](#make-motion-miss)
* [MakeNop](#make-nop)
* [Invoke](#invoke)
* [IsMiss](#is-miss)
* [IsHit](#is-hit)
* [IsNop](#is-nop)
* [GetRayDesc](#get-ray-desc)
* [GetShaderTableIndex](#get-shader-table-index)
* [GetInstanceIndex](#get-instance-index)
* [GetInstanceID](#get-instance-id)
* [GetGeometryIndex](#get-geometry-index)
* [GetPrimitiveIndex](#get-primitive-index)
* [GetHitKind](#get-hit-kind)
* [LoadLocalRootTableConstant](#load-local-root-table-constant)

--------------------------------------------------------------------------------
<a id="trace-ray"></a>
# `HitObject.TraceRay`

## Description

Executes ray traversal (including anyhit and intersection shaders) like TraceRay, but returns the
resulting hit information as a HitObject and does not trigger closesthit or miss shaders.

## Signature 

```
static HitObject HitObject.TraceRay<payload_t>(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                 RayFlags,
    uint                 InstanceInclusionMask,
    uint                 RayContributionToHitGroupIndex,
    uint                 MultiplierForGeometryContributionToHitGroupIndex,
    uint                 MissShaderIndex,
    RayDesc              Ray,
    inout payload_t      Payload);
```

--------------------------------------------------------------------------------
<a id="trace-motion-ray"></a>
# `HitObject.TraceMotionRay`

## Description

Executes motion ray traversal (including anyhit and intersection shaders) like TraceRay, but returns the
resulting hit information as a HitObject and does not trigger closesthit or miss shaders.

## Signature 

```
static HitObject HitObject.TraceMotionRay<payload_t>(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                 RayFlags,
    uint                 InstanceInclusionMask,
    uint                 RayContributionToHitGroupIndex,
    uint                 MultiplierForGeometryContributionToHitGroupIndex,
    uint                 MissShaderIndex,
    RayDesc              Ray,
    float                CurrentTime,
    inout payload_t      Payload);
```


--------------------------------------------------------------------------------
<a id="make-hit"></a>
# `HitObject.MakeHit`

## Description

Creates a HitObject representing a hit based on values explicitly passed as arguments, without
tracing a ray. The primitive specified by AccelerationStructure, InstanceIndex, GeometryIndex,
and PrimitiveIndex must exist. The shader table index is computed using the formula used with
TraceRay. The computed index must reference a valid hit group record in the shader table. The
Attributes parameter must either be an attribute struct, such as
BuiltInTriangleIntersectionAttributes, or another HitObject to copy the attributes from.

## Signature 

```
static HitObject HitObject.MakeHit<attr_t>(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                 InstanceIndex,
    uint                 GeometryIndex,
    uint                 PrimitiveIndex,
    uint                 HitKind,
    uint                 RayContributionToHitGroupIndex,
    uint                 MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc              Ray,
    attr_t               attributes);
static HitObject HitObject.MakeHit<attr_t>(
    uint                 HitGroupRecordIndex,
    RaytracingAccelerationStructure AccelerationStructure,
    uint                 InstanceIndex,
    uint                 GeometryIndex,
    uint                 PrimitiveIndex,
    uint                 HitKind,
    RayDesc              Ray,
    attr_t               attributes);
```

--------------------------------------------------------------------------------
<a id="make-motion-hit"></a>
# `HitObject.MakeMotionHit`

## Description

See MakeHit but handles Motion 
Currently only supported on VK

## Signature 

```
static HitObject HitObject.MakeMotionHit<attr_t>(
    RaytracingAccelerationStructure AccelerationStructure,
    uint                 InstanceIndex,
    uint                 GeometryIndex,
    uint                 PrimitiveIndex,
    uint                 HitKind,
    uint                 RayContributionToHitGroupIndex,
    uint                 MultiplierForGeometryContributionToHitGroupIndex,
    RayDesc              Ray,
    float                CurrentTime,
    attr_t               attributes);
static HitObject HitObject.MakeMotionHit<attr_t>(
    uint                 HitGroupRecordIndex,
    RaytracingAccelerationStructure AccelerationStructure,
    uint                 InstanceIndex,
    uint                 GeometryIndex,
    uint                 PrimitiveIndex,
    uint                 HitKind,
    RayDesc              Ray,
    float                CurrentTime,
    attr_t               attributes);
```

--------------------------------------------------------------------------------
<a id="make-miss"></a>
# `HitObject.MakeMiss`

## Description

Creates a HitObject representing a miss based on values explicitly passed as arguments, without
tracing a ray. The provided shader table index must reference a valid miss record in the shader
table.

## Signature 

```
static HitObject HitObject.MakeMiss(
    uint                 MissShaderIndex,
    RayDesc              Ray);
```

--------------------------------------------------------------------------------
<a id="make-motion-miss"></a>
# `HitObject.MakeMotionMiss`

## Description

See MakeMiss but handles Motion 
Currently only supported on VK

## Signature 

```
static HitObject HitObject.MakeMotionMiss(
    uint                 MissShaderIndex,
    RayDesc              Ray,
    float                CurrentTime);
```

--------------------------------------------------------------------------------
<a id="make-nop"></a>
# `HitObject.MakeNop`

## Description

Creates a HitObject representing “NOP” (no operation) which is neither a hit nor a miss. Invoking a
NOP hit object using HitObject::Invoke has no effect. Reordering by hit objects using
ReorderThread will group NOP hit objects together. This can be useful in some reordering
scenarios where future control flow for some threads is known to process neither a hit nor a
miss.

## Signature 

```
static HitObject HitObject.MakeNop();
```

--------------------------------------------------------------------------------
<a id="invoke"></a>
# `HitObject.Invoke`

## Description

Invokes closesthit or miss shading for the specified hit object. In case of a NOP HitObject, no
shader is invoked.

## Signature 

```
static void HitObject.Invoke<payload_t>(
    RaytracingAccelerationStructure AccelerationStructure,
    HitObject            HitOrMiss,
    inout payload_t      Payload);
```

--------------------------------------------------------------------------------
<a id="is-miss"></a>
# `HitObject.IsMiss`

## Description

Returns true if the HitObject encodes a miss, otherwise returns false.

## Signature 

```
bool HitObject.IsMiss();
```

--------------------------------------------------------------------------------
<a id="is-hit"></a>
# `HitObject.IsHit`

## Description

Returns true if the HitObject encodes a hit, otherwise returns false.

## Signature 

```
bool HitObject.IsHit();
```

--------------------------------------------------------------------------------
<a id="is-nop"></a>
# `HitObject.IsNop`

## Description

Returns true if the HitObject encodes a nop, otherwise returns false.

## Signature 

```
bool HitObject.IsNop();
```

--------------------------------------------------------------------------------
<a id="get-ray-desc"></a>
# `HitObject.GetRayDesc`

## Description

Queries ray properties from HitObject. Valid if the hit object represents a hit or a miss.

## Signature 

```
RayDesc HitObject.GetRayDesc();
```

--------------------------------------------------------------------------------
<a id="get-shader-table-index"></a>
# `HitObject.GetShaderTableIndex`

## Description

Queries shader table index from HitObject. Valid if the hit object represents a hit or a miss.

## Signature 

```
uint HitObject.GetShaderTableIndex();
```

--------------------------------------------------------------------------------
<a id="get-instance-index"></a>
# `HitObject.GetInstanceIndex`

## Description

Returns the instance index of a hit. Valid if the hit object represents a hit.

## Signature 

```
uint HitObject.GetInstanceIndex();
```

--------------------------------------------------------------------------------
<a id="get-instance-id"></a>
# `HitObject.GetInstanceID`

## Description

Returns the instance ID of a hit. Valid if the hit object represents a hit.

## Signature 

```
uint HitObject.GetInstanceID();
```

--------------------------------------------------------------------------------
<a id="get-geometry-index"></a>
# `HitObject.GetGeometryIndex`

## Description

Returns the geometry index of a hit. Valid if the hit object represents a hit.

## Signature 

```
uint HitObject.GetGeometryIndex();
```

--------------------------------------------------------------------------------
<a id="get-primitive-index"></a>
# `HitObject.GetPrimitiveIndex`

## Description

Returns the primitive index of a hit. Valid if the hit object represents a hit.

## Signature 

```
uint HitObject.GetPrimitiveIndex();
```

--------------------------------------------------------------------------------
<a id="get-hit-kind"></a>
# `HitObject.GetHitKind`

## Description

Returns the hit kind. Valid if the hit object represents a hit.

## Signature 

```
uint HitObject.GetHitKind();
```

--------------------------------------------------------------------------------
<a id="get-attributes"></a>
# `HitObject.GetAttributes`

## Description

Returns the attributes of a hit. Valid if the hit object represents a hit or a miss.

## Signature 

```
attr_t HitObject.GetAttributes<attr_t>();
```

--------------------------------------------------------------------------------
<a id="load-local-root-table-constant"></a>
# `HitObject.LoadLocalRootTableConstant`

## Description

Loads a root constant from the local root table referenced by the hit object. Valid if the hit object
represents a hit or a miss. RootConstantOffsetInBytes must be a multiple of 4.

## Signature 

```
uint HitObject.LoadLocalRootTableConstant(uint RootConstantOffsetInBytes);
```

--------------------------------------------------------------------------------
<a id="reorder-thread"></a>
# `ReorderThread`

## Description

Reorders threads based on a coherence hint value. NumCoherenceHintBits indicates how many of
the least significant bits of CoherenceHint should be considered during reordering (max: 16).
Applications should set this to the lowest value required to represent all possible values in
CoherenceHint. For best performance, all threads should provide the same value for
NumCoherenceHintBits.
Where possible, reordering will also attempt to retain locality in the thread’s launch indices
(DispatchRaysIndex in DXR).

`ReorderThread(HitOrMiss)` is equivalent to

```
void ReorderThread( HitObject HitOrMiss, uint CoherenceHint, uint NumCoherenceHintBitsFromLSB );
```

With CoherenceHint and NumCoherenceHintBitsFromLSB as 0, meaning they are ignored.

## Signature 

```
void ReorderThread(
    uint                 CoherenceHint,
    uint                 NumCoherenceHintBitsFromLSB);
void ReorderThread(
    HitObject            HitOrMiss,
    uint                 CoherenceHint,
    uint                 NumCoherenceHintBitsFromLSB);
void ReorderThread(HitObject HitOrMiss);
```
