Wave Intrinsics
===============

Slang has support for Wave intrinsics introduced to HLSL in [SM6.0](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12) and [SM6.5](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/HLSL_ShaderModel6_5.md). All intrinsics are available on D3D12, and a subset on Vulkan. 

On GLSL targets such as Vulkan wave intrinsics map to ['subgroup' extension] (https://github.com/KhronosGroup/GLSL/blob/master/extensions/khr/GL_KHR_shader_subgroup.txt).  There is no subgroup support for Matrix types, and currently this means that Matrix is not a supported type for Wave intrinsics on Vulkan, but may be in the future.

Also introduced are some 'non standard' Wave intrinsics which are only available on Slang. All WaveMask intrinsics are non standard. Other non standard intrinsics expose more accurately different behaviours which are either not distinguished on HLSL, or perhaps currently unavailable. Two examples would be `WaveShuffle` and `WaveBroadcastLaneAt`. 

There are three styles of wave intrinsics...

## WaveActive

The majority of 'regular' HLSL Wave intrinsics which operate on implicit 'active' lanes. 

In the [DXC Wiki](https://github.com/Microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics) active lanes are described as

> These intrinsics are dependent on active lanes and therefore flow control. In the model of this document, implementations
> must enforce that the number of active lanes exactly corresponds to the programmer’s view of flow control.
 
In practice this appears to imply that the programming model is that all lanes operate in 'lock step'. That the 'active lanes' are the lanes doing processing at a particular point in the control flow. On some hardware this may match how processing actually works. There is also a large amount of hardware in the field that doesn't follow this model, and allows lanes to diverge and not necessarily on flow control. On this style of hardware Active intrinsics may act to also converge lanes to give the appearance of 'in step' ness. 
 
## WaveMask

The WaveMask intrinsics take an explicit mask of lanes to operate on, in the same vein as CUDA. Requesting data from a from an inactive lane, can lead to undefined behavior, that includes locking up the shader. The WaveMask is an integer type that can hold the maximum amount of active lanes for this model - currently 32. In the future the WaveMask type may be made an opaque type, but can largely be operated on as if it is an integer.

Using WaveMask intrinsics is generally more verbose and prone to error than the 'Active' style, but it does have a few advantages

* It works across all supported targets - including CUDA (currently WaveActive intrinics do not)
* Gives more fine control
* Might allow for higher performance (for example it gives more control of divergence)
* Maps most closely to CUDA

On D3D12 and Vulkan the WaveMask intrinsics can be used, but the mask is effectively ignored. For this to work across targets including CUDA, the mask must be calculated such that it exactly matches that of HLSL defined 'active' lanes, else the behavior is undefined.

The WaveMask intrinsics are a non standard Slang feature, and may change in the future. 

```
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // It is the programmers responsibility to determine the initial mask, and that is dependent on the launch
    // It's common to launch such that all lanes are active - with CUDA this would mean 32 lanes. 
    // Here the launch only has 4 lanes active, and so the initial mask is 0xf.
    const WaveMask mask0 = 0xf;
    
    int idx = int(dispatchThreadID.x);
    
    int value = 0;
    
    // When there is a conditional/flow control we typically need to work out a new mask.
    // This can be achieved by calling WaveMaskBallot with the current mask, and the condition
    // used in the flow control - here the subsequent 'if'.
    const WaveMask mask1 = WaveMaskBallot(mask0, idx == 2);
    
    if (idx == 2)
    {
        // At this point the mask is `mask1`, although no WaveMask intrinsics are used along this path, 
        // so it's not used.
    
        // diverge
        return;
    }
    
    // If we get here, the active lanes must be the opposite of mask1 (because we took the other side of the condition), but cannot include
    // any lanes which were not active before. We can calculate this as mask0 & ~mask1.
    
    const WaveMask mask2 = mask0 & ~mask1;
    
    // mask2 holds the correct active mask to use with WaveMaskMin
    value = WaveMaskMin(mask2, idx + 1);
    
    // Write out the result
    outputBuffer[idx] = value;
}
```

Many of the nuances of writing code in this way are discussed in the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions).

The above example written via the regular intrinsics is significantly simpler, as we do not need to track 'active lanes' in the masks. 

```
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int idx = int(dispatchThreadID.x);
    
    int value = 0;
    
    if (idx == 2)
    {    
        // diverge
        return;
    }
    
    value = WaveActiveMin(idx + 1);
    
    // Write out the result
    outputBuffer[idx] = value;
}
```

## WaveMulti

The standard 'Multi' intrinsics were added to HLSL is SM6.5, they can specify a mask of lanes via uint4. They introduce some intrinsics that work in a similar fashion to the `WaveMask` intrinsics. The available intrisnics is currently significantly restricted compared to WaveMask. 

Standard Wave intrinsics
=========================

The Wave Intrinsics supported on Slang are listed below. Note that typically T generic types also include vector and matrix forms. 

```
// Lane info

uint WaveGetLaneCount();

uint WaveGetLaneIndex();

bool WaveIsFirstLane();

// Ballot

bool WaveActiveAllTrue(bool condition);

bool WaveActiveAnyTrue(bool condition);

uint4 WaveActiveBallot(bool condition);

uint WaveActiveCountBits(bool value);

// Across Lanes

T WaveActiveBitAnd<T>(T expr);

T WaveActiveBitOr<T>(T expr);

T WaveActiveBitXor<T>(T expr);

T WaveActiveMax<T>(T expr);

T WaveActiveMin<T>(T expr);

T WaveActiveProduct<T>(T expr);

T WaveActiveSum<T>(T expr);

bool WaveActiveAllEqual<T>(T value);

// Prefix

T WavePrefixProduct<T>(T expr);

T WavePrefixSum<T>(T expr);

// Communication

T WaveReadLaneFirst<T>(T expr);

T WaveReadLaneAt<T>(T value, int lane);

// Prefix

uint WavePrefixCountBits(bool value);

// Shader model 6.5 stuff
// https://github.com/microsoft/DirectX-Specs/blob/master/d3d/HLSL_ShaderModel6_5.md

uint4 WaveMatch<T>(T value);

uint WaveMultiPrefixCountBits(bool value, uint4 mask);

T WaveMultiPrefixBitAnd<T>(T expr, uint4 mask);

T WaveMultiPrefixBitOr<T>(T expr, uint4 mask);

T WaveMultiPrefixBitXor<T>(T expr, uint4 mask);

T WaveMultiPrefixProduct<T>(T value, uint4 mask);

T WaveMultiPrefixSum<T>(T value, uint4 mask);
```

Non Standard Wave Intrinsics
============================

The following intrinsics are not part of the HLSL Wave intrinsics standard, but were added to Slang for a variety of reasons. Within the following signatures T can be scalar, vector or matrix, except on Vulkan which doesn't (currently) support Matrix.

```
T WaveBroadcastLaneAt<T>(T value, constexpr int lane);

T WaveShuffle<T>(T value, int lane);

uint4 WaveGetActiveMulti();

uint4 WaveGetConvergedMulti();

// Barriers 

void AllMemoryBarrierWithWaveSync();

void GroupMemoryBarrierWithWaveSync();
```

## Description

```
T WaveBroadcastLaneAt<T>(T value, constexpr int lane);
```

All lanes receive the value specified in lane. Lane must be an active lane, otherwise the result is undefined. 
This is a more restrictive version of `WaveReadLaneAt` - which can take a non constexpr lane, *but* must be the same value for all lanes in the warp. Or 'dynamically uniform' as described in the HLSL documentation.

```
T WaveShuffle<T>(T value, int lane);
```

Shuffle is a less restrictive version of `WaveReadLaneAt` in that it has no restriction on the lane value - it does *not* require the value to be same on all lanes. 

There isn't explicit support for WaveShuffle in HLSL, and for now it will emit `WaveReadLaneAt`. As it turns out for a sizable set of hardware WaveReadLaneAt does work correctly when the lane is not 'dynamically uniform'. This is not necessarily the case for hardware general though, so if targeting HLSL it is important to make sure that this does work correctly on your target hardware.

Our intention is that Slang will support the appropriate HLSL mechanism that makes this work on all hardware when it's available.  

```
void AllMemoryBarrierWithWaveSync();
```

Synchronizes all lanes to the same AllMemoryBarrierWithWaveSync in program flow. Orders all memory accesses such that accesses after the barrier can be seen by writes before.  

```
void GroupMemoryBarrierWithWaveSync();
```

Synchronizes all lanes to the same GroupMemoryBarrierWithWaveSync in program flow. Orders group shared memory accesses such that accesses after the barrier can be seen by writes before.  

Wave Rotate Intrinsics
======================

These intrinsics are specific to Slang and were added to support the subgroup rotate functionalities provided by SPIRV (through the `GroupNonUniformRotateKHR` capability), GLSL (through the `GL_KHR_shader_subgroup_rotate
` extension), and Metal.

```
// Supported on SPIRV, GLSL, and Metal targets.
T WaveRotate(T value, uint delta);

// Supported on SPIRV and GLSL targets.
T WaveClusteredRotate(T value, uint delta, constexpr uint clusterSize);
```

Wave Mask Intrinsics
====================

CUDA has a different programming model for inter warp/wave communication based around masks of active lanes. This is because the CUDA programming model allows for divergence that is more granualar than just on program flow, and that there isn't implied reconvergence at the end of a conditional. 

In the future Slang may have the capability to work out the masks required such that the regular HLSL Wave intrinsics work. As it stands there does not appear to be any way to implement the regular Wave intrinsics directly. To work around this problem we introduce 'WaveMask' intrinsics, which are essentially the same as the regular HLSL Wave intrinsics with the first parameter as the WaveMask which identifies the participating lanes. 

The WaveMask intrinsics will work across targets, but *only* if on CUDA targets the mask captures exactly the same lanes as the 'Active' lanes concept in HLSL. If the masks deviate then the behavior is undefined. On non CUDA based targets currently the mask is ignored. This behavior may change on GLSL which has an extension to support a more CUDA like behavior.  

Most of the `WaveMask` functions are identical to the regular Wave intrinsics, but they take a WaveMask as the first parameter, and the intrinsic name starts with `WaveMask`. 

```
WaveMask WaveGetConvergedMask();
```

Gets the mask of lanes which are converged within the Wave. Note that this is *not* the same as Active threads, and may be some subset of that. It is equivalent to the `__activemask()` in CUDA.

On non CUDA targets the the function will return all lanes as active - even though this is not the case. This is 'ok' in so far as on non CUDA targets the mask is ignored. It is *not* okay if the code uses the value other than as a superset of the 'really converged' lanes. For example testing the bit's and changing behavior would likely not work correctly on non CUDA targets. 

```
void AllMemoryBarrierWithWaveMaskSync(WaveMask mask);
```

Same as AllMemoryBarrierWithWaveSync but takes a mask of active lanes to sync with. 

```
void GroupMemoryBarrierWithWaveMaskSync(WaveMask mask);
```

Same as GroupMemoryBarrierWithWaveSync but takes a mask of active lanes to sync with. 
 
The intrinsics that make up the Slang `WaveMask` extension. 
 
```
// Lane info

WaveMask WaveGetConvergedMask();

WaveMask WaveGetActiveMask();

bool WaveMaskIsFirstLane(WaveMask mask);

// Ballot

bool WaveMaskAllTrue(WaveMask mask, bool condition);

bool WaveMaskAnyTrue(WaveMask mask, bool condition);

WaveMask WaveMaskBallot(WaveMask mask, bool condition);

WaveMask WaveMaskCountBits(WaveMask mask, bool value);

WaveMask WaveMaskMatch<T>(WaveMask mask, T value);

// Barriers

void AllMemoryBarrierWithWaveMaskSync(WaveMask mask);

void GroupMemoryBarrierWithWaveMaskSync(WaveMask mask);

// Across lane ops

T WaveMaskBitAnd<T>(WaveMask mask, T expr);

T WaveMaskBitOr<T>(WaveMask mask, T expr);

T WaveMaskBitXor<T>(WaveMask mask, T expr);

T WaveMaskMax<T>(WaveMask mask, T expr);

T WaveMaskMin<T>(WaveMask mask, T expr);

T WaveMaskProduct<T>(WaveMask mask, T expr);

T WaveMaskSum<T>(WaveMask mask, T expr);

bool WaveMaskAllEqual<T>(WaveMask mask, T value);

// Prefix

T WaveMaskPrefixProduct<T>(WaveMask mask, T expr);

T WaveMaskPrefixSum<T>(WaveMask mask, T expr);

T WaveMaskPrefixBitAnd<T>(WaveMask mask, T expr);

T WaveMaskPrefixBitOr<T>(WaveMask mask, T expr);

T WaveMaskPrefixBitXor<T>(WaveMask mask, T expr);

uint WaveMaskPrefixCountBits(WaveMask mask, bool value);

// Communication

T WaveMaskReadLaneFirst<T>(WaveMask mask, T expr);

T WaveMaskBroadcastLaneAt<T>(WaveMask mask, T value, constexpr int lane);

T WaveMaskReadLaneAt<T>(WaveMask mask, T value, int lane);

T WaveMaskShuffle<T>(WaveMask mask, T value, int lane);
```
