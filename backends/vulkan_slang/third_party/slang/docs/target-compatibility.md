# Slang Target Compatibility

Shader Model (SM) numbers are D3D Shader Model versions, unless explicitly stated otherwise.
OpenGL compatibility is not listed here, because OpenGL isn't an officially supported target.

Items with a + means that the feature is anticipated to be added in the future.
Items with ^ means there is some discussion about support later in the document for this target.

| Feature                                              | D3D11 | D3D12     | VK      | CUDA           | Metal | CPU       |
| ---------------------------------------------------- | ----- | --------- | ------- | -------------- | ----- | --------- |
| [Half Type](#half)                                   | No    | Yes ^     | Yes     | Yes ^          | Yes   | No +      |
| Double Type                                          | Yes   | Yes       | Yes     | Yes            | No    | Yes       |
| Double Intrinsics                                    | No    | Limited + | Limited | Most           | No    | Yes       |
| [u/int8_t Type](#int8_t)                             | No    | No        | Yes ^   | Yes            | Yes   | Yes       |
| [u/int16_t Type](#int16_t)                           | No    | Yes ^     | Yes ^   | Yes            | Yes   | Yes       |
| [u/int64_t Type](#int64_t)                           | No    | Yes ^     | Yes     | Yes            | Yes   | Yes       |
| u/int64_t Intrinsics                                 | No    | No        | Yes     | Yes            | Yes   | Yes       |
| [int matrix](#int-matrix)                            | Yes   | Yes       | No +    | Yes            | No    | Yes       |
| [tex.GetDimensions](#tex-get-dimensions)             | Yes   | Yes       | Yes     | No             | Yes   | Yes       |
| [SM6.0 Wave Intrinsics](#sm6-wave)                   | No    | Yes       | Partial | Yes ^          | No    | No        |
| SM6.0 Quad Intrinsics                                | No    | Yes       | No +    | No             | No    | No        |
| [SM6.5 Wave Intrinsics](#sm6.5-wave)                 | No    | Yes ^     | No +    | Yes ^          | No    | No        |
| [WaveMask Intrinsics](#wave-mask)                    | Yes ^ | Yes ^     | Yes +   | Yes            | No    | No        |
| [WaveShuffle](#wave-shuffle)                         | No    | Limited ^ | Yes     | Yes            | No    | No        |
| [Tesselation](#tesselation)                          | Yes ^ | Yes ^     | No +    | No             | No    | No        |
| [Graphics Pipeline](#graphics-pipeline)              | Yes   | Yes       | Yes     | No             | Yes   | No        |
| [Ray Tracing DXR 1.0](#ray-tracing-1.0)              | No    | Yes ^     | Yes ^   | No             | No    | No        |
| Ray Tracing DXR 1.1                                  | No    | Yes       | No +    | No             | No    | No        |
| [Native Bindless](#native-bindless)                  | No    | No        | No      | Yes            | No    | Yes       |
| [Buffer bounds](#buffer-bounds)                      | Yes   | Yes       | Yes     | Limited ^      | No ^  | Limited ^ |
| [Resource bounds](#resource-bounds)                  | Yes   | Yes       | Yes     | Yes (optional) | Yes   | Yes       |
| Atomics                                              | Yes   | Yes       | Yes     | Yes            | Yes   | Yes       |
| Group shared mem/Barriers                            | Yes   | Yes       | Yes     | Yes            | Yes   | No +      |
| [TextureArray.Sample float](#tex-array-sample-float) | Yes   | Yes       | Yes     | No             | Yes   | Yes       |
| [Separate Sampler](#separate-sampler)                | Yes   | Yes       | Yes     | No             | Yes   | Yes       |
| [tex.Load](#tex-load)                                | Yes   | Yes       | Yes     | Limited ^      | Yes   | Yes       |
| [Full bool](#full-bool)                              | Yes   | Yes       | Yes     | No             | Yes   | Yes ^     |
| [Mesh Shader](#mesh-shader)                          | No    | Yes       | Yes     | No             | Yes   | No        |
| [`[unroll]`](#unroll]                                | Yes   | Yes       | Yes ^   | Yes            | No ^  | Limited + |
| Atomics                                              | Yes   | Yes       | Yes     | Yes            | Yes   | No +      |
| [Atomics on RWBuffer](#rwbuffer-atomics)             | Yes   | Yes       | Yes     | No             | Yes   | No +      |
| [Sampler Feedback](#sampler-feedback)                | No    | Yes       | No +    | No             | No    | Yes ^     |
| [RWByteAddressBuffer Atomic](#byte-address-atomic)   | No    | Yes ^     | Yes ^   | Yes            | Yes   | No +      |
| [Shader Execution Reordering](#ser)                  | No    | Yes ^     | Yes ^   | No             | No    | No        |
| [debugBreak](#debug-break)                           | No    | No        | Yes     | Yes            | No    | Yes       |
| [realtime clock](#realtime-clock)                    | No    | Yes ^     | Yes     | Yes            | No    | No        |

<a id="half"></a>

## Half Type

There appears to be a problem writing to a StructuredBuffer containing half on D3D12. D3D12 also appears to have problems doing calculations with half.

In order for half to work in CUDA, NVRTC must be able to include `cuda_fp16.h` and related files. Please read the [CUDA target documentation](cuda-target.md) for more details.

<a id="int8_t"></a>

## u/int8_t Type

Not currently supported in D3D11/D3D12 because not supported in HLSL/DXIL/DXBC.

Supported in Vulkan via the extensions `GL_EXT_shader_explicit_arithmetic_types` and `GL_EXT_shader_8bit_storage`.

<a id="int16_t"></a>

## u/int16_t Type

Requires SM6.2 which requires DXIL and therefore DXC and D3D12. For DXC this is discussed [here](https://github.com/Microsoft/DirectXShaderCompiler/wiki/16-Bit-Scalar-Types).

Supported in Vulkan via the extensions `GL_EXT_shader_explicit_arithmetic_types` and `GL_EXT_shader_16bit_storage`.

<a id="int64_t"></a>

## u/int64_t Type

Requires SM6.0 which requires DXIL for D3D12. Therefore not available with DXBC on D3D11 or D3D12.

<a id="int-matrix"></a>

## int matrix

Means can use matrix types containing integer types.

<a id="tex-get-dimensions"></a>

## tex.GetDimensions

tex.GetDimensions is the GetDimensions method on 'texture' objects. This is not supported on CUDA as CUDA has no equivalent functionality to get these values. GetDimensions work on Buffer resource types on CUDA.

<a id="sm6-wave"></a>

## SM6.0 Wave Intrinsics

CUDA has premliminary support for Wave Intrinsics, introduced in [PR #1352](https://github.com/shader-slang/slang/pull/1352). Slang synthesizes the 'WaveMask' based on program flow and the implied 'programmer view' of execution. This support is built on top of WaveMask intrinsics with Wave Intrinsics being replaced with WaveMask Intrinsic calls with Slang generating the code to calculate the appropriate WaveMasks.

Please read [PR #1352](https://github.com/shader-slang/slang/pull/1352) for a better description of the status.

<a id="sm6.5-wave"></a>

## SM6.5 Wave Intrinsics

SM6.5 Wave Intrinsics are supported, but requires a downstream DXC compiler that supports SM6.5. As it stands the DXC shipping with windows does not.

<a id="wave-mask"></a>

## WaveMask Intrinsics

In order to map better to the CUDA sync/mask model Slang supports 'WaveMask' intrinsics. They operate in broadly the same way as the Wave intrinsics, but require the programmer to specify the lanes that are involved. To write code that uses wave intrinsics across targets including CUDA, currently the WaveMask intrinsics must be used. For this to work, the masks passed to the WaveMask functions should exactly match the 'Active lanes' concept that HLSL uses, otherwise the result is undefined.

The WaveMask intrinsics are not part of HLSL and are only available on Slang.

<a id="wave-shuffle"></a>

## WaveShuffle

`WaveShuffle` and `WaveBroadcastLaneAt` are Slang specific intrinsic additions to expand the options available around `WaveReadLaneAt`.

To be clear this means they will not compile directly on 'standard' HLSL compilers such as `dxc`, but Slang HLSL _output_ (which will not contain these intrinsics) can (and typically is) compiled via dxc.

The difference between them can be summarized as follows

- WaveBroadcastLaneAt - laneId must be a compile time constant
- WaveReadLaneAt - laneId can be dynamic but _MUST_ be the same value across the Wave ie 'dynamically uniform' across the Wave
- WaveShuffle - laneId can be truly dynamic (NOTE! That it is not strictly truly available currently on all targets, specifically HLSL)

Other than the different restrictions on laneId they act identically to WaveReadLaneAt.

`WaveBroadcastLaneAt` and `WaveReadLaneAt` will work on all targets that support wave intrinsics, with the only current restriction being that on GLSL targets, only scalars and vectors are supported.

`WaveShuffle` will always work on CUDA/Vulkan.

On HLSL based targets currently `WaveShuffle` will be converted into `WaveReadLaneAt`. Strictly speaking this means it _requires_ the `laneId` to be `dynamically uniform` across the Wave. In practice some hardware supports the loosened usage, and others does not. In the future this may be fixed in Slang and/or HLSL to work across all hardware. For now if you use `WaveShuffle` on HLSL based targets it will be necessary to confirm that `WaveReadLaneAt` has the loosened behavior for all the hardware intended. If target hardware does not support the loosened restrictions it's behavior is undefined.

<a id="tesselation"></a>

## Tesselation

Although tesselation stages should work on D3D11 and D3D12 they are not tested within our test framework, and may have problems.

<a id="native-bindless"></a>

## Native Bindless

Bindless is possible on targets that support it - but is not the default behavior for those targets, and typically require significant effort in Slang code.

'Native Bindless' targets use a form of 'bindless' for all targets. On CUDA this requires the target to use 'texture object' style binding and for the device to have 'compute capability 3.0' or higher.

<a id="resource-bounds"></a>

## Resource bounds

For CUDA this is optional as can be controlled via the SLANG_CUDA_BOUNDARY_MODE macro in the `slang-cuda-prelude.h`. By default it's behavior is `cudaBoundaryModeZero`.

<a id="buffer-bounds"></a>

## Buffer Bounds

This is the feature when accessing outside of the bounds of a Buffer there is well defined behavior - on read returning all 0s, and on write, the write being ignored.

On CPU there is only bounds checking on debug compilation of C++ code. This will assert if the access is out of range.

On CUDA out of bounds accesses default to element 0 (!). The behavior can be controlled via the SLANG_CUDA_BOUND_CHECK macro in the `slang-cuda-prelude.h`. This behavior may seem a little strange - and it requires a buffer that has at least one member to not do something nasty. It is really a 'least worst' answer to a difficult problem and is better than out of range accesses or worse writes.

In Metal, accessing a buffer out of bounds is undefined behavior.

<a id="tex-array-sample-float"></a>

## TextureArray.Sample float

When using 'Sample' on a TextureArray, CUDA treats the array index parameter as an int, even though it is passed as a float.

<a id="separate-sampler"></a>

## Separate Sampler

This feature means that a multiple Samplers can be used with a Texture. In terms of the HLSL code this can be seen as the 'SamplerState' being a parameter passed to the 'Sample' method on a texture object.

On CUDA the SamplerState is ignored, because on this target a 'texture object' is the Texture and Sampler combination.

<a id="graphics-pipeline"></a>

## Graphics Pipeline

CPU and CUDA only currently support compute shaders.

<a id="ray-tracing-1.0"></a>

## Ray Tracing DXR 1.0

Vulkan does not support a local root signature, but there is the concept of a 'shader record'. In Slang a single constant buffer can be marked as a shader record with the `[[vk::shader_record]]` attribute, for example:

```
[[vk::shader_record]]
cbuffer ShaderRecord
{
	uint shaderRecordID;
}
```

In practice to write shader code that works across D3D12 and VK you should have a single constant buffer marked as 'shader record' for VK and then on D3D that constant buffer should be bound in the local root signature on D3D.

<a id="tex-load"></a>

## tex.Load

tex.Load is only supported on CUDA for Texture1D. Additionally CUDA only allows such access for linear memory, meaning the bound texture can also not have mip maps. Load _is_ allowed on RWTexture types of other dimensions including 1D on CUDA.

<a id="full-bool"></a>

## Full bool

Means fully featured bool support. CUDA has issues around bool because there isn't a vector bool type built in. Currently bool aliases to an int vector type.

On CPU there are some issues in so far as bool's size is not well defined in size an alignment. Most C++ compilers now use a byte to represent a bool. In the past it has been backed by an int on some compilers.

<a id="unroll"></a>

## `[unroll]`

The unroll attribute allows for unrolling `for` loops. At the moment the feature is dependent on downstream compiler support which is mixed. In the longer term the intention is for Slang to contain it's own loop unroller - and therefore not be dependent on the feature on downstream compilers.

On C++ this attribute becomes SLANG_UNROLL which is defined in the prelude. This can be predefined if there is a suitable mechanism, if there isn't a definition SLANG_UNROLL will be an empty definition.

On GLSL and VK targets loop unrolling uses the [GL_EXT_control_flow_attributes](https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_control_flow_attributes.txt) extension.

Metal Shading Language does not support loop unrolling.

Slang does have a cross target mechanism to [unroll loops](language-reference/06-statements.md), in the section `Compile-Time For Statement`.

<a id="rwbuffer-atomics"></a>

## Atomics on RWBuffer

For VK the GLSL output from Slang seems plausible, but VK binding fails in tests harness.

On CUDA RWBuffer becomes CUsurfObject, which is a 'texture' type and does not support atomics.

On the CPU atomics are not supported, but will be in the future.

<a id="sampler-feedback"></a>

## Sampler Feedback

The HLSL [sampler feedback feature](https://microsoft.github.io/DirectX-Specs/d3d/SamplerFeedback.html) is available for DirectX12. The features requires shader model 6.5 and therefore a version of [DXC](https://github.com/Microsoft/DirectXShaderCompiler) that supports that model or higher. The Shader Model 6.5 requirement also means only DXIL binary format is supported.

There doesn't not appear to be a similar feature available in Vulkan yet, but when it is available support should be added.

For CPU targets there is the IFeedbackTexture interface that requires an implementation for use. Slang does not currently include CPU implementations for texture types.

<a id="byte-address-atomic"></a>

## RWByteAddressBuffer Atomic

The additional supported methods on RWByteAddressBuffer are...

```
void RWByteAddressBuffer::InterlockedAddF32(uint byteAddress, float valueToAdd, out float originalValue);
void RWByteAddressBuffer::InterlockedAddF32(uint byteAddress, float valueToAdd);

void RWByteAddressBuffer::InterlockedAddI64(uint byteAddress, int64_t valueToAdd, out int64_t originalValue);
void RWByteAddressBuffer::InterlockedAddI64(uint byteAddress, int64_t valueToAdd);

void RWByteAddressBuffer::InterlockedCompareExchangeU64(uint byteAddress, uint64_t compareValue, uint64_t value, out uint64_t outOriginalValue);

uint64_t RWByteAddressBuffer::InterlockedExchangeU64(uint byteAddress, uint64_t value);

uint64_t RWByteAddressBuffer::InterlockedMaxU64(uint byteAddress, uint64_t value);
uint64_t RWByteAddressBuffer::InterlockedMinU64(uint byteAddress, uint64_t value);

uint64_t RWByteAddressBuffer::InterlockedAndU64(uint byteAddress, uint64_t value);
uint64_t RWByteAddressBuffer::InterlockedOrU64(uint byteAddress, uint64_t value);
uint64_t RWByteAddressBuffer::InterlockedXorU64(uint byteAddress, uint64_t value);
```

On HLSL based targets this functionality is achieved using [NVAPI](https://developer.nvidia.com/nvapi). Support for NVAPI is described
in the separate [NVAPI Support](nvapi-support.md) document.

On Vulkan, for float the [`GL_EXT_shader_atomic_float`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_atomic_float.html) extension is required. For int64 the [`GL_EXT_shader_atomic_int64`](https://raw.githubusercontent.com/KhronosGroup/GLSL/master/extensions/ext/GL_EXT_shader_atomic_int64.txt) extension is required.

CUDA requires SM6.0 or higher for int64 support.

<a id="mesh-shader"></a>

## Mesh Shader

There is preliminary [Mesh Shader support](https://github.com/shader-slang/slang/pull/2464).

<a id="ser"></a>

## Shader Execution Reordering

More information about [Shader Execution Reordering](shader-execution-reordering.md).

Currently support is available in D3D12 via NVAPI, and for Vulkan via the [GL_NV_shader_invocation_reorder](https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GLSL_NV_shader_invocation_reorder.txt) extension.

<a id="debug-break"></a>

## Debug Break

Slang has preliminary support for `debugBreak()` intrinsic. With the appropriate tooling, when `debugBreak` is hit it will cause execution to halt and display in the attached debugger.

This is not supported on HLSL, GLSL, SPIR-V or Metal backends. Note that on some targets if there isn't an appropriate debugging environment the debugBreak might cause execution to fail or potentially it is ignored.

On C++ targets debugBreak is implemented using SLANG_BREAKPOINT defined in "slang-cpp-prelude.h". If there isn't a suitable intrinsic, this will default to attempting to write to `nullptr` leading to a crash.

Some additional details:

- If [slang-llvm](cpu-target.md#slang-llvm) is being used as the downstream compiler (as is typical with `host-callable`), it will crash into the debugger, but may not produce a usable stack trace.
- For "normal" C++ downstream compilers such as Clang/Gcc/Visual Studio, to break into readable source code, debug information is typically necessary. Disabling optimizations may be useful to break on the appropriate specific line, and have variables inspectable.

<a id="realtime-clock"></a>

## Realtime Clock

Realtime clock support is available via the API

```
// Get low 32 bits of realtime clock
uint getRealtimeClockLow();
// Get 64 bit realtime clock, with low bits in .x and high bits in .y
uint2 getRealtimeClock();
```

On D3D this is supported through NVAPI via `NvGetSpecial`.

On Vulkan this is supported via [VK_KHR_shader_clock extension](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_clock.html)

On CUDA this is supported via [clock](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#time-function).

Currently this is not supported on CPU, although this will potentially be added in the future.
