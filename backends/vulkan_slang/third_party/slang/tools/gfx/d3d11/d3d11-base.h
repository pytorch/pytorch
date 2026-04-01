// d3d11-base.h
// Shared header file for D3D11 implementation
#pragma once

#include "../d3d/d3d-swapchain.h"
#include "../d3d/d3d-util.h"
#include "../flag-combiner.h"
#include "../immediate-renderer-base.h"
#include "../mutable-shader-object.h"
#include "../nvapi/nvapi-util.h"
#include "core/slang-basic.h"
#include "core/slang-blob.h"
#include "slang-com-ptr.h"

#pragma push_macro("WIN32_LEAN_AND_MEAN")
#pragma push_macro("NOMINMAX")
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef NOMINMAX
#define NOMINMAX
#include <windows.h>
#pragma pop_macro("NOMINMAX")
#pragma pop_macro("WIN32_LEAN_AND_MEAN")

#include <d3d11_2.h>
#include <d3dcompiler.h>

#ifdef GFX_NVAPI
// NVAPI integration is described here
// https://developer.nvidia.com/unlocking-gpu-intrinsics-hlsl

#include "../nvapi/nvapi-include.h"
#endif

// We will use the C standard library just for printing error messages.
#include <stdio.h>

#ifdef _MSC_VER
#include <stddef.h>
#if (_MSC_VER < 1900)
#define snprintf sprintf_s
#endif
#endif

namespace gfx
{
namespace d3d11
{
class DeviceImpl;
class ShaderProgramImpl;
class BufferResourceImpl;
class TextureResourceImpl;
class SamplerStateImpl;
class ResourceViewImpl;
class ShaderResourceViewImpl;
class UnorderedAccessViewImpl;
class DepthStencilViewImpl;
class RenderTargetViewImpl;
class FramebufferLayoutImpl;
class FramebufferImpl;
class SwapchainImpl;
class InputLayoutImpl;
class QueryPoolImpl;
class PipelineStateImpl;
class GraphicsPipelineStateImpl;
class ComputePipelineStateImpl;
class ShaderObjectLayoutImpl;
class RootShaderObjectLayoutImpl;
class ShaderObjectImpl;
class MutableShaderObjectImpl;
class RootShaderObjectImpl;
} // namespace d3d11
} // namespace gfx
