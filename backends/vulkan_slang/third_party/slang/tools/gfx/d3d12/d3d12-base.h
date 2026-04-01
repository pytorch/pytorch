// d3d12-base.h
// Shared header file for D3D12 implementation
#pragma once

#include "../command-encoder-com-forward.h"
#include "../d3d/d3d-swapchain.h"
#include "../mutable-shader-object.h"
#include "../renderer-shared.h"
#include "../simple-render-pass-layout.h"
#include "../transient-resource-heap-base.h"
#include "core/slang-basic.h"
#include "core/slang-blob.h"
#include "core/slang-chunked-list.h"
#include "d3d12-descriptor-heap.h"
#include "d3d12-posix-synchapi.h"
#include "d3d12-resource.h"

#pragma push_macro("WIN32_LEAN_AND_MEAN")
#pragma push_macro("NOMINMAX")
#pragma push_macro("_CRT_SECURE_NO_WARNINGS")
#undef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#undef NOMINMAX
#define NOMINMAX
#undef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#pragma pop_macro("_CRT_SECURE_NO_WARNINGS")
#pragma pop_macro("NOMINMAX")
#pragma pop_macro("WIN32_LEAN_AND_MEAN")

#include <d3d12.h>
#include <dxgi1_4.h>

#ifndef __ID3D12GraphicsCommandList1_FWD_DEFINED__
// If can't find a definition of CommandList1, just use an empty definition
struct ID3D12GraphicsCommandList1
{
};
#endif

namespace gfx
{
namespace d3d12
{
class DeviceImpl;
class BufferResourceImpl;
class TextureResourceImpl;
class CommandBufferImpl;
class PipelineCommandEncoder;
class ResourceCommandEncoderImpl;
class ComputeCommandEncoderImpl;
class RenderCommandEncoderImpl;
class CommandQueueImpl;
class FenceImpl;
class FramebufferLayoutImpl;
class FramebufferImpl;
class QueryPoolImpl;
class PlainBufferProxyQueryPoolImpl;
class PipelineStateImpl;
class RenderPassLayoutImpl;
class ResourceViewInternalImpl;
class ResourceViewImpl;
class AccelerationStructureImpl;
class SamplerStateImpl;
class ShaderObjectImpl;
class RootShaderObjectImpl;
class MutableRootShaderObjectImpl;
class ShaderObjectLayoutImpl;
class RootShaderObjectLayoutImpl;
class ShaderProgramImpl;
class ShaderTableImpl;
class SwapChainImpl;
class TransientResourceHeapImpl;
class InputLayoutImpl;

#if SLANG_GFX_HAS_DXR_SUPPORT
class RayTracingCommandEncoderImpl;
class RayTracingPipelineStateImpl;
#endif
} // namespace d3d12
} // namespace gfx
