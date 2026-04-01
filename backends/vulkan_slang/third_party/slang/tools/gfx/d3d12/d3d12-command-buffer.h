// d3d12-command-buffer.h
#pragma once

#include "d3d12-base.h"
#include "d3d12-command-encoder.h"
#include "d3d12-shader-object.h"

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

using namespace Slang;

class CommandBufferImpl : public ICommandBufferD3D12, public ComObject
{
public:
    // There are a pair of cyclic references between a `TransientResourceHeap` and
    // a `CommandBuffer` created from the heap. We need to break the cycle upon
    // the public reference count of a command buffer dropping to 0.
    SLANG_COM_OBJECT_IUNKNOWN_ALL

    ICommandBufferD3D12* getInterface(const Guid& guid);
    virtual void comFree() override { m_transientHeap.breakStrongReference(); }

    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* handle) override;

public:
    ComPtr<ID3D12GraphicsCommandList> m_cmdList;
    ComPtr<ID3D12GraphicsCommandList1> m_cmdList1;
    ComPtr<ID3D12GraphicsCommandList4> m_cmdList4;
    ComPtr<ID3D12GraphicsCommandList6> m_cmdList6;

    BreakableReference<TransientResourceHeapImpl> m_transientHeap;
    // Weak reference is fine here since `m_transientHeap` already holds strong reference to
    // device.
    DeviceImpl* m_renderer;
    RootShaderObjectImpl m_rootShaderObject;
    RefPtr<MutableRootShaderObjectImpl> m_mutableRootShaderObject;
    bool m_descriptorHeapsBound = false;

    void bindDescriptorHeaps();

    virtual SLANG_NO_THROW void SLANG_MCALL invalidateDescriptorHeapBinding() override
    {
        m_descriptorHeapsBound = false;
    }
    virtual SLANG_NO_THROW void SLANG_MCALL ensureInternalDescriptorHeapsBound() override
    {
        bindDescriptorHeaps();
    }

    void reinit();

    void init(
        DeviceImpl* renderer,
        ID3D12GraphicsCommandList* d3dCommandList,
        TransientResourceHeapImpl* transientHeap);

    ResourceCommandEncoderImpl m_resourceCommandEncoder;

    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeResourceCommands(IResourceCommandEncoder** outEncoder) override;

    RenderCommandEncoderImpl m_renderCommandEncoder;
    virtual SLANG_NO_THROW void SLANG_MCALL encodeRenderCommands(
        IRenderPassLayout* renderPass,
        IFramebuffer* framebuffer,
        IRenderCommandEncoder** outEncoder) override;

    ComputeCommandEncoderImpl m_computeCommandEncoder;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeComputeCommands(IComputeCommandEncoder** outEncoder) override;

#if SLANG_GFX_HAS_DXR_SUPPORT
    RayTracingCommandEncoderImpl m_rayTracingCommandEncoder;
#endif
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL close() override;
};

} // namespace d3d12
} // namespace gfx
