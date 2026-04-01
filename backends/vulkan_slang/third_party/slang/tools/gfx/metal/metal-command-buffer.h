// metal-command-buffer.h
#pragma once

#include "../simple-transient-resource-heap.h"
#include "metal-base.h"
#include "metal-command-encoder.h"
#include "metal-shader-object.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class CommandBufferImpl : public ICommandBuffer, public ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ICommandBuffer* getInterface(const Guid& guid);

public:
    RefPtr<DeviceImpl> m_device;
    NS::SharedPtr<MTL::CommandBuffer> m_commandBuffer;
    RootShaderObjectImpl m_rootObject;
    // RefPtr<MutableRootShaderObjectImpl> m_mutableRootShaderObject;

    RefPtr<ResourceCommandEncoder> m_resourceCommandEncoder = nullptr;
    RefPtr<ComputeCommandEncoder> m_computeCommandEncoder = nullptr;
    RefPtr<RenderCommandEncoder> m_renderCommandEncoder = nullptr;
    RefPtr<RayTracingCommandEncoder> m_rayTracingCommandEncoder = nullptr;

    NS::SharedPtr<MTL::RenderCommandEncoder> m_metalRenderCommandEncoder;
    NS::SharedPtr<MTL::ComputeCommandEncoder> m_metalComputeCommandEncoder;
    NS::SharedPtr<MTL::BlitCommandEncoder> m_metalBlitCommandEncoder;

    // Command buffers are deallocated by its command pool,
    // so no need to free individually.
    ~CommandBufferImpl() = default;

    Result init(DeviceImpl* device, TransientResourceHeapImpl* transientHeap);

    void beginCommandBuffer();

    MTL::RenderCommandEncoder* getMetalRenderCommandEncoder(
        MTL::RenderPassDescriptor* renderPassDesc);
    MTL::ComputeCommandEncoder* getMetalComputeCommandEncoder();
    MTL::BlitCommandEncoder* getMetalBlitCommandEncoder();
    void endMetalCommandEncoder();

public:
    virtual SLANG_NO_THROW void SLANG_MCALL encodeRenderCommands(
        IRenderPassLayout* renderPass,
        IFramebuffer* framebuffer,
        IRenderCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeComputeCommands(IComputeCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeResourceCommands(IResourceCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL
    encodeRayTracingCommands(IRayTracingCommandEncoder** outEncoder) override;
    virtual SLANG_NO_THROW void SLANG_MCALL close() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getNativeHandle(InteropHandle* outHandle) override;
};

} // namespace metal
} // namespace gfx
