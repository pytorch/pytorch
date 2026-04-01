// vk-command-buffer.h
#pragma once

#include "vk-base.h"
#include "vk-command-encoder.h"
#include "vk-shader-object.h"
#include "vk-transient-heap.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class CommandBufferImpl : public ICommandBuffer, public ComObject
{
public:
    // There are a pair of cyclic references between a `TransientResourceHeap` and
    // a `CommandBuffer` created from the heap. We need to break the cycle when
    // the public reference count of a command buffer drops to 0.
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ICommandBuffer* getInterface(const Guid& guid);
    virtual void comFree() override;

public:
    VkCommandBuffer m_commandBuffer;
    VkCommandBuffer m_preCommandBuffer = VK_NULL_HANDLE;
    VkCommandPool m_pool;
    DeviceImpl* m_renderer;
    BreakableReference<TransientResourceHeapImpl> m_transientHeap;
    bool m_isPreCommandBufferEmpty = true;
    RootShaderObjectImpl m_rootObject;
    RefPtr<MutableRootShaderObjectImpl> m_mutableRootShaderObject;

    RefPtr<ResourceCommandEncoder> m_resourceCommandEncoder;
    RefPtr<ComputeCommandEncoder> m_computeCommandEncoder;
    RefPtr<RenderCommandEncoder> m_renderCommandEncoder;
    RefPtr<RayTracingCommandEncoder> m_rayTracingCommandEncoder;

    // Command buffers are deallocated by its command pool,
    // so no need to free individually.
    ~CommandBufferImpl() = default;

    Result init(DeviceImpl* renderer, VkCommandPool pool, TransientResourceHeapImpl* transientHeap);

    void beginCommandBuffer();

    Result createPreCommandBuffer();

    VkCommandBuffer getPreCommandBuffer();

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

} // namespace vk
} // namespace gfx
