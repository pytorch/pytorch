// immediate-renderer-base.h
#pragma once

// Provides shared implementation of public API objects for targets with
// an immediate mode execution context.

#include "renderer-shared.h"

namespace gfx
{

enum class MapFlavor
{
    Unknown, ///< Unknown mapping type
    HostRead,
    HostWrite,
    WriteDiscard,
};

class ImmediateCommandQueueBase : public ICommandQueue, public Slang::ComObject
{
public:
    // Immediate device also holds a strong reference to an instance of `ImmediateCommandQueue`,
    // forming a cyclic reference. Therefore we need a free-op here to break the cycle when
    // the public reference count of the queue drops to 0.
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ICommandQueue* getInterface(const Slang::Guid& guid)
    {
        if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ICommandQueue)
            return static_cast<ICommandQueue*>(this);
        return nullptr;
    }
    virtual void comFree() override { breakStrongReferenceToDevice(); }

public:
    BreakableReference<RendererBase> m_renderer;
    void breakStrongReferenceToDevice() { m_renderer.breakStrongReference(); }
    void establishStrongReferenceToDevice() { m_renderer.establishStrongReference(); }
};

struct CommandBufferInfo
{
    bool hasWriteTimestamps;
};

class ImmediateRendererBase : public RendererBase
{
public:
    // Immediate commands to be implemented by each target.
    virtual Result createRootShaderObject(
        IShaderProgram* program,
        ShaderObjectBase** outObject) = 0;
    virtual void bindRootShaderObject(IShaderObject* rootObject) = 0;
    virtual void setPipelineState(IPipelineState* state) = 0;
    virtual void setFramebuffer(IFramebuffer* frameBuffer) = 0;
    virtual void clearFrame(uint32_t colorBufferMask, bool clearDepth, bool clearStencil) = 0;
    virtual void setViewports(GfxCount count, const Viewport* viewports) = 0;
    virtual void setScissorRects(GfxCount count, const ScissorRect* scissors) = 0;
    virtual void setPrimitiveTopology(PrimitiveTopology topology) = 0;
    virtual void setVertexBuffers(
        GfxIndex startSlot,
        GfxCount slotCount,
        IBufferResource* const* buffers,
        const Offset* offsets) = 0;
    virtual void setIndexBuffer(IBufferResource* buffer, Format indexFormat, Offset offset = 0) = 0;
    virtual void draw(GfxCount vertexCount, GfxIndex startVertex = 0) = 0;
    virtual void drawIndexed(
        GfxCount indexCount,
        GfxIndex startIndex = 0,
        GfxIndex baseVertex = 0) = 0;
    virtual void drawInstanced(
        GfxCount vertexCount,
        GfxCount instanceCount,
        GfxIndex startVertex,
        GfxIndex startInstanceLocation) = 0;
    virtual void drawIndexedInstanced(
        GfxCount indexCount,
        GfxCount instanceCount,
        GfxIndex startIndexLocation,
        GfxIndex baseVertexLocation,
        GfxIndex startInstanceLocation) = 0;
    virtual void setStencilReference(uint32_t referenceValue) = 0;
    virtual void dispatchCompute(int x, int y, int z) = 0;
    virtual void copyBuffer(
        IBufferResource* dst,
        Offset dstOffset,
        IBufferResource* src,
        Offset srcOffset,
        Size size) = 0;
    virtual void submitGpuWork() = 0;
    virtual void waitForGpu() = 0;
    virtual void* map(IBufferResource* buffer, MapFlavor flavor) = 0;
    virtual void unmap(IBufferResource* buffer, size_t offsetWritten, size_t sizeWritten) = 0;
    virtual void writeTimestamp(IQueryPool* pool, GfxIndex index) = 0;
    virtual void beginCommandBuffer(const CommandBufferInfo&) {}
    virtual void endCommandBuffer(const CommandBufferInfo&) {}

public:
    Slang::RefPtr<ImmediateCommandQueueBase> m_queue;
    uint32_t m_queueCreateCount = 0;

    ImmediateRendererBase();

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandQueue(const ICommandQueue::Desc& desc, ICommandQueue** outQueue) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTransientResourceHeap(
        const ITransientResourceHeap::Desc& desc,
        ITransientResourceHeap** outHeap) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createRenderPassLayout(
        const IRenderPassLayout::Desc& desc,
        IRenderPassLayout** outRenderPassLayout) override;

    void uploadBufferData(IBufferResource* dst, Offset offset, Size size, void* data);

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL readBufferResource(
        IBufferResource* buffer,
        Offset offset,
        Size size,
        ISlangBlob** outBlob) override;
};

class ImmediateComputeDeviceBase : public ImmediateRendererBase
{
public:
    // Provide empty implementation for devices without graphics support.
    virtual void setFramebuffer(IFramebuffer* frameBuffer) override { SLANG_UNUSED(frameBuffer); }
    virtual void clearFrame(uint32_t colorBufferMask, bool clearDepth, bool clearStencil) override
    {
        SLANG_UNUSED(colorBufferMask);
        SLANG_UNUSED(clearDepth);
        SLANG_UNUSED(clearStencil);
    }
    virtual void setViewports(GfxCount count, const Viewport* viewports) override
    {
        SLANG_UNUSED(count);
        SLANG_UNUSED(viewports);
    }
    virtual void setScissorRects(GfxCount count, const ScissorRect* scissors) override
    {
        SLANG_UNUSED(count);
        SLANG_UNUSED(scissors);
    }
    virtual void setPrimitiveTopology(PrimitiveTopology topology) override
    {
        SLANG_UNUSED(topology);
    }
    virtual void setVertexBuffers(
        GfxIndex startSlot,
        GfxCount slotCount,
        IBufferResource* const* buffers,
        const Offset* offsets) override
    {
        SLANG_UNUSED(startSlot);
        SLANG_UNUSED(slotCount);
        SLANG_UNUSED(buffers);
        SLANG_UNUSED(offsets);
    }
    virtual void setIndexBuffer(IBufferResource* buffer, Format indexFormat, Offset offset = 0)
        override
    {
        SLANG_UNUSED(buffer);
        SLANG_UNUSED(indexFormat);
        SLANG_UNUSED(offset);
    }
    virtual void draw(GfxCount vertexCount, GfxIndex startVertex = 0) override
    {
        SLANG_UNUSED(vertexCount);
        SLANG_UNUSED(startVertex);
    }
    virtual void drawIndexed(GfxCount indexCount, GfxIndex startIndex = 0, GfxIndex baseVertex = 0)
        override
    {
        SLANG_UNUSED(indexCount);
        SLANG_UNUSED(startIndex);
        SLANG_UNUSED(baseVertex);
    }
    virtual void drawInstanced(
        GfxCount vertexCount,
        GfxCount instanceCount,
        GfxIndex startVertex,
        GfxIndex startInstanceLocation) override
    {
        SLANG_UNUSED(vertexCount);
        SLANG_UNUSED(instanceCount);
        SLANG_UNUSED(startVertex);
        SLANG_UNUSED(startInstanceLocation);
    }

    virtual void drawIndexedInstanced(
        GfxCount indexCount,
        GfxCount instanceCount,
        GfxIndex startIndexLocation,
        GfxIndex baseVertexLocation,
        GfxIndex startInstanceLocation) override
    {
        SLANG_UNUSED(indexCount);
        SLANG_UNUSED(instanceCount);
        SLANG_UNUSED(startIndexLocation);
        SLANG_UNUSED(baseVertexLocation);
        SLANG_UNUSED(startInstanceLocation);
    }
    virtual void setStencilReference(uint32_t referenceValue) override
    {
        SLANG_UNUSED(referenceValue);
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL createSwapchain(
        const ISwapchain::Desc& desc,
        WindowHandle window,
        ISwapchain** outSwapchain) override
    {
        SLANG_UNUSED(desc);
        SLANG_UNUSED(window);
        SLANG_UNUSED(outSwapchain);
        return SLANG_FAIL;
    }
    virtual SLANG_NO_THROW Result SLANG_MCALL createFramebufferLayout(
        const IFramebufferLayout::Desc& desc,
        IFramebufferLayout** outLayout) override
    {
        SLANG_UNUSED(desc);
        SLANG_UNUSED(outLayout);
        return SLANG_FAIL;
    }
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createFramebuffer(const IFramebuffer::Desc& desc, IFramebuffer** outFramebuffer) override
    {
        SLANG_UNUSED(desc);
        SLANG_UNUSED(outFramebuffer);
        return SLANG_FAIL;
    }
    virtual SLANG_NO_THROW Result SLANG_MCALL createRenderPassLayout(
        const IRenderPassLayout::Desc& desc,
        IRenderPassLayout** outRenderPassLayout) override
    {
        SLANG_UNUSED(desc);
        SLANG_UNUSED(outRenderPassLayout);
        return SLANG_FAIL;
    }

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createInputLayout(IInputLayout::Desc const& desc, IInputLayout** outLayout) override
    {
        SLANG_UNUSED(desc);
        SLANG_UNUSED(outLayout);
        return SLANG_E_NOT_AVAILABLE;
    }
    virtual SLANG_NO_THROW Result SLANG_MCALL createGraphicsPipelineState(
        const GraphicsPipelineStateDesc& desc,
        IPipelineState** outState) override
    {
        SLANG_UNUSED(desc);
        SLANG_UNUSED(outState);
        return SLANG_E_NOT_AVAILABLE;
    }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL readTextureResource(
        ITextureResource* texture,
        ResourceState state,
        ISlangBlob** outBlob,
        Size* outRowPitch,
        Size* outPixelSize) override
    {
        SLANG_UNUSED(texture);
        SLANG_UNUSED(outBlob);
        SLANG_UNUSED(outRowPitch);
        SLANG_UNUSED(outPixelSize);

        return SLANG_E_NOT_AVAILABLE;
    }
};
} // namespace gfx
