// metal-command-encoder.h
#pragma once

#include "metal-base.h"
#include "metal-pipeline-state.h"
#include "metal-render-pass.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class PipelineCommandEncoder : public ComObject
{
public:
    CommandBufferImpl* m_commandBuffer;
    MTL::CommandBuffer* m_metalCommandBuffer;
    RefPtr<PipelineStateImpl> m_currentPipeline;

    void init(CommandBufferImpl* commandBuffer);
    void endEncodingImpl();

    Result setPipelineStateImpl(IPipelineState* state, IShaderObject** outRootObject);
};

class ResourceCommandEncoder : public IResourceCommandEncoder, public PipelineCommandEncoder
{
public:
    virtual void* getInterface(SlangUUID const& guid)
    {
        if (guid == GfxGUID::IID_IResourceCommandEncoder || guid == ISlangUnknown::getTypeGuid())
            return this;
        return nullptr;
    }
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) override
    {
        if (auto ptr = getInterface(uuid))
        {
            *outObject = ptr;
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 1; }
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 1; }

    virtual SLANG_NO_THROW void SLANG_MCALL endEncoding() override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    writeTimestamp(IQueryPool* queryPool, GfxIndex index) override;

    virtual SLANG_NO_THROW void SLANG_MCALL copyBuffer(
        IBufferResource* dst,
        Offset dstOffset,
        IBufferResource* src,
        Offset srcOffset,
        Size size) override;

    virtual SLANG_NO_THROW void SLANG_MCALL copyTexture(
        ITextureResource* dst,
        ResourceState dstState,
        SubresourceRange dstSubresource,
        ITextureResource::Offset3D dstOffset,
        ITextureResource* src,
        ResourceState srcState,
        SubresourceRange srcSubresource,
        ITextureResource::Offset3D srcOffset,
        ITextureResource::Extents extent) override;

    virtual SLANG_NO_THROW void SLANG_MCALL copyTextureToBuffer(
        IBufferResource* dst,
        Offset dstOffset,
        Size dstSize,
        Size dstRowStride,
        ITextureResource* src,
        ResourceState srcState,
        SubresourceRange srcSubresource,
        ITextureResource::Offset3D srcOffset,
        ITextureResource::Extents extent) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    uploadBufferData(IBufferResource* buffer, Offset offset, Size size, void* data) override;

    virtual SLANG_NO_THROW void SLANG_MCALL uploadTextureData(
        ITextureResource* dst,
        SubresourceRange subResourceRange,
        ITextureResource::Offset3D offset,
        ITextureResource::Extents extend,
        ITextureResource::SubresourceData* subResourceData,
        GfxCount subResourceDataCount) override;

    virtual SLANG_NO_THROW void SLANG_MCALL bufferBarrier(
        GfxCount count,
        IBufferResource* const* buffers,
        ResourceState src,
        ResourceState dst) override;

    virtual SLANG_NO_THROW void SLANG_MCALL textureBarrier(
        GfxCount count,
        ITextureResource* const* textures,
        ResourceState src,
        ResourceState dst) override;

    virtual SLANG_NO_THROW void SLANG_MCALL textureSubresourceBarrier(
        ITextureResource* texture,
        SubresourceRange subresourceRange,
        ResourceState src,
        ResourceState dst) override;

    void _clearColorImage(TextureResourceViewImpl* viewImpl, ClearValue* clearValue);

    void _clearDepthImage(
        TextureResourceViewImpl* viewImpl,
        ClearValue* clearValue,
        ClearResourceViewFlags::Enum flags);

    virtual SLANG_NO_THROW void SLANG_MCALL clearResourceView(
        IResourceView* view,
        ClearValue* clearValue,
        ClearResourceViewFlags::Enum flags) override;

    virtual SLANG_NO_THROW void SLANG_MCALL resolveResource(
        ITextureResource* source,
        ResourceState sourceState,
        SubresourceRange sourceRange,
        ITextureResource* dest,
        ResourceState destState,
        SubresourceRange destRange) override;

    virtual SLANG_NO_THROW void SLANG_MCALL resolveQuery(
        IQueryPool* queryPool,
        GfxIndex index,
        GfxCount count,
        IBufferResource* buffer,
        Offset offset) override;


    virtual SLANG_NO_THROW void SLANG_MCALL
    beginDebugEvent(const char* name, float rgbColor[3]) override;
    virtual SLANG_NO_THROW void SLANG_MCALL endDebugEvent() override;
};

class RenderCommandEncoder : public IRenderCommandEncoder, public ResourceCommandEncoder
{
    SLANG_GFX_FORWARD_RESOURCE_COMMAND_ENCODER_IMPL(ResourceCommandEncoder)
    virtual void* getInterface(SlangUUID const& uuid) override
    {
        if (uuid == GfxGUID::IID_IResourceCommandEncoder ||
            uuid == GfxGUID::IID_IRenderCommandEncoder || uuid == ISlangUnknown::getTypeGuid())
        {
            return this;
        }
        return nullptr;
    }

public:
    RefPtr<RenderPassLayoutImpl> m_renderPassLayout;
    RefPtr<FramebufferImpl> m_framebuffer;
    NS::SharedPtr<MTL::RenderPassDescriptor> m_renderPassDesc;

    ShortList<MTL::Viewport, 16> m_viewports;
    ShortList<MTL::ScissorRect, 16> m_scissorRects;
    MTL::PrimitiveType m_primitiveType = MTL::PrimitiveTypeTriangle;

    ShortList<MTL::Buffer*, 16> m_vertexBuffers;
    ShortList<NS::UInteger, 16> m_vertexBufferOffsets;

    MTL::Buffer* m_indexBuffer = nullptr;
    NS::UInteger m_indexBufferOffset = 0;
    MTL::IndexType m_indexBufferType = MTL::IndexTypeUInt16;

    uint32_t m_stencilReferenceValue = 0;

public:
    void beginPass(IRenderPassLayout* renderPass, IFramebuffer* framebuffer);

    virtual SLANG_NO_THROW void SLANG_MCALL endEncoding() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipeline(IPipelineState* pipelineState, IShaderObject** outRootObject) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipelineWithRootObject(IPipelineState* pipelineState, IShaderObject* rootObject) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setViewports(GfxCount count, const Viewport* viewports) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setScissorRects(GfxCount count, const ScissorRect* rects) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setPrimitiveTopology(PrimitiveTopology topology) override;

    virtual SLANG_NO_THROW void SLANG_MCALL setVertexBuffers(
        GfxIndex startSlot,
        GfxCount slotCount,
        IBufferResource* const* buffers,
        const Offset* offsets) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setIndexBuffer(IBufferResource* buffer, Format indexFormat, Offset offset = 0) override;

    virtual SLANG_NO_THROW void SLANG_MCALL setStencilReference(uint32_t referenceValue) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL setSamplePositions(
        GfxCount samplesPerPixel,
        GfxCount pixelCount,
        const SamplePosition* samplePositions) override;

    Result prepareDraw(MTL::RenderCommandEncoder*& encoder);

    virtual SLANG_NO_THROW Result SLANG_MCALL
    draw(GfxCount vertexCount, GfxIndex startVertex = 0) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    drawIndexed(GfxCount indexCount, GfxIndex startIndex = 0, GfxIndex baseVertex = 0) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL drawIndirect(
        GfxCount maxDrawCount,
        IBufferResource* argBuffer,
        Offset argOffset,
        IBufferResource* countBuffer,
        Offset countOffset) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL drawIndexedIndirect(
        GfxCount maxDrawCount,
        IBufferResource* argBuffer,
        Offset argOffset,
        IBufferResource* countBuffer,
        Offset countOffset) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL drawInstanced(
        GfxCount vertexCount,
        GfxCount instanceCount,
        GfxIndex startVertex,
        GfxIndex startInstanceLocation) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL drawIndexedInstanced(
        GfxCount indexCount,
        GfxCount instanceCount,
        GfxIndex startIndexLocation,
        GfxIndex baseVertexLocation,
        GfxIndex startInstanceLocation) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL drawMeshTasks(int x, int y, int z) override;
};

class ComputeCommandEncoder : public IComputeCommandEncoder, public ResourceCommandEncoder
{
public:
    SLANG_GFX_FORWARD_RESOURCE_COMMAND_ENCODER_IMPL(ResourceCommandEncoder)
    virtual void* getInterface(SlangUUID const& uuid) override
    {
        if (uuid == GfxGUID::IID_IResourceCommandEncoder ||
            uuid == GfxGUID::IID_IComputeCommandEncoder || uuid == ISlangUnknown::getTypeGuid())
        {
            return this;
        }
        return nullptr;
    }

    virtual SLANG_NO_THROW void SLANG_MCALL endEncoding() override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipeline(IPipelineState* pipelineState, IShaderObject** outRootObject) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipelineWithRootObject(IPipelineState* pipelineState, IShaderObject* rootObject) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL dispatchCompute(int x, int y, int z) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    dispatchComputeIndirect(IBufferResource* argBuffer, Offset offset) override;
};

class RayTracingCommandEncoder : public IRayTracingCommandEncoder, public ResourceCommandEncoder
{
public:
    SLANG_GFX_FORWARD_RESOURCE_COMMAND_ENCODER_IMPL(ResourceCommandEncoder)
    virtual void* getInterface(SlangUUID const& uuid) override
    {
        if (uuid == GfxGUID::IID_IResourceCommandEncoder ||
            uuid == GfxGUID::IID_IRayTracingCommandEncoder || uuid == ISlangUnknown::getTypeGuid())
        {
            return this;
        }
        return nullptr;
    }

public:
    void _memoryBarrier(
        int count,
        IAccelerationStructure* const* structures,
        AccessFlag srcAccess,
        AccessFlag destAccess);

    void _queryAccelerationStructureProperties(
        GfxCount accelerationStructureCount,
        IAccelerationStructure* const* accelerationStructures,
        GfxCount queryCount,
        AccelerationStructureQueryDesc* queryDescs);

    virtual SLANG_NO_THROW void SLANG_MCALL buildAccelerationStructure(
        const IAccelerationStructure::BuildDesc& desc,
        GfxCount propertyQueryCount,
        AccelerationStructureQueryDesc* queryDescs) override;

    virtual SLANG_NO_THROW void SLANG_MCALL copyAccelerationStructure(
        IAccelerationStructure* dest,
        IAccelerationStructure* src,
        AccelerationStructureCopyMode mode) override;

    virtual SLANG_NO_THROW void SLANG_MCALL queryAccelerationStructureProperties(
        GfxCount accelerationStructureCount,
        IAccelerationStructure* const* accelerationStructures,
        GfxCount queryCount,
        AccelerationStructureQueryDesc* queryDescs) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    serializeAccelerationStructure(DeviceAddress dest, IAccelerationStructure* source) override;

    virtual SLANG_NO_THROW void SLANG_MCALL
    deserializeAccelerationStructure(IAccelerationStructure* dest, DeviceAddress source) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipeline(IPipelineState* pipeline, IShaderObject** outRootObject) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    bindPipelineWithRootObject(IPipelineState* pipelineState, IShaderObject* rootObject) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL dispatchRays(
        GfxIndex raygenShaderIndex,
        IShaderTable* shaderTable,
        GfxCount width,
        GfxCount height,
        GfxCount depth) override;

    virtual SLANG_NO_THROW void SLANG_MCALL endEncoding() override;
};

} // namespace metal
} // namespace gfx
