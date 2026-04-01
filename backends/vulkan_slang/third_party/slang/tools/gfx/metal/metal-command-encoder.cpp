// metal-command-encoder.cpp
#include "metal-command-encoder.h"

#include "metal-buffer.h"
#include "metal-command-buffer.h"
#include "metal-helper-functions.h"
#include "metal-query.h"
#include "metal-render-pass.h"
#include "metal-resource-views.h"
#include "metal-shader-object.h"
#include "metal-shader-program.h"
#include "metal-shader-table.h"
#include "metal-texture.h"
#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

void PipelineCommandEncoder::init(CommandBufferImpl* commandBuffer)
{
    m_commandBuffer = commandBuffer;
    m_metalCommandBuffer = m_commandBuffer->m_commandBuffer.get();
}

void PipelineCommandEncoder::endEncodingImpl()
{
    m_commandBuffer->endMetalCommandEncoder();
}

Result PipelineCommandEncoder::setPipelineStateImpl(
    IPipelineState* state,
    IShaderObject** outRootObject)
{
    m_currentPipeline = static_cast<PipelineStateImpl*>(state);
    // m_commandBuffer->m_mutableRootShaderObject = nullptr;
    SLANG_RETURN_ON_FAIL(m_commandBuffer->m_rootObject.init(
        m_commandBuffer->m_device,
        m_currentPipeline->getProgram<ShaderProgramImpl>()->m_rootObjectLayout));
    *outRootObject = &m_commandBuffer->m_rootObject;
    return SLANG_OK;
}

void ResourceCommandEncoder::endEncoding()
{
    PipelineCommandEncoder::endEncodingImpl();
}

void ResourceCommandEncoder::writeTimestamp(IQueryPool* queryPool, GfxIndex index)
{
    auto encoder = m_commandBuffer->getMetalBlitCommandEncoder();
    encoder->sampleCountersInBuffer(
        static_cast<QueryPoolImpl*>(queryPool)->m_counterSampleBuffer.get(),
        index,
        true);
}

void ResourceCommandEncoder::copyBuffer(
    IBufferResource* dst,
    Offset dstOffset,
    IBufferResource* src,
    Offset srcOffset,
    Size size)
{
    auto encoder = m_commandBuffer->getMetalBlitCommandEncoder();
    encoder->copyFromBuffer(
        static_cast<BufferResourceImpl*>(src)->m_buffer.get(),
        srcOffset,
        static_cast<BufferResourceImpl*>(dst)->m_buffer.get(),
        dstOffset,
        size);
}

void ResourceCommandEncoder::copyTexture(
    ITextureResource* dst,
    ResourceState dstState,
    SubresourceRange dstSubresource,
    ITextureResource::Offset3D dstOffset,
    ITextureResource* src,
    ResourceState srcState,
    SubresourceRange srcSubresource,
    ITextureResource::Offset3D srcOffset,
    ITextureResource::Extents extent)
{
    auto encoder = m_commandBuffer->getMetalBlitCommandEncoder();

    if (dstSubresource.layerCount == 0 && dstSubresource.mipLevelCount == 0 &&
        srcSubresource.layerCount == 0 && srcSubresource.mipLevelCount == 0)
    {
        encoder->copyFromTexture(
            static_cast<TextureResourceImpl*>(src)->m_texture.get(),
            static_cast<TextureResourceImpl*>(dst)->m_texture.get());
    }
    else
    {
        for (GfxIndex layer = 0; layer < dstSubresource.layerCount; layer++)
        {
            encoder->copyFromTexture(
                static_cast<TextureResourceImpl*>(src)->m_texture.get(),
                srcSubresource.baseArrayLayer + layer,
                srcSubresource.mipLevel,
                MTL::Origin(srcOffset.x, srcOffset.y, srcOffset.z),
                MTL::Size(extent.width, extent.height, extent.depth),
                static_cast<TextureResourceImpl*>(dst)->m_texture.get(),
                dstSubresource.baseArrayLayer + layer,
                dstSubresource.mipLevel,
                MTL::Origin(dstOffset.x, dstOffset.y, dstOffset.z));
        }
    }
}

void ResourceCommandEncoder::copyTextureToBuffer(
    IBufferResource* dst,
    Offset dstOffset,
    Size dstSize,
    Size dstRowStride,
    ITextureResource* src,
    ResourceState srcState,
    SubresourceRange srcSubresource,
    ITextureResource::Offset3D srcOffset,
    ITextureResource::Extents extent)
{
    assert(srcSubresource.mipLevelCount <= 1);

    auto encoder = m_commandBuffer->getMetalBlitCommandEncoder();
    auto& desc = *static_cast<TextureResourceImpl*>(src)->getDesc();
    const TextureResource::Extents mipSize = calcMipSize(desc.size, srcSubresource.mipLevel);
    Size bytesPerImage = mipSize.height * dstRowStride;

    encoder->copyFromTexture(
        static_cast<TextureResourceImpl*>(src)->m_texture.get(),
        srcSubresource.baseArrayLayer,
        srcSubresource.mipLevel,
        MTL::Origin(srcOffset.x, srcOffset.y, srcOffset.z),
        MTL::Size(extent.width, extent.height, extent.depth),
        static_cast<BufferResourceImpl*>(dst)->m_buffer.get(),
        dstOffset,
        dstRowStride,
        extent.depth == 1 ? 0 : bytesPerImage);
}

void ResourceCommandEncoder::uploadBufferData(
    IBufferResource* buffer,
    Offset offset,
    Size size,
    void* data)
{
    SLANG_UNIMPLEMENTED_X("uploadBufferData");
}

void ResourceCommandEncoder::uploadTextureData(
    ITextureResource* dst,
    SubresourceRange subResourceRange,
    ITextureResource::Offset3D offset,
    ITextureResource::Extents extend,
    ITextureResource::SubresourceData* subResourceData,
    GfxCount subResourceDataCount)
{
    auto dstTexture = static_cast<TextureResourceImpl*>(dst);
    auto& desc = *dstTexture->getDesc();

    // Calculate buffer size needed
    Size bufferSize = 0;
    FormatInfo sizeInfo;
    gfxGetFormatInfo(desc.format, &sizeInfo);
    MTL::PixelFormat pixelFormat = MetalUtil::translatePixelFormat(desc.format);
    bool isCompressed = gfxIsCompressedFormat(desc.format);

    Size rowAlignment =
        isCompressed
            ? 1
            : m_commandBuffer->m_device->m_device->minimumLinearTextureAlignmentForPixelFormat(
                  pixelFormat);

    for (GfxIndex i = 0; i < subResourceRange.mipLevelCount; ++i)
    {
        GfxIndex currentLevel = subResourceRange.mipLevel + i;
        const TextureResource::Extents mipSize = calcMipSize(desc.size, currentLevel);

        auto rowSizeInBytes = (mipSize.width + sizeInfo.blockWidth - 1) / sizeInfo.blockWidth *
                              sizeInfo.blockSizeInBytes;
        rowSizeInBytes = (rowSizeInBytes + rowAlignment - 1) & ~(rowAlignment - 1);

        auto numRows = (mipSize.height + sizeInfo.blockHeight - 1) / sizeInfo.blockHeight;
        bufferSize += (rowSizeInBytes * numRows) * mipSize.depth;
    }
    bufferSize *= subResourceRange.layerCount;

    // Create staging buffer
    NS::SharedPtr<MTL::Buffer> stagingBuffer = NS::TransferPtr(
        m_commandBuffer->m_device->m_device->newBuffer(bufferSize, MTL::ResourceStorageModeShared));

    if (!stagingBuffer)
        return;

    auto encoder = m_commandBuffer->getMetalBlitCommandEncoder();
    if (!encoder)
        return;

    // Copy data to staging buffer and then to texture
    Size bufferOffset = 0;
    for (GfxIndex i = 0; i < subResourceRange.layerCount; i++)
    {
        // We only allocate staging buffer with size of one slice.
        GfxIndex currentSlice = subResourceRange.baseArrayLayer + i;
        uint8_t* bufferData = (uint8_t*)stagingBuffer->contents();
        Size dstOffset = 0;

        for (GfxIndex j = 0; j < subResourceRange.mipLevelCount; j++)
        {
            GfxIndex currentLevel = subResourceRange.mipLevel + j;
            const auto& subresourceData = subResourceData[j];
            const TextureResource::Extents mipSize = calcMipSize(desc.size, currentLevel);

            auto rowSizeInBytes = (mipSize.width + sizeInfo.blockWidth - 1) / sizeInfo.blockWidth *
                                  sizeInfo.blockSizeInBytes;
            auto rowSizeInBytesAligned = (rowSizeInBytes + rowAlignment - 1) & ~(rowAlignment - 1);

            auto numRows = (mipSize.height + sizeInfo.blockHeight - 1) / sizeInfo.blockHeight;

            const uint8_t* srcData = (const uint8_t*)subresourceData.data;
            if (rowSizeInBytesAligned == rowSizeInBytes)
            {
                // If the row size is already aligned, we can copy the data directly.
                memcpy(bufferData + dstOffset, srcData, rowSizeInBytes * numRows * mipSize.depth);
            }
            else
            {
                for (GfxIndex k = 0; k < mipSize.depth; ++k)
                {
                    for (GfxIndex row = 0; row < numRows; ++row)
                    {
                        // Copy data to staging buffer, note that the staging buffer has to have the
                        // same alignment as the texture while the src data doesn't have such
                        // requirement, therefore we have to copy the data row by row. We don't care
                        // about the content of the alignment padding.
                        memcpy(bufferData + dstOffset, srcData, rowSizeInBytes);
                        dstOffset += rowSizeInBytesAligned;
                        srcData += rowSizeInBytes;
                    }
                }
            }

            // Copy from staging buffer to texture
            encoder->copyFromBuffer(
                stagingBuffer.get(),
                bufferOffset,
                rowSizeInBytesAligned,
                rowSizeInBytesAligned * numRows,
                MTL::Size(mipSize.width, mipSize.height, mipSize.depth),
                dstTexture->m_texture.get(),
                currentSlice,
                currentLevel,
                MTL::Origin(offset.x, offset.y, offset.z));

            bufferOffset += rowSizeInBytes * numRows * mipSize.depth;
        }
    }
}

void ResourceCommandEncoder::bufferBarrier(
    GfxCount count,
    IBufferResource* const* buffers,
    ResourceState src,
    ResourceState dst)
{
    // We use automatic hazard tracking for now, no need for barriers.
}

void ResourceCommandEncoder::textureBarrier(
    GfxCount count,
    ITextureResource* const* textures,
    ResourceState src,
    ResourceState dst)
{
    // We use automatic hazard tracking for now, no need for barriers.
}

void ResourceCommandEncoder::textureSubresourceBarrier(
    ITextureResource* texture,
    SubresourceRange subresourceRange,
    ResourceState src,
    ResourceState dst)
{
    // We use automatic hazard tracking for now, no need for barriers.
}

void ResourceCommandEncoder::clearResourceView(
    IResourceView* view,
    ClearValue* clearValue,
    ClearResourceViewFlags::Enum flags)
{
    SLANG_UNIMPLEMENTED_X("clearResourceView");
}

void ResourceCommandEncoder::resolveResource(
    ITextureResource* source,
    ResourceState sourceState,
    SubresourceRange sourceRange,
    ITextureResource* dest,
    ResourceState destState,
    SubresourceRange destRange)
{
    SLANG_UNIMPLEMENTED_X("resolveResource");
}

void ResourceCommandEncoder::resolveQuery(
    IQueryPool* queryPool,
    GfxIndex index,
    GfxCount count,
    IBufferResource* buffer,
    Offset offset)
{
    auto encoder = m_commandBuffer->getMetalBlitCommandEncoder();
    encoder->resolveCounters(
        static_cast<QueryPoolImpl*>(queryPool)->m_counterSampleBuffer.get(),
        NS::Range(index, count),
        static_cast<BufferResourceImpl*>(buffer)->m_buffer.get(),
        offset);
}

void ResourceCommandEncoder::beginDebugEvent(const char* name, float rgbColor[3])
{
    NS::SharedPtr<NS::String> string = MetalUtil::createString(name);
    m_commandBuffer->m_commandBuffer->pushDebugGroup(string.get());
}

void ResourceCommandEncoder::endDebugEvent()
{
    m_commandBuffer->m_commandBuffer->popDebugGroup();
}

void RenderCommandEncoder::beginPass(IRenderPassLayout* renderPass, IFramebuffer* framebuffer)
{
    m_renderPassLayout = static_cast<RenderPassLayoutImpl*>(renderPass);
    m_framebuffer = static_cast<FramebufferImpl*>(framebuffer);
    if (!m_framebuffer)
    {
        // TODO use empty framebuffer
        return;
    }

    // Create a copy of the render pass descriptor and fill in remaining information.
    m_renderPassDesc = NS::TransferPtr(m_renderPassLayout->m_renderPassDesc->copy());

    m_renderPassDesc->setRenderTargetWidth(m_framebuffer->m_width);
    m_renderPassDesc->setRenderTargetHeight(m_framebuffer->m_height);

    for (Index i = 0; i < m_framebuffer->m_renderTargetViews.getCount(); ++i)
    {
        TextureResourceViewImpl* renderTargetView = m_framebuffer->m_renderTargetViews[i];
        MTL::RenderPassColorAttachmentDescriptor* colorAttachment =
            m_renderPassDesc->colorAttachments()->object(i);
        colorAttachment->setTexture(renderTargetView->m_textureView.get());
        colorAttachment->setLevel(renderTargetView->m_desc.subresourceRange.mipLevel);
        colorAttachment->setSlice(renderTargetView->m_desc.subresourceRange.baseArrayLayer);
    }

    if (m_framebuffer->m_depthStencilView)
    {
        TextureResourceViewImpl* depthStencilView = m_framebuffer->m_depthStencilView.get();
        MTL::PixelFormat pixelFormat =
            MetalUtil::translatePixelFormat(depthStencilView->m_desc.format);
        if (MetalUtil::isDepthFormat(pixelFormat))
        {
            MTL::RenderPassDepthAttachmentDescriptor* depthAttachment =
                m_renderPassDesc->depthAttachment();
            depthAttachment->setTexture(depthStencilView->m_textureView.get());
            depthAttachment->setLevel(depthStencilView->m_desc.subresourceRange.mipLevel);
            depthAttachment->setSlice(depthStencilView->m_desc.subresourceRange.baseArrayLayer);
        }
        if (MetalUtil::isStencilFormat(pixelFormat))
        {
            MTL::RenderPassStencilAttachmentDescriptor* stencilAttachment =
                m_renderPassDesc->stencilAttachment();
            stencilAttachment->setTexture(depthStencilView->m_textureView.get());
            stencilAttachment->setLevel(depthStencilView->m_desc.subresourceRange.mipLevel);
            stencilAttachment->setSlice(depthStencilView->m_desc.subresourceRange.baseArrayLayer);
        }
    }
}

void RenderCommandEncoder::endEncoding()
{
    PipelineCommandEncoder::endEncodingImpl();
}

Result RenderCommandEncoder::bindPipeline(
    IPipelineState* pipelineState,
    IShaderObject** outRootObject)
{
    return setPipelineStateImpl(pipelineState, outRootObject);
}

Result RenderCommandEncoder::bindPipelineWithRootObject(
    IPipelineState* pipelineState,
    IShaderObject* rootObject)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

void RenderCommandEncoder::setViewports(GfxCount count, const Viewport* viewports)
{
    m_viewports.setCount(count);
    for (GfxIndex i = 0; i < count; ++i)
    {
        const auto& viewport = viewports[i];
        auto& mtlViewport = m_viewports[i];
        mtlViewport.originX = viewport.originX;
        mtlViewport.originY = viewport.originY;
        mtlViewport.width = viewport.extentX;
        mtlViewport.height = viewport.extentY;
        mtlViewport.znear = viewport.minZ;
        mtlViewport.zfar = viewport.maxZ;
    }
}

void RenderCommandEncoder::setScissorRects(GfxCount count, const ScissorRect* rects)
{
    m_scissorRects.setCount(count);
    for (GfxIndex i = 0; i < count; ++i)
    {
        const auto& rect = rects[i];
        auto& mtlRect = m_scissorRects[i];
        mtlRect.x = rect.minX;
        mtlRect.y = rect.minY;
        mtlRect.width = rect.maxX - rect.minX;
        mtlRect.height = rect.maxY - rect.minY;
    }
}

void RenderCommandEncoder::setPrimitiveTopology(PrimitiveTopology topology)
{
    m_primitiveType = MetalUtil::translatePrimitiveType(topology);
}

void RenderCommandEncoder::setVertexBuffers(
    GfxIndex startSlot,
    GfxCount slotCount,
    IBufferResource* const* buffers,
    const Offset* offsets)
{
    Index count = Math::Max(m_vertexBuffers.getCount(), Index(startSlot + slotCount));
    m_vertexBuffers.setCount(count);
    m_vertexBufferOffsets.setCount(count);

    for (Index i = 0; i < Index(slotCount); i++)
    {
        Index slotIndex = startSlot + i;
        m_vertexBuffers[slotIndex] = static_cast<BufferResourceImpl*>(buffers[i])->m_buffer.get();
        m_vertexBufferOffsets[slotIndex] = offsets[i];
    }
}

void RenderCommandEncoder::setIndexBuffer(
    IBufferResource* buffer,
    Format indexFormat,
    Offset offset)
{
    m_indexBuffer = static_cast<BufferResourceImpl*>(buffer)->m_buffer.get();
    m_indexBufferOffset = offset;

    switch (indexFormat)
    {
    case Format::R16_UINT:
        m_indexBufferType = MTL::IndexTypeUInt16;
        break;
    case Format::R32_UINT:
        m_indexBufferType = MTL::IndexTypeUInt32;
        break;
    default:
        assert(!"unsupported index format");
    }
}

void RenderCommandEncoder::setStencilReference(uint32_t referenceValue)
{
    m_stencilReferenceValue = referenceValue;
}

Result RenderCommandEncoder::setSamplePositions(
    GfxCount samplesPerPixel,
    GfxCount pixelCount,
    const SamplePosition* samplePositions)
{
    return SLANG_E_NOT_AVAILABLE;
}

Result RenderCommandEncoder::prepareDraw(MTL::RenderCommandEncoder*& encoder)
{
    auto pipeline = static_cast<PipelineStateImpl*>(m_currentPipeline.Ptr());
    pipeline->ensureAPIPipelineStateCreated();

    encoder = m_commandBuffer->getMetalRenderCommandEncoder(m_renderPassDesc.get());
    encoder->setRenderPipelineState(pipeline->m_renderPipelineState.get());

    RenderBindingContext bindingContext;
    bindingContext.init(m_commandBuffer->m_device, encoder);
    auto program = static_cast<ShaderProgramImpl*>(m_currentPipeline->m_program.get());
    m_commandBuffer->m_rootObject.bindAsRoot(&bindingContext, program->m_rootObjectLayout);

    for (Index i = 0; i < m_vertexBuffers.getCount(); ++i)
    {
        encoder->setVertexBuffer(
            m_vertexBuffers[i],
            m_vertexBufferOffsets[i],
            m_currentPipeline->m_vertexBufferOffset + i);
    }

    encoder->setViewports(m_viewports.getArrayView().getBuffer(), m_viewports.getCount());
    encoder->setScissorRects(m_scissorRects.getArrayView().getBuffer(), m_scissorRects.getCount());

    const RasterizerDesc& rasterDesc = pipeline->desc.graphics.rasterizer;
    const DepthStencilDesc& depthStencilDesc = pipeline->desc.graphics.depthStencil;
    encoder->setFrontFacingWinding(MetalUtil::translateWinding(rasterDesc.frontFace));
    encoder->setCullMode(MetalUtil::translateCullMode(rasterDesc.cullMode));
    encoder->setDepthClipMode(
        rasterDesc.depthClipEnable ? MTL::DepthClipModeClip
                                   : MTL::DepthClipModeClamp); // TODO correct?
    encoder->setDepthBias(
        rasterDesc.depthBias,
        rasterDesc.slopeScaledDepthBias,
        rasterDesc.depthBiasClamp);
    encoder->setTriangleFillMode(MetalUtil::translateTriangleFillMode(rasterDesc.fillMode));
    // encoder->setBlendColor(); // not supported by gfx
    if (m_framebuffer->m_depthStencilView)
    {
        encoder->setDepthStencilState(pipeline->m_depthStencilState.get());
    }
    encoder->setStencilReferenceValue(m_stencilReferenceValue);

    return SLANG_OK;
}

Result RenderCommandEncoder::draw(GfxCount vertexCount, GfxIndex startVertex)
{
    MTL::RenderCommandEncoder* encoder;
    SLANG_RETURN_ON_FAIL(prepareDraw(encoder));
    encoder->drawPrimitives(m_primitiveType, startVertex, vertexCount);
    return SLANG_OK;
}

Result RenderCommandEncoder::drawIndexed(
    GfxCount indexCount,
    GfxIndex startIndex,
    GfxIndex baseVertex)
{
    MTL::RenderCommandEncoder* encoder;
    SLANG_RETURN_ON_FAIL(prepareDraw(encoder));
    // TODO baseVertex is not supported by Metal
    encoder->drawIndexedPrimitives(
        m_primitiveType,
        indexCount,
        m_indexBufferType,
        m_indexBuffer,
        m_indexBufferOffset);
    return SLANG_OK;
}

Result RenderCommandEncoder::drawIndirect(
    GfxCount maxDrawCount,
    IBufferResource* argBuffer,
    Offset argOffset,
    IBufferResource* countBuffer,
    Offset countOffset)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

Result RenderCommandEncoder::drawIndexedIndirect(
    GfxCount maxDrawCount,
    IBufferResource* argBuffer,
    Offset argOffset,
    IBufferResource* countBuffer,
    Offset countOffset)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

Result RenderCommandEncoder::drawInstanced(
    GfxCount vertexCount,
    GfxCount instanceCount,
    GfxIndex startVertex,
    GfxIndex startInstanceLocation)
{
    MTL::RenderCommandEncoder* encoder;
    SLANG_RETURN_ON_FAIL(prepareDraw(encoder));
    encoder->drawPrimitives(
        m_primitiveType,
        startVertex,
        vertexCount,
        instanceCount,
        startInstanceLocation);
    return SLANG_OK;
}

Result RenderCommandEncoder::drawIndexedInstanced(
    GfxCount indexCount,
    GfxCount instanceCount,
    GfxIndex startIndexLocation,
    GfxIndex baseVertexLocation,
    GfxIndex startInstanceLocation)
{
    MTL::RenderCommandEncoder* encoder;
    SLANG_RETURN_ON_FAIL(prepareDraw(encoder));
    encoder->drawIndexedPrimitives(
        m_primitiveType,
        indexCount,
        m_indexBufferType,
        m_indexBuffer,
        startIndexLocation,
        instanceCount,
        baseVertexLocation,
        startIndexLocation);
    return SLANG_OK;
}

Result RenderCommandEncoder::drawMeshTasks(int x, int y, int z)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

void ComputeCommandEncoder::endEncoding()
{
    ResourceCommandEncoder::endEncoding();
}

Result ComputeCommandEncoder::bindPipeline(
    IPipelineState* pipelineState,
    IShaderObject** outRootObject)
{
    return setPipelineStateImpl(pipelineState, outRootObject);
}

Result ComputeCommandEncoder::bindPipelineWithRootObject(
    IPipelineState* pipelineState,
    IShaderObject* rootObject)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

Result ComputeCommandEncoder::dispatchCompute(int x, int y, int z)
{
    MTL::ComputeCommandEncoder* encoder = m_commandBuffer->getMetalComputeCommandEncoder();

    ComputeBindingContext bindingContext;
    bindingContext.init(m_commandBuffer->m_device, encoder);
    auto program = static_cast<ShaderProgramImpl*>(m_currentPipeline->m_program.get());
    m_commandBuffer->m_rootObject.bindAsRoot(&bindingContext, program->m_rootObjectLayout);

    auto pipeline = static_cast<PipelineStateImpl*>(m_currentPipeline.Ptr());
    RootShaderObjectImpl* rootObjectImpl = &m_commandBuffer->m_rootObject;
    RefPtr<PipelineStateBase> newPipeline;
    SLANG_RETURN_ON_FAIL(m_commandBuffer->m_device->maybeSpecializePipeline(
        m_currentPipeline,
        rootObjectImpl,
        newPipeline));
    PipelineStateImpl* newPipelineImpl = static_cast<PipelineStateImpl*>(newPipeline.Ptr());

    SLANG_RETURN_ON_FAIL(newPipelineImpl->ensureAPIPipelineStateCreated());
    m_currentPipeline = newPipelineImpl;

    m_currentPipeline->ensureAPIPipelineStateCreated();
    encoder->setComputePipelineState(m_currentPipeline->m_computePipelineState.get());


    encoder->dispatchThreadgroups(MTL::Size(x, y, z), m_currentPipeline->m_threadGroupSize);

    return SLANG_OK;
}

Result ComputeCommandEncoder::dispatchComputeIndirect(IBufferResource* argBuffer, Offset offset)
{
    SLANG_UNIMPLEMENTED_X("dispatchComputeIndirect");
}

void RayTracingCommandEncoder::_memoryBarrier(
    int count,
    IAccelerationStructure* const* structures,
    AccessFlag srcAccess,
    AccessFlag destAccess)
{
}

void RayTracingCommandEncoder::_queryAccelerationStructureProperties(
    GfxCount accelerationStructureCount,
    IAccelerationStructure* const* accelerationStructures,
    GfxCount queryCount,
    AccelerationStructureQueryDesc* queryDescs)
{
}

void RayTracingCommandEncoder::buildAccelerationStructure(
    const IAccelerationStructure::BuildDesc& desc,
    GfxCount propertyQueryCount,
    AccelerationStructureQueryDesc* queryDescs)
{
}

void RayTracingCommandEncoder::copyAccelerationStructure(
    IAccelerationStructure* dest,
    IAccelerationStructure* src,
    AccelerationStructureCopyMode mode)
{
}

void RayTracingCommandEncoder::queryAccelerationStructureProperties(
    GfxCount accelerationStructureCount,
    IAccelerationStructure* const* accelerationStructures,
    GfxCount queryCount,
    AccelerationStructureQueryDesc* queryDescs)
{
    _queryAccelerationStructureProperties(
        accelerationStructureCount,
        accelerationStructures,
        queryCount,
        queryDescs);
}

void RayTracingCommandEncoder::serializeAccelerationStructure(
    DeviceAddress dest,
    IAccelerationStructure* source)
{
}

void RayTracingCommandEncoder::deserializeAccelerationStructure(
    IAccelerationStructure* dest,
    DeviceAddress source)
{
}

Result RayTracingCommandEncoder::bindPipeline(
    IPipelineState* pipeline,
    IShaderObject** outRootObject)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

Result RayTracingCommandEncoder::bindPipelineWithRootObject(
    IPipelineState* pipelineState,
    IShaderObject* rootObject)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

Result RayTracingCommandEncoder::dispatchRays(
    GfxIndex raygenShaderIndex,
    IShaderTable* shaderTable,
    GfxCount width,
    GfxCount height,
    GfxCount depth)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

void RayTracingCommandEncoder::endEncoding() {}

} // namespace metal
} // namespace gfx
