// debug-command-encoder.cpp
#include "debug-command-encoder.h"

#include "debug-buffer.h"
#include "debug-command-buffer.h"
#include "debug-helper-functions.h"
#include "debug-pipeline-state.h"
#include "debug-query.h"
#include "debug-resource-views.h"
#include "debug-texture.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

void DebugComputeCommandEncoder::endEncoding()
{
    SLANG_GFX_API_FUNC;
    isOpen = false;
    baseObject->endEncoding();
}

Result DebugComputeCommandEncoder::bindPipeline(
    IPipelineState* state,
    IShaderObject** outRootShaderObject)
{
    SLANG_GFX_API_FUNC;

    auto innerState = getInnerObj(state);
    IShaderObject* innerRootObject = nullptr;
    commandBuffer->rootObject.reset();
    auto result = baseObject->bindPipeline(innerState, &innerRootObject);
    commandBuffer->rootObject.baseObject.attach(innerRootObject);
    *outRootShaderObject = &commandBuffer->rootObject;
    return result;
}

Result DebugComputeCommandEncoder::bindPipelineWithRootObject(
    IPipelineState* state,
    IShaderObject* rootObject)
{
    SLANG_GFX_API_FUNC;
    return baseObject->bindPipelineWithRootObject(getInnerObj(state), getInnerObj(rootObject));
}

Result DebugComputeCommandEncoder::dispatchCompute(int x, int y, int z)
{
    SLANG_GFX_API_FUNC;
    return baseObject->dispatchCompute(x, y, z);
}

Result DebugComputeCommandEncoder::dispatchComputeIndirect(
    IBufferResource* cmdBuffer,
    Offset offset)
{
    SLANG_GFX_API_FUNC;
    return baseObject->dispatchComputeIndirect(getInnerObj(cmdBuffer), offset);
}

void DebugRenderCommandEncoder::endEncoding()
{
    SLANG_GFX_API_FUNC;
    isOpen = false;
    baseObject->endEncoding();
}

Result DebugRenderCommandEncoder::bindPipeline(
    IPipelineState* state,
    IShaderObject** outRootShaderObject)
{
    SLANG_GFX_API_FUNC;

    auto innerState = getInnerObj(state);
    IShaderObject* innerRootObject = nullptr;
    commandBuffer->rootObject.reset();
    auto result = baseObject->bindPipeline(innerState, &innerRootObject);
    commandBuffer->rootObject.baseObject.attach(innerRootObject);
    *outRootShaderObject = &commandBuffer->rootObject;
    return result;
}

Result DebugRenderCommandEncoder::bindPipelineWithRootObject(
    IPipelineState* state,
    IShaderObject* rootObject)
{
    SLANG_GFX_API_FUNC;
    return baseObject->bindPipelineWithRootObject(getInnerObj(state), getInnerObj(rootObject));
}

void DebugRenderCommandEncoder::setViewports(GfxCount count, const Viewport* viewports)
{
    SLANG_GFX_API_FUNC;
    baseObject->setViewports(count, viewports);
}

void DebugRenderCommandEncoder::setScissorRects(GfxCount count, const ScissorRect* scissors)
{
    SLANG_GFX_API_FUNC;
    baseObject->setScissorRects(count, scissors);
}

void DebugRenderCommandEncoder::setPrimitiveTopology(PrimitiveTopology topology)
{
    SLANG_GFX_API_FUNC;
    baseObject->setPrimitiveTopology(topology);
}

void DebugRenderCommandEncoder::setVertexBuffers(
    GfxIndex startSlot,
    GfxCount slotCount,
    IBufferResource* const* buffers,
    const Offset* offsets)
{
    SLANG_GFX_API_FUNC;

    List<IBufferResource*> innerBuffers;
    for (GfxIndex i = 0; i < slotCount; i++)
    {
        innerBuffers.add(static_cast<DebugBufferResource*>(buffers[i])->baseObject.get());
    }
    baseObject->setVertexBuffers(startSlot, slotCount, innerBuffers.getBuffer(), offsets);
}

void DebugRenderCommandEncoder::setIndexBuffer(
    IBufferResource* buffer,
    Format indexFormat,
    Offset offset)
{
    SLANG_GFX_API_FUNC;
    auto innerBuffer = static_cast<DebugBufferResource*>(buffer)->baseObject.get();
    baseObject->setIndexBuffer(innerBuffer, indexFormat, offset);
}

Result DebugRenderCommandEncoder::draw(GfxCount vertexCount, GfxIndex startVertex)
{
    SLANG_GFX_API_FUNC;
    return baseObject->draw(vertexCount, startVertex);
}

Result DebugRenderCommandEncoder::drawIndexed(
    GfxCount indexCount,
    GfxIndex startIndex,
    GfxIndex baseVertex)
{
    SLANG_GFX_API_FUNC;
    return baseObject->drawIndexed(indexCount, startIndex, baseVertex);
}

Result DebugRenderCommandEncoder::drawIndirect(
    GfxCount maxDrawCount,
    IBufferResource* argBuffer,
    Offset argOffset,
    IBufferResource* countBuffer,
    Offset countOffset)
{
    SLANG_GFX_API_FUNC;
    return baseObject->drawIndirect(
        maxDrawCount,
        getInnerObj(argBuffer),
        argOffset,
        getInnerObj(countBuffer),
        countOffset);
}

Result DebugRenderCommandEncoder::drawIndexedIndirect(
    GfxCount maxDrawCount,
    IBufferResource* argBuffer,
    Offset argOffset,
    IBufferResource* countBuffer,
    Offset countOffset)
{
    SLANG_GFX_API_FUNC;
    return baseObject->drawIndexedIndirect(
        maxDrawCount,
        getInnerObj(argBuffer),
        argOffset,
        getInnerObj(countBuffer),
        countOffset);
}

void DebugRenderCommandEncoder::setStencilReference(uint32_t referenceValue)
{
    SLANG_GFX_API_FUNC;
    return baseObject->setStencilReference(referenceValue);
}

Result DebugRenderCommandEncoder::setSamplePositions(
    GfxCount samplesPerPixel,
    GfxCount pixelCount,
    const SamplePosition* samplePositions)
{
    SLANG_GFX_API_FUNC;
    return baseObject->setSamplePositions(samplesPerPixel, pixelCount, samplePositions);
}

Result DebugRenderCommandEncoder::drawInstanced(
    GfxCount vertexCount,
    GfxCount instanceCount,
    GfxIndex startVertex,
    GfxIndex startInstanceLocation)
{
    SLANG_GFX_API_FUNC;
    return baseObject
        ->drawInstanced(vertexCount, instanceCount, startVertex, startInstanceLocation);
}

Result DebugRenderCommandEncoder::drawIndexedInstanced(
    GfxCount indexCount,
    GfxCount instanceCount,
    GfxIndex startIndexLocation,
    GfxIndex baseVertexLocation,
    GfxIndex startInstanceLocation)
{
    SLANG_GFX_API_FUNC;
    return baseObject->drawIndexedInstanced(
        indexCount,
        instanceCount,
        startIndexLocation,
        baseVertexLocation,
        startInstanceLocation);
}

Result DebugRenderCommandEncoder::drawMeshTasks(int x, int y, int z)
{
    SLANG_GFX_API_FUNC;
    return baseObject->drawMeshTasks(x, y, z);
}

void DebugResourceCommandEncoder::endEncoding()
{
    SLANG_GFX_API_FUNC;
    isOpen = false;
    baseObject->endEncoding();
}

void DebugResourceCommandEncoderImpl::writeTimestamp(IQueryPool* pool, GfxIndex index)
{
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()->writeTimestamp(static_cast<DebugQueryPool*>(pool)->baseObject, index);
}

void DebugResourceCommandEncoderImpl::copyBuffer(
    IBufferResource* dst,
    Offset dstOffset,
    IBufferResource* src,
    Offset srcOffset,
    Size size)
{
    SLANG_GFX_API_FUNC;
    auto dstImpl = static_cast<DebugBufferResource*>(dst);
    auto srcImpl = static_cast<DebugBufferResource*>(src);
    getBaseResourceEncoder()
        ->copyBuffer(dstImpl->baseObject, dstOffset, srcImpl->baseObject, srcOffset, size);
}

void DebugResourceCommandEncoderImpl::uploadBufferData(
    IBufferResource* dst,
    Offset offset,
    Size size,
    void* data)
{
    SLANG_GFX_API_FUNC;
    auto dstImpl = static_cast<DebugBufferResource*>(dst);
    getBaseResourceEncoder()->uploadBufferData(dstImpl->baseObject, offset, size, data);
}

void DebugResourceCommandEncoderImpl::textureBarrier(
    GfxCount count,
    ITextureResource* const* textures,
    ResourceState src,
    ResourceState dst)
{
    SLANG_GFX_API_FUNC;

    List<ITextureResource*> innerTextures;
    for (GfxIndex i = 0; i < count; i++)
    {
        innerTextures.add(static_cast<DebugTextureResource*>(textures[i])->baseObject.get());
    }
    getBaseResourceEncoder()->textureBarrier(count, innerTextures.getBuffer(), src, dst);
}

void DebugResourceCommandEncoderImpl::bufferBarrier(
    GfxCount count,
    IBufferResource* const* buffers,
    ResourceState src,
    ResourceState dst)
{
    SLANG_GFX_API_FUNC;

    List<IBufferResource*> innerBuffers;
    for (GfxIndex i = 0; i < count; i++)
    {
        innerBuffers.add(static_cast<DebugBufferResource*>(buffers[i])->baseObject.get());
    }
    getBaseResourceEncoder()->bufferBarrier(count, innerBuffers.getBuffer(), src, dst);
}

void DebugResourceCommandEncoderImpl::copyTexture(
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
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()->copyTexture(
        getInnerObj(dst),
        dstState,
        dstSubresource,
        dstOffset,
        getInnerObj(src),
        srcState,
        srcSubresource,
        srcOffset,
        extent);
}

void DebugResourceCommandEncoderImpl::uploadTextureData(
    ITextureResource* dst,
    SubresourceRange subResourceRange,
    ITextureResource::Offset3D offset,
    ITextureResource::Extents extent,
    ITextureResource::SubresourceData* subResourceData,
    GfxCount subResourceDataCount)
{
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()->uploadTextureData(
        getInnerObj(dst),
        subResourceRange,
        offset,
        extent,
        subResourceData,
        subResourceDataCount);
}

void DebugResourceCommandEncoderImpl::clearResourceView(
    IResourceView* view,
    ClearValue* clearValue,
    ClearResourceViewFlags::Enum flags)
{
    SLANG_GFX_API_FUNC;
    switch (view->getViewDesc()->type)
    {
    case IResourceView::Type::DepthStencil:
    case IResourceView::Type::RenderTarget:
    case IResourceView::Type::UnorderedAccess:
        break;
    default:
        GFX_DIAGNOSE_ERROR_FORMAT(
            "Resource view %lld cannot be cleared. Only DepthStencil, "
            "RenderTarget or UnorderedAccess views can be cleared.",
            getDebugObj(view)->uid);
    }
    getBaseResourceEncoder()->clearResourceView(getInnerObj(view), clearValue, flags);
}

void DebugResourceCommandEncoderImpl::resolveResource(
    ITextureResource* source,
    ResourceState sourceState,
    SubresourceRange sourceRange,
    ITextureResource* dest,
    ResourceState destState,
    SubresourceRange destRange)
{
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()->resolveResource(
        getInnerObj(source),
        sourceState,
        sourceRange,
        getInnerObj(dest),
        destState,
        destRange);
}

void DebugResourceCommandEncoderImpl::resolveQuery(
    IQueryPool* queryPool,
    GfxIndex index,
    GfxCount count,
    IBufferResource* buffer,
    Offset offset)
{
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()
        ->resolveQuery(getInnerObj(queryPool), index, count, getInnerObj(buffer), offset);
}

void DebugResourceCommandEncoderImpl::copyTextureToBuffer(
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
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()->copyTextureToBuffer(
        getInnerObj(dst),
        dstOffset,
        dstSize,
        dstRowStride,
        getInnerObj(src),
        srcState,
        srcSubresource,
        srcOffset,
        extent);
}

void DebugResourceCommandEncoderImpl::textureSubresourceBarrier(
    ITextureResource* texture,
    SubresourceRange subresourceRange,
    ResourceState src,
    ResourceState dst)
{
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()
        ->textureSubresourceBarrier(getInnerObj(texture), subresourceRange, src, dst);
}

void DebugResourceCommandEncoderImpl::beginDebugEvent(const char* name, float rgbColor[3])
{
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()->beginDebugEvent(name, rgbColor);
}

void DebugResourceCommandEncoderImpl::endDebugEvent()
{
    SLANG_GFX_API_FUNC;
    getBaseResourceEncoder()->endDebugEvent();
}

void DebugRayTracingCommandEncoder::endEncoding()
{
    SLANG_GFX_API_FUNC;
    isOpen = false;
    baseObject->endEncoding();
}

void DebugRayTracingCommandEncoder::buildAccelerationStructure(
    const IAccelerationStructure::BuildDesc& desc,
    GfxCount propertyQueryCount,
    AccelerationStructureQueryDesc* queryDescs)
{
    SLANG_GFX_API_FUNC;
    IAccelerationStructure::BuildDesc innerDesc = desc;
    innerDesc.dest = getInnerObj(innerDesc.dest);
    innerDesc.source = getInnerObj(innerDesc.source);
    List<AccelerationStructureQueryDesc> innerQueryDescs;
    innerQueryDescs.addRange(queryDescs, propertyQueryCount);
    for (auto& innerQueryDesc : innerQueryDescs)
    {
        innerQueryDesc.queryPool = getInnerObj(innerQueryDesc.queryPool);
    }
    validateAccelerationStructureBuildInputs(desc.inputs);
    baseObject->buildAccelerationStructure(
        innerDesc,
        propertyQueryCount,
        innerQueryDescs.getBuffer());
}

void DebugRayTracingCommandEncoder::copyAccelerationStructure(
    IAccelerationStructure* dest,
    IAccelerationStructure* src,
    AccelerationStructureCopyMode mode)
{
    SLANG_GFX_API_FUNC;
    auto innerDest = getInnerObj(dest);
    auto innerSrc = getInnerObj(src);
    baseObject->copyAccelerationStructure(innerDest, innerSrc, mode);
}

void DebugRayTracingCommandEncoder::queryAccelerationStructureProperties(
    GfxCount accelerationStructureCount,
    IAccelerationStructure* const* accelerationStructures,
    GfxCount queryCount,
    AccelerationStructureQueryDesc* queryDescs)
{
    SLANG_GFX_API_FUNC;
    List<IAccelerationStructure*> innerAS;
    for (GfxIndex i = 0; i < accelerationStructureCount; i++)
    {
        innerAS.add(getInnerObj(accelerationStructures[i]));
    }
    List<AccelerationStructureQueryDesc> innerQueryDescs;
    innerQueryDescs.addRange(queryDescs, queryCount);
    for (auto& innerQueryDesc : innerQueryDescs)
    {
        innerQueryDesc.queryPool = getInnerObj(innerQueryDesc.queryPool);
    }
    baseObject->queryAccelerationStructureProperties(
        accelerationStructureCount,
        innerAS.getBuffer(),
        queryCount,
        innerQueryDescs.getBuffer());
}

void DebugRayTracingCommandEncoder::serializeAccelerationStructure(
    DeviceAddress dest,
    IAccelerationStructure* source)
{
    SLANG_GFX_API_FUNC;
    baseObject->serializeAccelerationStructure(dest, getInnerObj(source));
}

void DebugRayTracingCommandEncoder::deserializeAccelerationStructure(
    IAccelerationStructure* dest,
    DeviceAddress source)
{
    SLANG_GFX_API_FUNC;
    baseObject->deserializeAccelerationStructure(getInnerObj(dest), source);
}

Result DebugRayTracingCommandEncoder::bindPipeline(
    IPipelineState* state,
    IShaderObject** outRootObject)
{
    SLANG_GFX_API_FUNC;
    auto innerPipeline = getInnerObj(state);
    IShaderObject* innerRootObject = nullptr;
    commandBuffer->rootObject.reset();
    Result result = baseObject->bindPipeline(innerPipeline, &innerRootObject);
    commandBuffer->rootObject.baseObject.attach(innerRootObject);
    *outRootObject = &commandBuffer->rootObject;
    return result;
}

Result DebugRayTracingCommandEncoder::bindPipelineWithRootObject(
    IPipelineState* state,
    IShaderObject* rootObject)
{
    SLANG_GFX_API_FUNC;
    return baseObject->bindPipelineWithRootObject(getInnerObj(state), getInnerObj(rootObject));
}

Result DebugRayTracingCommandEncoder::dispatchRays(
    GfxIndex rayGenShaderIndex,
    IShaderTable* shaderTable,
    GfxCount width,
    GfxCount height,
    GfxCount depth)
{
    SLANG_GFX_API_FUNC;
    return baseObject
        ->dispatchRays(rayGenShaderIndex, getInnerObj(shaderTable), width, height, depth);
}

} // namespace debug
} // namespace gfx
