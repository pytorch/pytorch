// vk-command-encoder.cpp
#include "vk-command-encoder.h"

#include "vk-buffer.h"
#include "vk-command-buffer.h"
#include "vk-helper-functions.h"
#include "vk-query.h"
#include "vk-render-pass.h"
#include "vk-resource-views.h"
#include "vk-shader-object.h"
#include "vk-shader-program.h"
#include "vk-shader-table.h"
#include "vk-texture.h"
#include "vk-transient-heap.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

int PipelineCommandEncoder::getBindPointIndex(VkPipelineBindPoint bindPoint)
{
    switch (bindPoint)
    {
    case VK_PIPELINE_BIND_POINT_GRAPHICS:
        return 0;
    case VK_PIPELINE_BIND_POINT_COMPUTE:
        return 1;
    case VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR:
        return 2;
    default:
        assert(!"unknown pipeline type.");
        return -1;
    }
}

void PipelineCommandEncoder::init(CommandBufferImpl* commandBuffer)
{
    m_commandBuffer = commandBuffer;
    m_device = commandBuffer->m_renderer;
    m_vkCommandBuffer = m_commandBuffer->m_commandBuffer;
    m_api = &m_commandBuffer->m_renderer->m_api;
}

void PipelineCommandEncoder::endEncodingImpl()
{
    for (auto& pipeline : m_boundPipelines)
        pipeline = VK_NULL_HANDLE;
}

void PipelineCommandEncoder::_uploadBufferData(
    VkCommandBuffer commandBuffer,
    TransientResourceHeapImpl* transientHeap,
    BufferResourceImpl* buffer,
    Offset offset,
    Size size,
    void* data)
{
    auto& api = buffer->m_renderer->m_api;
    IBufferResource* stagingBuffer = nullptr;
    Offset stagingBufferOffset = 0;
    transientHeap
        ->allocateStagingBuffer(size, stagingBuffer, stagingBufferOffset, MemoryType::Upload);

    BufferResourceImpl* stagingBufferImpl = static_cast<BufferResourceImpl*>(stagingBuffer);

    void* mappedData = nullptr;
    SLANG_VK_CHECK(api.vkMapMemory(
        api.m_device,
        stagingBufferImpl->m_buffer.m_memory,
        0,
        stagingBufferOffset + size,
        0,
        &mappedData));
    memcpy((char*)mappedData + stagingBufferOffset, data, size);
    api.vkUnmapMemory(api.m_device, stagingBufferImpl->m_buffer.m_memory);

    // Copy from staging buffer to real buffer
    VkBufferCopy copyInfo = {};
    copyInfo.size = size;
    copyInfo.dstOffset = offset;
    copyInfo.srcOffset = stagingBufferOffset;
    api.vkCmdCopyBuffer(
        commandBuffer,
        stagingBufferImpl->m_buffer.m_buffer,
        buffer->m_buffer.m_buffer,
        1,
        &copyInfo);
}

void PipelineCommandEncoder::uploadBufferDataImpl(
    IBufferResource* buffer,
    Offset offset,
    Size size,
    void* data)
{
    m_vkPreCommandBuffer = m_commandBuffer->getPreCommandBuffer();
    _uploadBufferData(
        m_vkPreCommandBuffer,
        m_commandBuffer->m_transientHeap.get(),
        static_cast<BufferResourceImpl*>(buffer),
        offset,
        size,
        data);
}

Result PipelineCommandEncoder::bindRootShaderObjectImpl(
    RootShaderObjectImpl* rootShaderObject,
    VkPipelineBindPoint bindPoint)
{
    // Obtain specialized root layout.
    auto specializedLayout = rootShaderObject->getSpecializedLayout();
    if (!specializedLayout)
        return SLANG_FAIL;

    // We will set up the context required when binding shader objects
    // to the pipeline. Note that this is mostly just being packaged
    // together to minimize the number of parameters that have to
    // be dealt with in the complex recursive call chains.
    //
    RootBindingContext context;
    context.pipelineLayout = specializedLayout->m_pipelineLayout;
    context.device = m_device;
    context.descriptorSetAllocator = &m_commandBuffer->m_transientHeap->m_descSetAllocator;
    context.pushConstantRanges = specializedLayout->getAllPushConstantRanges().getArrayView();

    // The context includes storage for the descriptor sets we will bind,
    // and the number of sets we need to make space for is determined
    // by the specialized program layout.
    //
    List<VkDescriptorSet> descriptorSetsStorage;

    context.descriptorSets = &descriptorSetsStorage;

    // We kick off recursive binding of shader objects to the pipeline (plus
    // the state in `context`).
    //
    // Note: this logic will directly write any push-constant ranges needed,
    // and will also fill in any descriptor sets. Currently it does not
    // *bind* the descriptor sets it fills in.
    //
    // TODO: It could probably bind the descriptor sets as well.
    //
    rootShaderObject->bindAsRoot(this, context, specializedLayout);

    // Once we've filled in all the descriptor sets, we bind them
    // to the pipeline at once.
    //
    if (descriptorSetsStorage.getCount() > 0)
    {
        m_device->m_api.vkCmdBindDescriptorSets(
            m_commandBuffer->m_commandBuffer,
            bindPoint,
            specializedLayout->m_pipelineLayout,
            0,
            (uint32_t)descriptorSetsStorage.getCount(),
            descriptorSetsStorage.getBuffer(),
            0,
            nullptr);
    }

    return SLANG_OK;
}

Result PipelineCommandEncoder::setPipelineStateImpl(
    IPipelineState* state,
    IShaderObject** outRootObject)
{
    m_currentPipeline = static_cast<PipelineStateImpl*>(state);
    m_commandBuffer->m_mutableRootShaderObject = nullptr;
    SLANG_RETURN_ON_FAIL(m_commandBuffer->m_rootObject.init(
        m_commandBuffer->m_renderer,
        m_currentPipeline->getProgram<ShaderProgramImpl>()->m_rootObjectLayout));
    *outRootObject = &m_commandBuffer->m_rootObject;
    return SLANG_OK;
}

Result PipelineCommandEncoder::setPipelineStateWithRootObjectImpl(
    IPipelineState* state,
    IShaderObject* rootObject)
{
    m_currentPipeline = static_cast<PipelineStateImpl*>(state);
    m_commandBuffer->m_mutableRootShaderObject =
        static_cast<MutableRootShaderObjectImpl*>(rootObject);
    return SLANG_OK;
}

Result PipelineCommandEncoder::bindRenderState(VkPipelineBindPoint pipelineBindPoint)
{
    auto& api = *m_api;

    // Get specialized pipeline state and bind it.
    //
    RootShaderObjectImpl* rootObjectImpl = m_commandBuffer->m_mutableRootShaderObject
                                               ? m_commandBuffer->m_mutableRootShaderObject.Ptr()
                                               : &m_commandBuffer->m_rootObject;
    RefPtr<PipelineStateBase> newPipeline;
    SLANG_RETURN_ON_FAIL(
        m_device->maybeSpecializePipeline(m_currentPipeline, rootObjectImpl, newPipeline));
    PipelineStateImpl* newPipelineImpl = static_cast<PipelineStateImpl*>(newPipeline.Ptr());

    SLANG_RETURN_ON_FAIL(newPipelineImpl->ensureAPIPipelineStateCreated());
    m_currentPipeline = newPipelineImpl;

    bindRootShaderObjectImpl(rootObjectImpl, pipelineBindPoint);

    auto pipelineBindPointId = getBindPointIndex(pipelineBindPoint);
    if (m_boundPipelines[pipelineBindPointId] != newPipelineImpl->m_pipeline)
    {
        api.vkCmdBindPipeline(m_vkCommandBuffer, pipelineBindPoint, newPipelineImpl->m_pipeline);
        m_boundPipelines[pipelineBindPointId] = newPipelineImpl->m_pipeline;
    }

    return SLANG_OK;
}

void ResourceCommandEncoder::copyBuffer(
    IBufferResource* dst,
    Offset dstOffset,
    IBufferResource* src,
    Offset srcOffset,
    Size size)
{
    auto& vkAPI = m_commandBuffer->m_renderer->m_api;

    auto dstBuffer = static_cast<BufferResourceImpl*>(dst);
    auto srcBuffer = static_cast<BufferResourceImpl*>(src);

    VkBufferCopy copyRegion;
    copyRegion.dstOffset = dstOffset;
    copyRegion.srcOffset = srcOffset;
    copyRegion.size = size;

    // Note: Vulkan puts the source buffer first in the copy
    // command, going against the dominant tradition for copy
    // operations in C/C++.
    //
    vkAPI.vkCmdCopyBuffer(
        m_commandBuffer->m_commandBuffer,
        srcBuffer->m_buffer.m_buffer,
        dstBuffer->m_buffer.m_buffer,
        /* regionCount: */ 1,
        &copyRegion);
}

void ResourceCommandEncoder::uploadBufferData(
    IBufferResource* buffer,
    Offset offset,
    Size size,
    void* data)
{
    PipelineCommandEncoder::_uploadBufferData(
        m_commandBuffer->m_commandBuffer,
        m_commandBuffer->m_transientHeap.get(),
        static_cast<BufferResourceImpl*>(buffer),
        offset,
        size,
        data);
}

void ResourceCommandEncoder::textureBarrier(
    GfxCount count,
    ITextureResource* const* textures,
    ResourceState src,
    ResourceState dst)
{
    ShortList<VkImageMemoryBarrier, 16> barriers;

    for (GfxIndex i = 0; i < count; i++)
    {
        auto image = static_cast<TextureResourceImpl*>(textures[i]);
        auto desc = image->getDesc();

        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image->m_image;
        barrier.oldLayout = translateImageLayout(src);
        barrier.newLayout = translateImageLayout(dst);
        barrier.subresourceRange.aspectMask =
            getAspectMaskFromFormat(VulkanUtil::getVkFormat(desc->format));
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
        barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        barrier.srcAccessMask = calcAccessFlags(src);
        barrier.dstAccessMask = calcAccessFlags(dst);
        barriers.add(barrier);
    }

    VkPipelineStageFlagBits srcStage = calcPipelineStageFlags(src, true);
    VkPipelineStageFlagBits dstStage = calcPipelineStageFlags(dst, false);

    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    vkApi.vkCmdPipelineBarrier(
        m_commandBuffer->m_commandBuffer,
        srcStage,
        dstStage,
        0,
        0,
        nullptr,
        0,
        nullptr,
        (uint32_t)count,
        barriers.getArrayView().getBuffer());
}

// TODO: Change size_t to Count?
void ResourceCommandEncoder::bufferBarrier(
    GfxCount count,
    IBufferResource* const* buffers,
    ResourceState src,
    ResourceState dst)
{
    List<VkBufferMemoryBarrier> barriers;
    barriers.reserve(count);

    for (GfxIndex i = 0; i < count; i++)
    {
        auto bufferImpl = static_cast<BufferResourceImpl*>(buffers[i]);

        VkBufferMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = calcAccessFlags(src);
        barrier.dstAccessMask = calcAccessFlags(dst);
        barrier.buffer = bufferImpl->m_buffer.m_buffer;
        barrier.offset = 0;
        barrier.size = bufferImpl->getDesc()->sizeInBytes;

        barriers.add(barrier);
    }

    VkPipelineStageFlagBits srcStage = calcPipelineStageFlags(src, true);
    VkPipelineStageFlagBits dstStage = calcPipelineStageFlags(dst, false);

    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    vkApi.vkCmdPipelineBarrier(
        m_commandBuffer->m_commandBuffer,
        srcStage,
        dstStage,
        0,
        0,
        nullptr,
        (uint32_t)count,
        barriers.getBuffer(),
        0,
        nullptr);
}

void ResourceCommandEncoder::endEncoding()
{
    // Insert memory barrier to ensure transfers are visible to the GPU.
    auto& vkAPI = m_commandBuffer->m_renderer->m_api;

    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    vkAPI.vkCmdPipelineBarrier(
        m_commandBuffer->m_commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0,
        1,
        &memBarrier,
        0,
        nullptr,
        0,
        nullptr);
}

void ResourceCommandEncoder::writeTimestamp(IQueryPool* queryPool, GfxIndex index)
{
    _writeTimestamp(
        &m_commandBuffer->m_renderer->m_api,
        m_commandBuffer->m_commandBuffer,
        queryPool,
        index);
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
    auto srcImage = static_cast<TextureResourceImpl*>(src);
    auto srcDesc = srcImage->getDesc();
    auto srcImageLayout = VulkanUtil::getImageLayoutFromState(srcState);
    auto dstImage = static_cast<TextureResourceImpl*>(dst);
    auto dstDesc = dstImage->getDesc();
    auto dstImageLayout = VulkanUtil::getImageLayoutFromState(dstState);
    if (dstSubresource.layerCount == 0 && dstSubresource.mipLevelCount == 0)
    {
        extent = dstDesc->size;
        dstSubresource.layerCount = dstDesc->arraySize;
        if (dstSubresource.layerCount == 0)
            dstSubresource.layerCount = 1;
        dstSubresource.mipLevelCount = dstDesc->numMipLevels;
    }
    if (srcSubresource.layerCount == 0 && srcSubresource.mipLevelCount == 0)
    {
        extent = srcDesc->size;
        srcSubresource.layerCount = srcDesc->arraySize;
        if (srcSubresource.layerCount == 0)
            srcSubresource.layerCount = 1;
        srcSubresource.mipLevelCount = dstDesc->numMipLevels;
    }
    VkImageCopy region = {};
    region.srcSubresource.aspectMask =
        VulkanUtil::getAspectMask(srcSubresource.aspectMask, srcImage->m_vkformat);
    region.srcSubresource.baseArrayLayer = srcSubresource.baseArrayLayer;
    region.srcSubresource.mipLevel = srcSubresource.mipLevel;
    region.srcSubresource.layerCount = srcSubresource.layerCount;
    region.srcOffset = {(int32_t)srcOffset.x, (int32_t)srcOffset.y, (int32_t)srcOffset.z};
    region.dstSubresource.aspectMask =
        VulkanUtil::getAspectMask(dstSubresource.aspectMask, dstImage->m_vkformat);
    region.dstSubresource.baseArrayLayer = dstSubresource.baseArrayLayer;
    region.dstSubresource.mipLevel = dstSubresource.mipLevel;
    region.dstSubresource.layerCount = dstSubresource.layerCount;
    region.dstOffset = {(int32_t)dstOffset.x, (int32_t)dstOffset.y, (int32_t)dstOffset.z};
    region.extent = {(uint32_t)extent.width, (uint32_t)extent.height, (uint32_t)extent.depth};

    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    vkApi.vkCmdCopyImage(
        m_commandBuffer->m_commandBuffer,
        srcImage->m_image,
        srcImageLayout,
        dstImage->m_image,
        dstImageLayout,
        1,
        &region);
}

void ResourceCommandEncoder::uploadTextureData(
    ITextureResource* dst,
    SubresourceRange subResourceRange,
    ITextureResource::Offset3D offset,
    ITextureResource::Extents extend,
    ITextureResource::SubresourceData* subResourceData,
    GfxCount subResourceDataCount)
{
    // VALIDATION: dst must be in TransferDst state.

    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    auto dstImpl = static_cast<TextureResourceImpl*>(dst);
    List<TextureResource::Extents> mipSizes;

    VkCommandBuffer commandBuffer = m_commandBuffer->m_commandBuffer;
    auto& desc = *dstImpl->getDesc();
    // Calculate how large the buffer has to be
    Size bufferSize = 0;
    // Calculate how large an array entry is
    for (GfxIndex j = subResourceRange.mipLevel;
         j < subResourceRange.mipLevel + subResourceRange.mipLevelCount;
         ++j)
    {
        const TextureResource::Extents mipSize = calcMipSize(desc.size, j);

        auto rowSizeInBytes = calcRowSize(desc.format, mipSize.width);
        auto numRows = calcNumRows(desc.format, mipSize.height);

        mipSizes.add(mipSize);

        bufferSize += (rowSizeInBytes * numRows) * mipSize.depth;
    }

    // Calculate the total size taking into account the array
    bufferSize *= subResourceRange.layerCount;

    IBufferResource* uploadBuffer = nullptr;
    Offset uploadBufferOffset = 0;
    m_commandBuffer->m_transientHeap
        ->allocateStagingBuffer(bufferSize, uploadBuffer, uploadBufferOffset, MemoryType::Upload);

    // Copy into upload buffer
    {
        int subResourceCounter = 0;

        uint8_t* dstData;
        uploadBuffer->map(nullptr, (void**)&dstData);
        dstData += uploadBufferOffset;
        uint8_t* dstDataStart;
        dstDataStart = dstData;

        Offset dstSubresourceOffset = 0;
        for (GfxIndex i = 0; i < subResourceRange.layerCount; ++i)
        {
            for (GfxIndex j = 0; j < (GfxCount)mipSizes.getCount(); ++j)
            {
                const auto& mipSize = mipSizes[j];

                int subResourceIndex = subResourceCounter++;
                auto initSubresource = subResourceData[subResourceIndex];

                const ptrdiff_t srcRowStride = (ptrdiff_t)initSubresource.strideY;
                const ptrdiff_t srcLayerStride = (ptrdiff_t)initSubresource.strideZ;

                auto dstRowSizeInBytes = calcRowSize(desc.format, mipSize.width);
                auto numRows = calcNumRows(desc.format, mipSize.height);
                auto dstLayerSizeInBytes = dstRowSizeInBytes * numRows;

                const uint8_t* srcLayer = (const uint8_t*)initSubresource.data;
                uint8_t* dstLayer = dstData + dstSubresourceOffset;

                for (int k = 0; k < mipSize.depth; k++)
                {
                    const uint8_t* srcRow = srcLayer;
                    uint8_t* dstRow = dstLayer;

                    for (GfxCount l = 0; l < numRows; l++)
                    {
                        ::memcpy(dstRow, srcRow, dstRowSizeInBytes);

                        dstRow += dstRowSizeInBytes;
                        srcRow += srcRowStride;
                    }

                    dstLayer += dstLayerSizeInBytes;
                    srcLayer += srcLayerStride;
                }

                dstSubresourceOffset += dstLayerSizeInBytes * mipSize.depth;
            }
        }
        uploadBuffer->unmap(nullptr);
    }
    {
        Offset srcOffset = uploadBufferOffset;
        for (GfxIndex i = 0; i < subResourceRange.layerCount; ++i)
        {
            for (GfxIndex j = 0; j < (GfxCount)mipSizes.getCount(); ++j)
            {
                const auto& mipSize = mipSizes[j];

                auto rowSizeInBytes = calcRowSize(desc.format, mipSize.width);
                auto numRows = calcNumRows(desc.format, mipSize.height);

                // https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkBufferImageCopy.html
                // bufferRowLength and bufferImageHeight specify the data in buffer
                // memory as a subregion of a larger two- or three-dimensional image,
                // and control the addressing calculations of data in buffer memory. If
                // either of these values is zero, that aspect of the buffer memory is
                // considered to be tightly packed according to the imageExtent.

                VkBufferImageCopy region = {};

                region.bufferOffset = srcOffset;
                region.bufferRowLength = 0; // rowSizeInBytes;
                region.bufferImageHeight = 0;

                region.imageSubresource.aspectMask = getAspectMaskFromFormat(dstImpl->m_vkformat);
                region.imageSubresource.mipLevel = subResourceRange.mipLevel + uint32_t(j);
                region.imageSubresource.baseArrayLayer = subResourceRange.baseArrayLayer + i;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = {0, 0, 0};
                region.imageExtent = {
                    uint32_t(mipSize.width),
                    uint32_t(mipSize.height),
                    uint32_t(mipSize.depth)};

                // Do the copy (do all depths in a single go)
                vkApi.vkCmdCopyBufferToImage(
                    commandBuffer,
                    static_cast<BufferResourceImpl*>(uploadBuffer)->m_buffer.m_buffer,
                    dstImpl->m_image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &region);

                // Next
                srcOffset += rowSizeInBytes * numRows * mipSize.depth;
            }
        }
    }
}

void ResourceCommandEncoder::_clearColorImage(
    TextureResourceViewImpl* viewImpl,
    ClearValue* clearValue)
{
    auto& api = m_commandBuffer->m_renderer->m_api;
    auto layout = viewImpl->m_layout;
    if (layout != VK_IMAGE_LAYOUT_GENERAL && layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        m_commandBuffer->m_renderer->_transitionImageLayout(
            m_commandBuffer->m_commandBuffer,
            viewImpl->m_texture->m_image,
            viewImpl->m_texture->m_vkformat,
            *viewImpl->m_texture->getDesc(),
            viewImpl->m_layout,
            layout);
    }

    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseArrayLayer = viewImpl->m_desc.subresourceRange.baseArrayLayer;
    subresourceRange.baseMipLevel = viewImpl->m_desc.subresourceRange.mipLevel;
    subresourceRange.layerCount = viewImpl->m_desc.subresourceRange.layerCount;
    subresourceRange.levelCount = 1;

    VkClearColorValue vkClearColor = {};
    memcpy(vkClearColor.float32, clearValue->color.floatValues, sizeof(float) * 4);

    api.vkCmdClearColorImage(
        m_commandBuffer->m_commandBuffer,
        viewImpl->m_texture->m_image,
        layout,
        &vkClearColor,
        1,
        &subresourceRange);

    if (layout != viewImpl->m_layout)
    {
        m_commandBuffer->m_renderer->_transitionImageLayout(
            m_commandBuffer->m_commandBuffer,
            viewImpl->m_texture->m_image,
            viewImpl->m_texture->m_vkformat,
            *viewImpl->m_texture->getDesc(),
            layout,
            viewImpl->m_layout);
    }
}

void ResourceCommandEncoder::_clearDepthImage(
    TextureResourceViewImpl* viewImpl,
    ClearValue* clearValue,
    ClearResourceViewFlags::Enum flags)
{
    auto& api = m_commandBuffer->m_renderer->m_api;
    auto layout = viewImpl->m_layout;
    if (layout != VK_IMAGE_LAYOUT_GENERAL && layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        m_commandBuffer->m_renderer->_transitionImageLayout(
            m_commandBuffer->m_commandBuffer,
            viewImpl->m_texture->m_image,
            viewImpl->m_texture->m_vkformat,
            *viewImpl->m_texture->getDesc(),
            viewImpl->m_layout,
            layout);
    }

    VkImageSubresourceRange subresourceRange = {};
    if (flags & ClearResourceViewFlags::ClearDepth)
    {
        if (VulkanUtil::isDepthFormat(viewImpl->m_texture->m_vkformat))
        {
            subresourceRange.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
        }
    }
    if (flags & ClearResourceViewFlags::ClearStencil)
    {
        if (VulkanUtil::isStencilFormat(viewImpl->m_texture->m_vkformat))
        {
            subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }
    subresourceRange.baseArrayLayer = viewImpl->m_desc.subresourceRange.baseArrayLayer;
    subresourceRange.baseMipLevel = viewImpl->m_desc.subresourceRange.mipLevel;
    subresourceRange.layerCount = viewImpl->m_desc.subresourceRange.layerCount;
    subresourceRange.levelCount = 1;

    VkClearDepthStencilValue vkClearValue = {};
    vkClearValue.depth = clearValue->depthStencil.depth;
    vkClearValue.stencil = clearValue->depthStencil.stencil;

    api.vkCmdClearDepthStencilImage(
        m_commandBuffer->m_commandBuffer,
        viewImpl->m_texture->m_image,
        layout,
        &vkClearValue,
        1,
        &subresourceRange);

    if (layout != viewImpl->m_layout)
    {
        m_commandBuffer->m_renderer->_transitionImageLayout(
            m_commandBuffer->m_commandBuffer,
            viewImpl->m_texture->m_image,
            viewImpl->m_texture->m_vkformat,
            *viewImpl->m_texture->getDesc(),
            layout,
            viewImpl->m_layout);
    }
}

void ResourceCommandEncoder::_clearBuffer(
    VkBuffer buffer,
    uint64_t bufferSize,
    const IResourceView::Desc& desc,
    uint32_t clearValue)
{
    auto& api = m_commandBuffer->m_renderer->m_api;
    auto clearOffset = desc.bufferRange.offset;
    auto clearSize = desc.bufferRange.size == 0 ? bufferSize - clearOffset : desc.bufferRange.size;
    api.vkCmdFillBuffer(
        m_commandBuffer->m_commandBuffer,
        buffer,
        clearOffset,
        clearSize,
        clearValue);
}

void ResourceCommandEncoder::clearResourceView(
    IResourceView* view,
    ClearValue* clearValue,
    ClearResourceViewFlags::Enum flags)
{
    auto& api = m_commandBuffer->m_renderer->m_api;
    switch (view->getViewDesc()->type)
    {
    case IResourceView::Type::RenderTarget:
        {
            auto viewImpl = static_cast<TextureResourceViewImpl*>(view);
            _clearColorImage(viewImpl, clearValue);
        }
        break;
    case IResourceView::Type::DepthStencil:
        {
            auto viewImpl = static_cast<TextureResourceViewImpl*>(view);
            _clearDepthImage(viewImpl, clearValue, flags);
        }
        break;
    case IResourceView::Type::UnorderedAccess:
        {
            auto viewImplBase = static_cast<ResourceViewImpl*>(view);
            switch (viewImplBase->m_type)
            {
            case ResourceViewImpl::ViewType::Texture:
                {
                    auto viewImpl = static_cast<TextureResourceViewImpl*>(viewImplBase);
                    if ((flags & ClearResourceViewFlags::ClearDepth) ||
                        (flags & ClearResourceViewFlags::ClearStencil))
                    {
                        _clearDepthImage(viewImpl, clearValue, flags);
                    }
                    else
                    {
                        _clearColorImage(viewImpl, clearValue);
                    }
                }
                break;
            case ResourceViewImpl::ViewType::PlainBuffer:
                {
                    assert(
                        clearValue->color.uintValues[1] == clearValue->color.uintValues[0] &&
                        clearValue->color.uintValues[2] == clearValue->color.uintValues[0] &&
                        clearValue->color.uintValues[3] == clearValue->color.uintValues[0]);
                    auto viewImpl = static_cast<PlainBufferResourceViewImpl*>(viewImplBase);
                    uint64_t clearStart = viewImpl->m_desc.bufferRange.offset;
                    uint64_t clearSize = viewImpl->m_desc.bufferRange.size;
                    if (clearSize == 0)
                        clearSize = viewImpl->m_buffer->getDesc()->sizeInBytes - clearStart;
                    api.vkCmdFillBuffer(
                        m_commandBuffer->m_commandBuffer,
                        viewImpl->m_buffer->m_buffer.m_buffer,
                        clearStart,
                        clearSize,
                        clearValue->color.uintValues[0]);
                }
                break;
            case ResourceViewImpl::ViewType::TexelBuffer:
                {
                    assert(
                        clearValue->color.uintValues[1] == clearValue->color.uintValues[0] &&
                        clearValue->color.uintValues[2] == clearValue->color.uintValues[0] &&
                        clearValue->color.uintValues[3] == clearValue->color.uintValues[0]);
                    auto viewImpl = static_cast<TexelBufferResourceViewImpl*>(viewImplBase);
                    _clearBuffer(
                        viewImpl->m_buffer->m_buffer.m_buffer,
                        viewImpl->m_buffer->getDesc()->sizeInBytes,
                        viewImpl->m_desc,
                        clearValue->color.uintValues[0]);
                }
                break;
            }
        }
        break;
    }
}

void ResourceCommandEncoder::resolveResource(
    ITextureResource* source,
    ResourceState sourceState,
    SubresourceRange sourceRange,
    ITextureResource* dest,
    ResourceState destState,
    SubresourceRange destRange)
{
    auto srcTexture = static_cast<TextureResourceImpl*>(source);
    auto srcExtent = srcTexture->getDesc()->size;
    auto dstTexture = static_cast<TextureResourceImpl*>(dest);

    auto srcImage = srcTexture->m_image;
    auto dstImage = dstTexture->m_image;

    auto srcImageLayout = VulkanUtil::getImageLayoutFromState(sourceState);
    auto dstImageLayout = VulkanUtil::getImageLayoutFromState(destState);

    for (GfxIndex layer = 0; layer < sourceRange.layerCount; ++layer)
    {
        for (GfxIndex mip = 0; mip < sourceRange.mipLevelCount; ++mip)
        {
            VkImageResolve region = {};
            region.srcSubresource.aspectMask =
                VulkanUtil::getAspectMask(sourceRange.aspectMask, srcTexture->m_vkformat);
            region.srcSubresource.baseArrayLayer = layer + sourceRange.baseArrayLayer;
            region.srcSubresource.layerCount = 1;
            region.srcSubresource.mipLevel = mip + sourceRange.mipLevel;
            region.srcOffset = {0, 0, 0};
            region.dstSubresource.aspectMask =
                VulkanUtil::getAspectMask(destRange.aspectMask, dstTexture->m_vkformat);
            region.dstSubresource.baseArrayLayer = layer + destRange.baseArrayLayer;
            region.dstSubresource.layerCount = 1;
            region.dstSubresource.mipLevel = mip + destRange.mipLevel;
            region.dstOffset = {0, 0, 0};
            region.extent = {
                (uint32_t)srcExtent.width,
                (uint32_t)srcExtent.height,
                (uint32_t)srcExtent.depth};

            auto& vkApi = m_commandBuffer->m_renderer->m_api;
            vkApi.vkCmdResolveImage(
                m_commandBuffer->m_commandBuffer,
                srcImage,
                srcImageLayout,
                dstImage,
                dstImageLayout,
                1,
                &region);
        }
    }
}

void ResourceCommandEncoder::resolveQuery(
    IQueryPool* queryPool,
    GfxIndex index,
    GfxCount count,
    IBufferResource* buffer,
    Offset offset)
{
    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    auto poolImpl = static_cast<QueryPoolImpl*>(queryPool);
    auto bufferImpl = static_cast<BufferResourceImpl*>(buffer);
    vkApi.vkCmdCopyQueryPoolResults(
        m_commandBuffer->m_commandBuffer,
        poolImpl->m_pool,
        index,
        count,
        bufferImpl->m_buffer.m_buffer,
        offset,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
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

    auto image = static_cast<TextureResourceImpl*>(src);
    auto desc = image->getDesc();
    auto buffer = static_cast<BufferResourceImpl*>(dst);
    auto srcImageLayout = VulkanUtil::getImageLayoutFromState(srcState);

    VkBufferImageCopy region = {};
    region.bufferOffset = dstOffset;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask =
        VulkanUtil::getAspectMask(srcSubresource.aspectMask, image->m_vkformat);
    region.imageSubresource.mipLevel = srcSubresource.mipLevel;
    region.imageSubresource.baseArrayLayer = srcSubresource.baseArrayLayer;
    region.imageSubresource.layerCount = srcSubresource.layerCount;
    region.imageOffset = {(int32_t)srcOffset.x, (int32_t)srcOffset.y, (int32_t)srcOffset.z};
    region.imageExtent = {uint32_t(extent.width), uint32_t(extent.height), uint32_t(extent.depth)};

    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    vkApi.vkCmdCopyImageToBuffer(
        m_commandBuffer->m_commandBuffer,
        image->m_image,
        srcImageLayout,
        buffer->m_buffer.m_buffer,
        1,
        &region);
}

void ResourceCommandEncoder::textureSubresourceBarrier(
    ITextureResource* texture,
    SubresourceRange subresourceRange,
    ResourceState src,
    ResourceState dst)
{
    ShortList<VkImageMemoryBarrier> barriers;
    auto image = static_cast<TextureResourceImpl*>(texture);
    auto desc = image->getDesc();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image->m_image;
    barrier.oldLayout = translateImageLayout(src);
    barrier.newLayout = translateImageLayout(dst);
    barrier.subresourceRange.aspectMask =
        VulkanUtil::getAspectMask(subresourceRange.aspectMask, image->m_vkformat);
    barrier.subresourceRange.baseArrayLayer = subresourceRange.baseArrayLayer;
    barrier.subresourceRange.baseMipLevel = subresourceRange.mipLevel;
    barrier.subresourceRange.layerCount = subresourceRange.layerCount;
    barrier.subresourceRange.levelCount = subresourceRange.mipLevelCount;
    barrier.srcAccessMask = calcAccessFlags(src);
    barrier.dstAccessMask = calcAccessFlags(dst);
    barriers.add(barrier);

    VkPipelineStageFlagBits srcStage = calcPipelineStageFlags(src, true);
    VkPipelineStageFlagBits dstStage = calcPipelineStageFlags(dst, false);

    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    vkApi.vkCmdPipelineBarrier(
        m_commandBuffer->m_commandBuffer,
        srcStage,
        dstStage,
        0,
        0,
        nullptr,
        0,
        nullptr,
        (uint32_t)barriers.getCount(),
        barriers.getArrayView().getBuffer());
}

void ResourceCommandEncoder::beginDebugEvent(const char* name, float rgbColor[3])
{
    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    if (vkApi.vkCmdDebugMarkerBeginEXT)
    {
        VkDebugMarkerMarkerInfoEXT eventInfo = {};
        eventInfo.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT;
        eventInfo.pMarkerName = name;
        eventInfo.color[0] = rgbColor[0];
        eventInfo.color[1] = rgbColor[1];
        eventInfo.color[2] = rgbColor[2];
        eventInfo.color[3] = 1.0f;
        vkApi.vkCmdDebugMarkerBeginEXT(m_commandBuffer->m_commandBuffer, &eventInfo);
    }
}

void ResourceCommandEncoder::endDebugEvent()
{
    auto& vkApi = m_commandBuffer->m_renderer->m_api;
    if (vkApi.vkCmdDebugMarkerEndEXT)
    {
        vkApi.vkCmdDebugMarkerEndEXT(m_commandBuffer->m_commandBuffer);
    }
}

void RenderCommandEncoder::beginPass(IRenderPassLayout* renderPass, IFramebuffer* framebuffer)
{
    FramebufferImpl* framebufferImpl = static_cast<FramebufferImpl*>(framebuffer);
    if (!framebuffer)
        framebufferImpl = this->m_device->m_emptyFramebuffer;
    RenderPassLayoutImpl* renderPassImpl = static_cast<RenderPassLayoutImpl*>(renderPass);
    VkClearValue clearValues[kMaxTargets] = {};
    VkRenderPassBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    beginInfo.framebuffer = framebufferImpl->m_handle;
    beginInfo.renderPass = renderPassImpl->m_renderPass;
    uint32_t targetCount = (uint32_t)framebufferImpl->renderTargetViews.getCount();
    if (framebufferImpl->depthStencilView)
        targetCount++;
    beginInfo.clearValueCount = targetCount;
    beginInfo.renderArea.extent.width = framebufferImpl->m_width;
    beginInfo.renderArea.extent.height = framebufferImpl->m_height;
    beginInfo.pClearValues = framebufferImpl->m_clearValues;
    auto& api = *m_api;
    api.vkCmdBeginRenderPass(m_vkCommandBuffer, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void RenderCommandEncoder::endEncoding()
{
    auto& api = *m_api;
    api.vkCmdEndRenderPass(m_vkCommandBuffer);
    endEncodingImpl();
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
    return setPipelineStateWithRootObjectImpl(pipelineState, rootObject);
}

void RenderCommandEncoder::setViewports(GfxCount count, const Viewport* viewports)
{
    static const int kMaxViewports = 8; // TODO: base on device caps
    assert(count <= kMaxViewports);

    m_viewports.setCount(count);
    for (GfxIndex ii = 0; ii < count; ++ii)
    {
        auto& inViewport = viewports[ii];
        auto& vkViewport = m_viewports[ii];

        vkViewport.x = inViewport.originX;
        vkViewport.y = inViewport.originY + inViewport.extentY;
        vkViewport.width = inViewport.extentX;
        vkViewport.height = -inViewport.extentY;
        vkViewport.minDepth = inViewport.minZ;
        vkViewport.maxDepth = inViewport.maxZ;
    }

    auto& api = *m_api;
    api.vkCmdSetViewport(m_vkCommandBuffer, 0, uint32_t(count), m_viewports.getBuffer());
}

void RenderCommandEncoder::setScissorRects(GfxCount count, const ScissorRect* rects)
{
    static const int kMaxScissorRects = 8; // TODO: base on device caps
    assert(count <= kMaxScissorRects);

    m_scissorRects.setCount(count);
    for (GfxIndex ii = 0; ii < count; ++ii)
    {
        auto& inRect = rects[ii];
        auto& vkRect = m_scissorRects[ii];

        vkRect.offset.x = int32_t(inRect.minX);
        vkRect.offset.y = int32_t(inRect.minY);
        vkRect.extent.width = uint32_t(inRect.maxX - inRect.minX);
        vkRect.extent.height = uint32_t(inRect.maxY - inRect.minY);
    }

    auto& api = *m_api;
    api.vkCmdSetScissor(m_vkCommandBuffer, 0, uint32_t(count), m_scissorRects.getBuffer());
}

void RenderCommandEncoder::setPrimitiveTopology(PrimitiveTopology topology)
{
    auto& api = *m_api;
    if (api.vkCmdSetPrimitiveTopologyEXT)
    {
        api.vkCmdSetPrimitiveTopologyEXT(
            m_vkCommandBuffer,
            VulkanUtil::getVkPrimitiveTopology(topology));
    }
    else
    {
        switch (topology)
        {
        case PrimitiveTopology::TriangleList:
            break;
        default:
            // We are using a non-list topology, but we don't have dynmaic state
            // extension, error out.
            assert(!"Non-list topology requires VK_EXT_extended_dynamic_states, which "
                    "is not present.");
            break;
        }
    }
}

void RenderCommandEncoder::setVertexBuffers(
    GfxIndex startSlot,
    GfxCount slotCount,
    IBufferResource* const* buffers,
    const Offset* offsets)
{
    for (GfxIndex i = 0; i < GfxIndex(slotCount); i++)
    {
        GfxIndex slotIndex = startSlot + i;
        BufferResourceImpl* buffer = static_cast<BufferResourceImpl*>(buffers[i]);
        if (buffer)
        {
            VkBuffer vertexBuffers[] = {buffer->m_buffer.m_buffer};
            VkDeviceSize offset = VkDeviceSize(offsets[i]);

            m_api->vkCmdBindVertexBuffers(
                m_vkCommandBuffer,
                (uint32_t)slotIndex,
                1,
                vertexBuffers,
                &offset);
        }
    }
}

void RenderCommandEncoder::setIndexBuffer(
    IBufferResource* buffer,
    Format indexFormat,
    Offset offset)
{
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    switch (indexFormat)
    {
    case Format::R16_UINT:
        indexType = VK_INDEX_TYPE_UINT16;
        break;
    case Format::R32_UINT:
        indexType = VK_INDEX_TYPE_UINT32;
        break;
    default:
        assert(!"unsupported index format");
    }

    BufferResourceImpl* bufferImpl = static_cast<BufferResourceImpl*>(buffer);

    m_api->vkCmdBindIndexBuffer(
        m_vkCommandBuffer,
        bufferImpl->m_buffer.m_buffer,
        (VkDeviceSize)offset,
        indexType);
}

Result RenderCommandEncoder::prepareDraw()
{
    auto pipeline = static_cast<PipelineStateImpl*>(m_currentPipeline.Ptr());
    if (!pipeline)
    {
        return SLANG_FAIL;
    }
    SLANG_RETURN_ON_FAIL(bindRenderState(VK_PIPELINE_BIND_POINT_GRAPHICS));
    return SLANG_OK;
}

Result RenderCommandEncoder::draw(GfxCount vertexCount, GfxIndex startVertex)
{
    SLANG_RETURN_ON_FAIL(prepareDraw());
    auto& api = *m_api;
    api.vkCmdDraw(m_vkCommandBuffer, vertexCount, 1, 0, 0);
    return SLANG_OK;
}

Result RenderCommandEncoder::drawIndexed(
    GfxCount indexCount,
    GfxIndex startIndex,
    GfxIndex baseVertex)
{
    SLANG_RETURN_ON_FAIL(prepareDraw());
    auto& api = *m_api;
    api.vkCmdDrawIndexed(m_vkCommandBuffer, indexCount, 1, startIndex, baseVertex, 0);
    return SLANG_OK;
}

void RenderCommandEncoder::setStencilReference(uint32_t referenceValue)
{
    auto& api = *m_api;
    api.vkCmdSetStencilReference(m_vkCommandBuffer, VK_STENCIL_FRONT_AND_BACK, referenceValue);
}

Result RenderCommandEncoder::drawIndirect(
    GfxCount maxDrawCount,
    IBufferResource* argBuffer,
    Offset argOffset,
    IBufferResource* countBuffer,
    Offset countOffset)
{
    SLANG_RETURN_ON_FAIL(prepareDraw());
    auto& api = *m_api;
    auto argBufferImpl = static_cast<BufferResourceImpl*>(argBuffer);

    if (countBuffer)
    {
        auto countBufferImpl = static_cast<BufferResourceImpl*>(countBuffer);
        api.vkCmdDrawIndirectCount(
            m_vkCommandBuffer,
            argBufferImpl->m_buffer.m_buffer,
            argOffset,
            countBufferImpl->m_buffer.m_buffer,
            countOffset,
            maxDrawCount,
            sizeof(VkDrawIndirectCommand));
    }
    else
    {
        api.vkCmdDrawIndirect(
            m_vkCommandBuffer,
            argBufferImpl->m_buffer.m_buffer,
            argOffset,
            maxDrawCount,
            sizeof(VkDrawIndirectCommand));
    }
    return SLANG_OK;
}

Result RenderCommandEncoder::drawIndexedIndirect(
    GfxCount maxDrawCount,
    IBufferResource* argBuffer,
    Offset argOffset,
    IBufferResource* countBuffer,
    Offset countOffset)
{
    SLANG_RETURN_ON_FAIL(prepareDraw());
    auto& api = *m_api;
    auto argBufferImpl = static_cast<BufferResourceImpl*>(argBuffer);

    if (countBuffer)
    {
        auto countBufferImpl = static_cast<BufferResourceImpl*>(countBuffer);
        api.vkCmdDrawIndexedIndirectCount(
            m_vkCommandBuffer,
            argBufferImpl->m_buffer.m_buffer,
            argOffset,
            countBufferImpl->m_buffer.m_buffer,
            countOffset,
            maxDrawCount,
            sizeof(VkDrawIndexedIndirectCommand));
    }
    else
    {
        api.vkCmdDrawIndexedIndirect(
            m_vkCommandBuffer,
            argBufferImpl->m_buffer.m_buffer,
            argOffset,
            maxDrawCount,
            sizeof(VkDrawIndexedIndirectCommand));
    }
    return SLANG_OK;
}

Result RenderCommandEncoder::setSamplePositions(
    GfxCount samplesPerPixel,
    GfxCount pixelCount,
    const SamplePosition* samplePositions)
{
    if (m_api->vkCmdSetSampleLocationsEXT)
    {
        VkSampleLocationsInfoEXT sampleLocInfo = {};
        sampleLocInfo.sType = VK_STRUCTURE_TYPE_SAMPLE_LOCATIONS_INFO_EXT;
        sampleLocInfo.sampleLocationsCount = samplesPerPixel * pixelCount;
        sampleLocInfo.sampleLocationsPerPixel = (VkSampleCountFlagBits)samplesPerPixel;
        m_api->vkCmdSetSampleLocationsEXT(m_vkCommandBuffer, &sampleLocInfo);
        return SLANG_OK;
    }
    return SLANG_E_NOT_AVAILABLE;
}

Result RenderCommandEncoder::drawInstanced(
    GfxCount vertexCount,
    GfxCount instanceCount,
    GfxIndex startVertex,
    GfxIndex startInstanceLocation)
{
    SLANG_RETURN_ON_FAIL(prepareDraw());
    auto& api = *m_api;
    api.vkCmdDraw(
        m_vkCommandBuffer,
        vertexCount,
        instanceCount,
        startVertex,
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
    SLANG_RETURN_ON_FAIL(prepareDraw());
    auto& api = *m_api;
    api.vkCmdDrawIndexed(
        m_vkCommandBuffer,
        indexCount,
        instanceCount,
        startIndexLocation,
        baseVertexLocation,
        startInstanceLocation);
    return SLANG_OK;
}

Result RenderCommandEncoder::drawMeshTasks(int x, int y, int z)
{
    SLANG_RETURN_ON_FAIL(prepareDraw());
    auto& api = *m_api;
    api.vkCmdDrawMeshTasksEXT(m_vkCommandBuffer, x, y, z);
    return SLANG_OK;
}

void ComputeCommandEncoder::endEncoding()
{
    endEncodingImpl();
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
    return setPipelineStateWithRootObjectImpl(pipelineState, rootObject);
}

Result ComputeCommandEncoder::dispatchCompute(int x, int y, int z)
{
    auto pipeline = static_cast<PipelineStateImpl*>(m_currentPipeline.Ptr());
    if (!pipeline)
    {
        return SLANG_FAIL;
    }

    // Also create descriptor sets based on the given pipeline layout
    SLANG_RETURN_ON_FAIL(bindRenderState(VK_PIPELINE_BIND_POINT_COMPUTE));
    m_api->vkCmdDispatch(m_vkCommandBuffer, x, y, z);
    return SLANG_OK;
}

Result ComputeCommandEncoder::dispatchComputeIndirect(IBufferResource* argBuffer, Offset offset)
{
    auto pipeline = static_cast<PipelineStateImpl*>(m_currentPipeline.Ptr());
    if (!pipeline)
    {
        return SLANG_FAIL;
    }

    // Also create descriptor sets based on the given pipeline layout
    SLANG_RETURN_ON_FAIL(bindRenderState(VK_PIPELINE_BIND_POINT_COMPUTE));
    auto argBufferImpl = static_cast<BufferResourceImpl*>(argBuffer);
    m_api->vkCmdDispatchIndirect(m_vkCommandBuffer, argBufferImpl->m_buffer.m_buffer, offset);
    return SLANG_OK;
}

void RayTracingCommandEncoder::_memoryBarrier(
    int count,
    IAccelerationStructure* const* structures,
    AccessFlag srcAccess,
    AccessFlag destAccess)
{
    ShortList<VkBufferMemoryBarrier> memBarriers;
    memBarriers.setCount(count);
    for (int i = 0; i < count; i++)
    {
        memBarriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        memBarriers[i].pNext = nullptr;
        memBarriers[i].dstAccessMask = translateAccelerationStructureAccessFlag(destAccess);
        memBarriers[i].srcAccessMask = translateAccelerationStructureAccessFlag(srcAccess);
        memBarriers[i].srcQueueFamilyIndex = m_commandBuffer->m_renderer->m_queueFamilyIndex;
        memBarriers[i].dstQueueFamilyIndex = m_commandBuffer->m_renderer->m_queueFamilyIndex;

        auto asImpl = static_cast<AccelerationStructureImpl*>(structures[i]);
        memBarriers[i].buffer = asImpl->m_buffer->m_buffer.m_buffer;
        memBarriers[i].offset = asImpl->m_offset;
        memBarriers[i].size = asImpl->m_size;
    }
    m_commandBuffer->m_renderer->m_api.vkCmdPipelineBarrier(
        m_commandBuffer->m_commandBuffer,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
        0,
        0,
        nullptr,
        (uint32_t)memBarriers.getCount(),
        memBarriers.getArrayView().getBuffer(),
        0,
        nullptr);
}

void RayTracingCommandEncoder::_queryAccelerationStructureProperties(
    GfxCount accelerationStructureCount,
    IAccelerationStructure* const* accelerationStructures,
    GfxCount queryCount,
    AccelerationStructureQueryDesc* queryDescs)
{
    ShortList<VkAccelerationStructureKHR> vkHandles;
    vkHandles.setCount(accelerationStructureCount);
    for (GfxIndex i = 0; i < accelerationStructureCount; i++)
    {
        vkHandles[i] =
            static_cast<AccelerationStructureImpl*>(accelerationStructures[i])->m_vkHandle;
    }
    auto vkHandlesView = vkHandles.getArrayView();
    for (GfxIndex i = 0; i < queryCount; i++)
    {
        VkQueryType queryType;
        switch (queryDescs[i].queryType)
        {
        case QueryType::AccelerationStructureCompactedSize:
            queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
            break;
        case QueryType::AccelerationStructureSerializedSize:
            queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
            break;
        case QueryType::AccelerationStructureCurrentSize:
            continue;
        default:
            getDebugCallback()->handleMessage(
                DebugMessageType::Error,
                DebugMessageSource::Layer,
                "Invalid query type for use in queryAccelerationStructureProperties.");
            return;
        }
        auto queryPool = static_cast<QueryPoolImpl*>(queryDescs[i].queryPool)->m_pool;
        m_commandBuffer->m_renderer->m_api.vkCmdResetQueryPool(
            m_commandBuffer->m_commandBuffer,
            queryPool,
            (uint32_t)queryDescs[i].firstQueryIndex,
            1);
        m_commandBuffer->m_renderer->m_api.vkCmdWriteAccelerationStructuresPropertiesKHR(
            m_commandBuffer->m_commandBuffer,
            accelerationStructureCount,
            vkHandlesView.getBuffer(),
            queryType,
            queryPool,
            queryDescs[i].firstQueryIndex);
    }
}

void RayTracingCommandEncoder::buildAccelerationStructure(
    const IAccelerationStructure::BuildDesc& desc,
    GfxCount propertyQueryCount,
    AccelerationStructureQueryDesc* queryDescs)
{
    AccelerationStructureBuildGeometryInfoBuilder geomInfoBuilder;
    if (geomInfoBuilder.build(desc.inputs, getDebugCallback()) != SLANG_OK)
        return;

    if (desc.dest)
    {
        geomInfoBuilder.buildInfo.dstAccelerationStructure =
            static_cast<AccelerationStructureImpl*>(desc.dest)->m_vkHandle;
    }
    if (desc.source)
    {
        geomInfoBuilder.buildInfo.srcAccelerationStructure =
            static_cast<AccelerationStructureImpl*>(desc.source)->m_vkHandle;
    }
    geomInfoBuilder.buildInfo.scratchData.deviceAddress = desc.scratchData;

    List<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
    rangeInfos.setCount(geomInfoBuilder.primitiveCounts.getCount());
    for (Index i = 0; i < geomInfoBuilder.primitiveCounts.getCount(); i++)
    {
        auto& rangeInfo = rangeInfos[i];
        rangeInfo.primitiveCount = geomInfoBuilder.primitiveCounts[i];
        rangeInfo.firstVertex = 0;
        rangeInfo.primitiveOffset = 0;
        rangeInfo.transformOffset = 0;
    }

    auto rangeInfoPtr = rangeInfos.getBuffer();
    m_commandBuffer->m_renderer->m_api.vkCmdBuildAccelerationStructuresKHR(
        m_commandBuffer->m_commandBuffer,
        1,
        &geomInfoBuilder.buildInfo,
        &rangeInfoPtr);

    if (propertyQueryCount)
    {
        _memoryBarrier(1, &desc.dest, AccessFlag::Write, AccessFlag::Read);
        _queryAccelerationStructureProperties(1, &desc.dest, propertyQueryCount, queryDescs);
    }
}

void RayTracingCommandEncoder::copyAccelerationStructure(
    IAccelerationStructure* dest,
    IAccelerationStructure* src,
    AccelerationStructureCopyMode mode)
{
    VkCopyAccelerationStructureInfoKHR copyInfo = {
        VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
    copyInfo.src = static_cast<AccelerationStructureImpl*>(src)->m_vkHandle;
    copyInfo.dst = static_cast<AccelerationStructureImpl*>(dest)->m_vkHandle;
    switch (mode)
    {
    case AccelerationStructureCopyMode::Clone:
        copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
        break;
    case AccelerationStructureCopyMode::Compact:
        copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
        break;
    default:
        getDebugCallback()->handleMessage(
            DebugMessageType::Error,
            DebugMessageSource::Layer,
            "Unsupported AccelerationStructureCopyMode.");
        return;
    }
    m_commandBuffer->m_renderer->m_api.vkCmdCopyAccelerationStructureKHR(
        m_commandBuffer->m_commandBuffer,
        &copyInfo);
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
    VkCopyAccelerationStructureToMemoryInfoKHR copyInfo = {
        VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR};
    copyInfo.src = static_cast<AccelerationStructureImpl*>(source)->m_vkHandle;
    copyInfo.dst.deviceAddress = dest;
    copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR;
    m_commandBuffer->m_renderer->m_api.vkCmdCopyAccelerationStructureToMemoryKHR(
        m_commandBuffer->m_commandBuffer,
        &copyInfo);
}

void RayTracingCommandEncoder::deserializeAccelerationStructure(
    IAccelerationStructure* dest,
    DeviceAddress source)
{
    VkCopyMemoryToAccelerationStructureInfoKHR copyInfo = {
        VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR};
    copyInfo.src.deviceAddress = source;
    copyInfo.dst = static_cast<AccelerationStructureImpl*>(dest)->m_vkHandle;
    copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;
    m_commandBuffer->m_renderer->m_api.vkCmdCopyMemoryToAccelerationStructureKHR(
        m_commandBuffer->m_commandBuffer,
        &copyInfo);
}

Result RayTracingCommandEncoder::bindPipeline(
    IPipelineState* pipeline,
    IShaderObject** outRootObject)
{
    return setPipelineStateImpl(pipeline, outRootObject);
}

Result RayTracingCommandEncoder::bindPipelineWithRootObject(
    IPipelineState* pipelineState,
    IShaderObject* rootObject)
{
    return setPipelineStateWithRootObjectImpl(pipelineState, rootObject);
}

Result RayTracingCommandEncoder::dispatchRays(
    GfxIndex raygenShaderIndex,
    IShaderTable* shaderTable,
    GfxCount width,
    GfxCount height,
    GfxCount depth)
{
    auto vkApi = m_commandBuffer->m_renderer->m_api;
    auto vkCommandBuffer = m_commandBuffer->m_commandBuffer;

    SLANG_RETURN_ON_FAIL(bindRenderState(VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR));

    auto rtProps = vkApi.m_rtProperties;
    auto shaderTableImpl = (ShaderTableImpl*)shaderTable;
    auto alignedHandleSize =
        VulkanUtil::calcAligned(rtProps.shaderGroupHandleSize, rtProps.shaderGroupHandleAlignment);

    auto shaderTableBuffer = shaderTableImpl->getOrCreateBuffer(
        m_currentPipeline,
        m_commandBuffer->m_transientHeap,
        static_cast<ResourceCommandEncoder*>(this));
    auto shaderTableAddr = shaderTableBuffer->getDeviceAddress();

    VkStridedDeviceAddressRegionKHR raygenSBT;
    raygenSBT.stride = VulkanUtil::calcAligned(alignedHandleSize, rtProps.shaderGroupBaseAlignment);
    raygenSBT.deviceAddress = shaderTableAddr + raygenShaderIndex * raygenSBT.stride;
    raygenSBT.size = raygenSBT.stride;

    VkStridedDeviceAddressRegionKHR missSBT;
    missSBT.deviceAddress = shaderTableAddr + shaderTableImpl->m_raygenTableSize;
    missSBT.stride = alignedHandleSize;
    missSBT.size = shaderTableImpl->m_missTableSize;

    VkStridedDeviceAddressRegionKHR hitSBT;
    hitSBT.deviceAddress = missSBT.deviceAddress + missSBT.size;
    hitSBT.stride = alignedHandleSize;
    hitSBT.size = shaderTableImpl->m_hitTableSize;

    VkStridedDeviceAddressRegionKHR callableSBT;
    callableSBT.deviceAddress = hitSBT.deviceAddress + hitSBT.size;
    callableSBT.stride = alignedHandleSize;
    callableSBT.size = shaderTableImpl->m_callableTableSize;

    vkApi.vkCmdTraceRaysKHR(
        vkCommandBuffer,
        &raygenSBT,
        &missSBT,
        &hitSBT,
        &callableSBT,
        (uint32_t)width,
        (uint32_t)height,
        (uint32_t)depth);

    return SLANG_OK;
}

void RayTracingCommandEncoder::endEncoding()
{
    endEncodingImpl();
}

} // namespace vk
} // namespace gfx
