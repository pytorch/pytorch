#include "core/slang-basic.h"
#include "gfx-test-texture-util.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

#if SLANG_WINDOWS_FAMILY
#include <d3d12.h>
#endif

using namespace Slang;
using namespace gfx;

namespace gfx_test
{
struct TextureToTextureCopyInfo
{
    SubresourceRange srcSubresource;
    SubresourceRange dstSubresource;
    ITextureResource::Extents extent;
    ITextureResource::Offset3D srcOffset;
    ITextureResource::Offset3D dstOffset;
};

struct TextureToBufferCopyInfo
{
    SubresourceRange srcSubresource;
    ITextureResource::Extents extent;
    ITextureResource::Offset3D textureOffset;
    Offset bufferOffset;
    Offset bufferSize;
};

struct BaseCopyTextureTest
{
    IDevice* device;
    UnitTestContext* context;

    Size alignedRowStride;

    RefPtr<TextureInfo> srcTextureInfo;
    RefPtr<TextureInfo> dstTextureInfo;
    TextureToTextureCopyInfo texCopyInfo;
    TextureToBufferCopyInfo bufferCopyInfo;

    ComPtr<ITextureResource> srcTexture;
    ComPtr<ITextureResource> dstTexture;
    ComPtr<IBufferResource> resultsBuffer;

    RefPtr<ValidationTextureFormatBase> validationFormat;

    void init(
        IDevice* device,
        UnitTestContext* context,
        Format format,
        RefPtr<ValidationTextureFormatBase> validationFormat,
        ITextureResource::Type type)
    {
        this->device = device;
        this->context = context;
        this->validationFormat = validationFormat;

        this->srcTextureInfo = new TextureInfo();
        this->srcTextureInfo->format = format;
        this->srcTextureInfo->textureType = type;

        this->dstTextureInfo = new TextureInfo();
        this->dstTextureInfo->format = format;
        this->dstTextureInfo->textureType = type;
    }

    void createRequiredResources()
    {
        ITextureResource::Desc srcTexDesc = {};
        srcTexDesc.type = srcTextureInfo->textureType;
        srcTexDesc.numMipLevels = srcTextureInfo->mipLevelCount;
        srcTexDesc.arraySize = srcTextureInfo->arrayLayerCount;
        srcTexDesc.size = srcTextureInfo->extents;
        srcTexDesc.defaultState = ResourceState::ShaderResource;
        srcTexDesc.allowedStates =
            ResourceStateSet(ResourceState::ShaderResource, ResourceState::CopySource);
        if (srcTextureInfo->format == Format::D32_FLOAT ||
            srcTextureInfo->format == Format::D16_UNORM)
        {
            srcTexDesc.allowedStates.add(ResourceState::DepthWrite);
            srcTexDesc.allowedStates.add(ResourceState::DepthRead);
        }
        srcTexDesc.format = srcTextureInfo->format;

        GFX_CHECK_CALL_ABORT(device->createTextureResource(
            srcTexDesc,
            srcTextureInfo->subresourceDatas.getBuffer(),
            srcTexture.writeRef()));

        ITextureResource::Desc dstTexDesc = {};
        dstTexDesc.type = dstTextureInfo->textureType;
        dstTexDesc.numMipLevels = dstTextureInfo->mipLevelCount;
        dstTexDesc.arraySize = dstTextureInfo->arrayLayerCount;
        dstTexDesc.size = dstTextureInfo->extents;
        dstTexDesc.defaultState = ResourceState::CopyDestination;
        dstTexDesc.allowedStates = ResourceStateSet(
            ResourceState::ShaderResource,
            ResourceState::CopyDestination,
            ResourceState::CopySource);
        if (dstTextureInfo->format == Format::D32_FLOAT ||
            dstTextureInfo->format == Format::D16_UNORM)
        {
            dstTexDesc.allowedStates.add(ResourceState::DepthWrite);
            dstTexDesc.allowedStates.add(ResourceState::DepthRead);
        }
        dstTexDesc.format = dstTextureInfo->format;

        GFX_CHECK_CALL_ABORT(device->createTextureResource(
            dstTexDesc,
            dstTextureInfo->subresourceDatas.getBuffer(),
            dstTexture.writeRef()));

        auto bufferCopyExtents = bufferCopyInfo.extent;
        auto texelSize = getTexelSize(dstTextureInfo->format);
        size_t alignment;
        device->getTextureRowAlignment(&alignment);
        alignedRowStride = (bufferCopyExtents.width * texelSize + alignment - 1) & ~(alignment - 1);
        IBufferResource::Desc bufferDesc = {};
        bufferDesc.sizeInBytes =
            bufferCopyExtents.height * bufferCopyExtents.depth * alignedRowStride;
        bufferDesc.format = Format::Unknown;
        bufferDesc.elementSize = 0;
        bufferDesc.allowedStates = ResourceStateSet(
            ResourceState::ShaderResource,
            ResourceState::UnorderedAccess,
            ResourceState::CopyDestination,
            ResourceState::CopySource);
        bufferDesc.defaultState = ResourceState::CopyDestination;
        bufferDesc.memoryType = MemoryType::DeviceLocal;

        GFX_CHECK_CALL_ABORT(
            device->createBufferResource(bufferDesc, nullptr, resultsBuffer.writeRef()));

        bufferCopyInfo.bufferSize = bufferDesc.sizeInBytes;
    }

    void submitGPUWork()
    {
        Slang::ComPtr<ITransientResourceHeap> transientHeap;
        ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096;
        GFX_CHECK_CALL_ABORT(
            device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = device->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        auto encoder = commandBuffer->encodeResourceCommands();

        encoder->textureSubresourceBarrier(
            srcTexture,
            texCopyInfo.srcSubresource,
            ResourceState::ShaderResource,
            ResourceState::CopySource);
        encoder->copyTexture(
            dstTexture,
            ResourceState::CopyDestination,
            texCopyInfo.dstSubresource,
            texCopyInfo.dstOffset,
            srcTexture,
            ResourceState::CopySource,
            texCopyInfo.srcSubresource,
            texCopyInfo.srcOffset,
            texCopyInfo.extent);

        encoder->textureSubresourceBarrier(
            dstTexture,
            bufferCopyInfo.srcSubresource,
            ResourceState::CopyDestination,
            ResourceState::CopySource);
        encoder->copyTextureToBuffer(
            resultsBuffer,
            bufferCopyInfo.bufferOffset,
            bufferCopyInfo.bufferSize,
            alignedRowStride,
            dstTexture,
            ResourceState::CopySource,
            bufferCopyInfo.srcSubresource,
            bufferCopyInfo.textureOffset,
            bufferCopyInfo.extent);

        encoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    bool isWithinCopyBounds(GfxIndex x, GfxIndex y, GfxIndex z)
    {
        auto copyExtents = texCopyInfo.extent;
        auto copyOffset = texCopyInfo.dstOffset;

        auto xLowerBound = copyOffset.x;
        auto xUpperBound = copyOffset.x + copyExtents.width;
        auto yLowerBound = copyOffset.y;
        auto yUpperBound = copyOffset.y + copyExtents.height;
        auto zLowerBound = copyOffset.z;
        auto zUpperBound = copyOffset.z + copyExtents.depth;

        if (x < xLowerBound || x >= xUpperBound || y < yLowerBound || y >= yUpperBound ||
            z < zLowerBound || z >= zUpperBound)
            return false;
        else
            return true;
    }

    void validateTestResults(
        ValidationTextureData actual,
        ValidationTextureData expectedCopied,
        ValidationTextureData expectedOriginal)
    {
        auto actualExtents = actual.extents;
        auto copyExtent = texCopyInfo.extent;
        auto srcTexOffset = texCopyInfo.srcOffset;
        auto dstTexOffset = texCopyInfo.dstOffset;

        for (GfxIndex x = 0; x < actualExtents.width; ++x)
        {
            for (GfxIndex y = 0; y < actualExtents.height; ++y)
            {
                for (GfxIndex z = 0; z < actualExtents.depth; ++z)
                {
                    auto actualBlock = actual.getBlockAt(x, y, z);
                    if (isWithinCopyBounds(x, y, z))
                    {
                        // Block is located within the bounds of the source texture
                        auto xSource = x + srcTexOffset.x - dstTexOffset.x;
                        auto ySource = y + srcTexOffset.y - dstTexOffset.y;
                        auto zSource = z + srcTexOffset.z - dstTexOffset.z;
                        auto expectedBlock = expectedCopied.getBlockAt(xSource, ySource, zSource);
                        validationFormat->validateBlocksEqual(actualBlock, expectedBlock);
                    }
                    else
                    {
                        // Block is located outside the bounds of the source texture and should be
                        // compared against known expected values for the destination texture.
                        auto expectedBlock = expectedOriginal.getBlockAt(x, y, z);
                        validationFormat->validateBlocksEqual(actualBlock, expectedBlock);
                    }
                }
            }
        }
    }

    void checkTestResults(
        ITextureResource::Extents srcMipExtent,
        const void* expectedCopiedData,
        const void* expectedOriginalData)
    {
        ComPtr<ISlangBlob> resultBlob;
        GFX_CHECK_CALL_ABORT(device->readBufferResource(
            resultsBuffer,
            0,
            bufferCopyInfo.bufferSize,
            resultBlob.writeRef()));
        auto results = resultBlob->getBufferPointer();

        ValidationTextureData actual;
        actual.extents = bufferCopyInfo.extent;
        actual.textureData = results;
        actual.strides.x = getTexelSize(dstTextureInfo->format);
        actual.strides.y = alignedRowStride;
        actual.strides.z = actual.extents.height * actual.strides.y;

        ValidationTextureData expectedCopied;
        expectedCopied.extents = srcMipExtent;
        expectedCopied.textureData = expectedCopiedData;
        expectedCopied.strides.x = getTexelSize(srcTextureInfo->format);
        expectedCopied.strides.y = expectedCopied.extents.width * expectedCopied.strides.x;
        expectedCopied.strides.z = expectedCopied.extents.height * expectedCopied.strides.y;

        ValidationTextureData expectedOriginal;
        if (expectedOriginalData)
        {
            expectedOriginal.extents = bufferCopyInfo.extent;
            expectedOriginal.textureData = expectedOriginalData;
            expectedOriginal.strides.x = getTexelSize(dstTextureInfo->format);
            expectedOriginal.strides.y =
                expectedOriginal.extents.width * expectedOriginal.strides.x;
            expectedOriginal.strides.z =
                expectedOriginal.extents.height * expectedOriginal.strides.y;
        }

        validateTestResults(actual, expectedCopied, expectedOriginal);
    }
};

struct SimpleCopyTexture : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 4;
        srcTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 4;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 1;
        srcTextureInfo->arrayLayerCount = 1;

        dstTextureInfo = srcTextureInfo;

        generateTextureData(srcTextureInfo, validationFormat);

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 0;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = 0;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent = srcTextureInfo->extents;
        texCopyInfo.srcOffset = {0, 0, 0};
        texCopyInfo.dstOffset = {0, 0, 0};

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent = dstTextureInfo->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        auto subresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        auto expectedData = srcTextureInfo->subresourceDatas[subresourceIndex];
        checkTestResults(srcTextureInfo->extents, expectedData.data, nullptr);
    }
};

struct CopyTextureSection : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 4;
        srcTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 4;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 2;
        srcTextureInfo->arrayLayerCount =
            (textureType == ITextureResource::Type::Texture3D) ? 1 : 2;

        dstTextureInfo = srcTextureInfo;

        generateTextureData(srcTextureInfo, validationFormat);

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 0;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = (textureType == ITextureResource::Type::Texture3D) ? 0 : 1;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent = srcTextureInfo->extents;
        texCopyInfo.srcOffset = {0, 0, 0};
        texCopyInfo.dstOffset = {0, 0, 0};

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent = dstTextureInfo->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        auto subresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedData =
            srcTextureInfo->subresourceDatas[subresourceIndex];
        checkTestResults(srcTextureInfo->extents, expectedData.data, nullptr);
    }
};

struct LargeSrcToSmallDst : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 8;
        srcTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 8;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 1;
        srcTextureInfo->arrayLayerCount = 1;

        generateTextureData(srcTextureInfo, validationFormat);

        dstTextureInfo->extents.width = 4;
        dstTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 4;
        dstTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        dstTextureInfo->mipLevelCount = 1;
        dstTextureInfo->arrayLayerCount = 1;

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 0;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = 0;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent = dstTextureInfo->extents;
        texCopyInfo.srcOffset = {0, 0, 0};
        texCopyInfo.dstOffset = {0, 0, 0};

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent = dstTextureInfo->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        auto subresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedData =
            srcTextureInfo->subresourceDatas[subresourceIndex];
        checkTestResults(srcTextureInfo->extents, expectedData.data, nullptr);
    }
};

struct SmallSrcToLargeDst : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 4;
        srcTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 4;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 1;
        srcTextureInfo->arrayLayerCount = 1;

        generateTextureData(srcTextureInfo, validationFormat);

        dstTextureInfo->extents.width = 8;
        dstTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 8;
        dstTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        dstTextureInfo->mipLevelCount = 1;
        dstTextureInfo->arrayLayerCount = 1;

        generateTextureData(dstTextureInfo, validationFormat);

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 0;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = 0;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent = srcTextureInfo->extents;
        texCopyInfo.srcOffset = {0, 0, 0};
        texCopyInfo.dstOffset = {0, 0, 0};

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent = dstTextureInfo->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        auto copiedSubresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedCopiedData =
            srcTextureInfo->subresourceDatas[copiedSubresourceIndex];
        auto originalSubresourceIndex = getSubresourceIndex(
            dstSubresource.mipLevel,
            dstTextureInfo->mipLevelCount,
            dstSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedOriginalData =
            dstTextureInfo->subresourceDatas[originalSubresourceIndex];
        checkTestResults(
            srcTextureInfo->extents,
            expectedCopiedData.data,
            expectedOriginalData.data);
    }
};

struct CopyBetweenMips : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 16;
        srcTextureInfo->extents.height =
            (textureType == ITextureResource::Type::Texture1D) ? 1 : 16;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 4;
        srcTextureInfo->arrayLayerCount = 1;

        generateTextureData(srcTextureInfo, validationFormat);

        dstTextureInfo->extents.width = 16;
        dstTextureInfo->extents.height =
            (textureType == ITextureResource::Type::Texture1D) ? 1 : 16;
        dstTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        dstTextureInfo->mipLevelCount = 4;
        dstTextureInfo->arrayLayerCount = 1;

        generateTextureData(dstTextureInfo, validationFormat);

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 2;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = 0;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 1;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        auto copiedSubresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        auto originalSubresourceIndex = getSubresourceIndex(
            dstSubresource.mipLevel,
            dstTextureInfo->mipLevelCount,
            dstSubresource.baseArrayLayer);

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent = srcTextureInfo->subresourceObjects[copiedSubresourceIndex]->extents;
        texCopyInfo.srcOffset = {0, 0, 0};
        texCopyInfo.dstOffset = {0, 0, 0};

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent =
            dstTextureInfo->subresourceObjects[originalSubresourceIndex]->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        ITextureResource::SubresourceData expectedCopiedData =
            srcTextureInfo->subresourceDatas[copiedSubresourceIndex];
        ITextureResource::SubresourceData expectedOriginalData =
            dstTextureInfo->subresourceDatas[originalSubresourceIndex];
        auto srcMipExtent = srcTextureInfo->subresourceObjects[2]->extents;
        checkTestResults(srcMipExtent, expectedCopiedData.data, expectedOriginalData.data);
    }
};

struct CopyBetweenLayers : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 4;
        srcTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 4;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 1;
        srcTextureInfo->arrayLayerCount =
            (textureType == ITextureResource::Type::Texture3D) ? 1 : 2;

        generateTextureData(srcTextureInfo, validationFormat);
        dstTextureInfo = srcTextureInfo;

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 0;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = 0;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = (textureType == ITextureResource::Type::Texture3D) ? 0 : 1;
        dstSubresource.layerCount = 1;

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent = srcTextureInfo->extents;
        texCopyInfo.srcOffset = {0, 0, 0};
        texCopyInfo.dstOffset = {0, 0, 0};

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent = dstTextureInfo->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        auto copiedSubresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedCopiedData =
            srcTextureInfo->subresourceDatas[copiedSubresourceIndex];
        auto originalSubresourceIndex = getSubresourceIndex(
            dstSubresource.mipLevel,
            dstTextureInfo->mipLevelCount,
            dstSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedOriginalData =
            dstTextureInfo->subresourceDatas[originalSubresourceIndex];
        checkTestResults(
            srcTextureInfo->extents,
            expectedCopiedData.data,
            expectedOriginalData.data);
    }
};

struct CopyWithOffsets : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 8;
        srcTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 8;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 1;
        srcTextureInfo->arrayLayerCount = 1;

        generateTextureData(srcTextureInfo, validationFormat);

        dstTextureInfo->extents.width = 16;
        dstTextureInfo->extents.height =
            (textureType == ITextureResource::Type::Texture1D) ? 1 : 16;
        dstTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 4 : 1;
        dstTextureInfo->mipLevelCount = 1;
        dstTextureInfo->arrayLayerCount = 1;

        generateTextureData(dstTextureInfo, validationFormat);

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 0;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = 0;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent.width = 4;
        texCopyInfo.extent.height = 4;
        texCopyInfo.extent.depth = 1;
        texCopyInfo.srcOffset = {2, 2, 0};
        texCopyInfo.dstOffset = {4, 4, 0};

        if (textureType == ITextureResource::Type::Texture1D)
        {
            texCopyInfo.extent.height = 1;
            texCopyInfo.srcOffset.y = 0;
            texCopyInfo.dstOffset.y = 0;
        }
        else if (textureType == ITextureResource::Type::Texture3D)
        {
            texCopyInfo.extent.depth = srcTextureInfo->extents.depth;
            texCopyInfo.dstOffset.z = 1;
        }

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent = dstTextureInfo->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        auto copiedSubresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedCopiedData =
            srcTextureInfo->subresourceDatas[copiedSubresourceIndex];
        auto originalSubresourceIndex = getSubresourceIndex(
            dstSubresource.mipLevel,
            dstTextureInfo->mipLevelCount,
            dstSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedOriginalData =
            dstTextureInfo->subresourceDatas[originalSubresourceIndex];
        checkTestResults(
            srcTextureInfo->extents,
            expectedCopiedData.data,
            expectedOriginalData.data);
    }
};

struct CopySectionWithSetExtent : BaseCopyTextureTest
{
    void run()
    {
        auto textureType = srcTextureInfo->textureType;
        auto format = srcTextureInfo->format;

        srcTextureInfo->extents.width = 8;
        srcTextureInfo->extents.height = (textureType == ITextureResource::Type::Texture1D) ? 1 : 8;
        srcTextureInfo->extents.depth = (textureType == ITextureResource::Type::Texture3D) ? 2 : 1;
        srcTextureInfo->mipLevelCount = 1;
        srcTextureInfo->arrayLayerCount = 1;

        generateTextureData(srcTextureInfo, validationFormat);
        dstTextureInfo = srcTextureInfo;

        SubresourceRange srcSubresource = {};
        srcSubresource.aspectMask = getTextureAspect(format);
        srcSubresource.mipLevel = 0;
        srcSubresource.mipLevelCount = 1;
        srcSubresource.baseArrayLayer = 0;
        srcSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = getTextureAspect(format);
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        texCopyInfo.srcSubresource = srcSubresource;
        texCopyInfo.dstSubresource = dstSubresource;
        texCopyInfo.extent.width = 4;
        texCopyInfo.extent.height = 4;
        texCopyInfo.extent.depth = 1;
        texCopyInfo.srcOffset = {0, 0, 0};
        texCopyInfo.dstOffset = {4, 4, 0};

        if (textureType == ITextureResource::Type::Texture1D)
        {
            texCopyInfo.extent.height = 1;
            texCopyInfo.dstOffset.y = 0;
        }
        else if (textureType == ITextureResource::Type::Texture3D)
        {
            texCopyInfo.extent.depth = srcTextureInfo->extents.depth;
        }

        bufferCopyInfo.srcSubresource = dstSubresource;
        bufferCopyInfo.extent = dstTextureInfo->extents;
        bufferCopyInfo.textureOffset = {0, 0, 0};
        bufferCopyInfo.bufferOffset = 0;

        createRequiredResources();
        submitGPUWork();

        auto copiedSubresourceIndex = getSubresourceIndex(
            srcSubresource.mipLevel,
            srcTextureInfo->mipLevelCount,
            srcSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedCopiedData =
            srcTextureInfo->subresourceDatas[copiedSubresourceIndex];
        auto originalSubresourceIndex = getSubresourceIndex(
            dstSubresource.mipLevel,
            dstTextureInfo->mipLevelCount,
            dstSubresource.baseArrayLayer);
        ITextureResource::SubresourceData expectedOriginalData =
            dstTextureInfo->subresourceDatas[originalSubresourceIndex];
        checkTestResults(
            srcTextureInfo->extents,
            expectedCopiedData.data,
            expectedOriginalData.data);
    }
};

template<typename T>
void copyTextureTestImpl(IDevice* device, UnitTestContext* context)
{
    const bool isVkd3d =
        SLANG_ENABLE_VKD3D && strcmp(device->getDeviceInfo().apiName, "Direct3D 12") == 0;

    // Skip Type::Unknown and Type::Buffer as well as Format::Unknown
    // TODO: Add support for TextureCube
    Format formats[] = {
        Format::R8G8B8A8_UNORM,
        Format::R16_FLOAT,
        Format::R16G16_FLOAT,
        Format::R10G10B10A2_UNORM,
        Format::B5G5R5A1_UNORM};
    for (uint32_t i = 2; i < (uint32_t)ITextureResource::Type::_Count - 1; ++i)
    {
        for (auto format : formats)
        {
            // Fails validation VUID-VkImageCreateInfo-imageCreateMaxMipLevels-02251
            if (isVkd3d &&
                (format == Format::R32G32B32_TYPELESS || format == Format::R32G32B32_FLOAT ||
                 format == Format::R32G32B32_UINT || format == Format::R32G32B32_SINT))
            {
                continue;
            }
            auto type = (ITextureResource::Type)i;
            auto validationFormat = getValidationTextureFormat(format);
            if (!validationFormat)
                continue;

            T test;
            test.init(device, context, format, validationFormat, type);
            test.run();
        }
    }
}

SLANG_UNIT_TEST(copyTextureSimple)
{
    runTestImpl(
        copyTextureTestImpl<SimpleCopyTexture>,
        unitTestContext,
        Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<SimpleCopyTexture>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(copyTextureSection)
{
    runTestImpl(
        copyTextureTestImpl<CopyTextureSection>,
        unitTestContext,
        Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<CopyTextureSection>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(copyLargeToSmallTexture)
{
    runTestImpl(
        copyTextureTestImpl<LargeSrcToSmallDst>,
        unitTestContext,
        Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<LargeSrcToSmallDst>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(copySmallToLargeTexture)
{
    runTestImpl(
        copyTextureTestImpl<SmallSrcToLargeDst>,
        unitTestContext,
        Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<SmallSrcToLargeDst>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(copyBetweenMips)
{
    runTestImpl(copyTextureTestImpl<CopyBetweenMips>, unitTestContext, Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<CopyBetweenMips>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(copyBetweenLayers)
{
    runTestImpl(
        copyTextureTestImpl<CopyBetweenLayers>,
        unitTestContext,
        Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<CopyBetweenLayers>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(copyWithOffsets)
{
    runTestImpl(copyTextureTestImpl<CopyWithOffsets>, unitTestContext, Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<CopyWithOffsets>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(copySectionWithSetExtent)
{
    runTestImpl(
        copyTextureTestImpl<CopySectionWithSetExtent>,
        unitTestContext,
        Slang::RenderApiFlag::D3D12);
    runTestImpl(
        copyTextureTestImpl<CopySectionWithSetExtent>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}
} // namespace gfx_test
