#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;
using namespace gfx;

namespace gfx_test
{
void clearTextureTestImpl(IDevice* device, UnitTestContext* context)
{
    Slang::ComPtr<ITransientResourceHeap> transientHeap;
    ITransientResourceHeap::Desc transientHeapDesc = {};
    transientHeapDesc.constantBufferSize = 4096;
    GFX_CHECK_CALL_ABORT(
        device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

    ITextureResource::Desc srcTexDesc = {};
    srcTexDesc.type = IResource::Type::Texture2D;
    srcTexDesc.numMipLevels = 1;
    srcTexDesc.arraySize = 1;
    srcTexDesc.size.width = 4;
    srcTexDesc.size.height = 4;
    srcTexDesc.size.depth = 1;
    srcTexDesc.defaultState = ResourceState::RenderTarget;
    srcTexDesc.allowedStates = ResourceStateSet(
        ResourceState::RenderTarget,
        ResourceState::CopySource,
        ResourceState::CopyDestination);
    srcTexDesc.format = Format::R32G32B32A32_FLOAT;

    Slang::ComPtr<ITextureResource> srcTexture;
    GFX_CHECK_CALL_ABORT(device->createTextureResource(srcTexDesc, nullptr, srcTexture.writeRef()));

    Slang::ComPtr<IResourceView> rtv;
    IResourceView::Desc rtvDesc = {};
    rtvDesc.type = IResourceView::Type::RenderTarget;
    rtvDesc.format = Format::R32G32B32A32_FLOAT;
    rtvDesc.renderTarget.shape = IResource::Type::Texture2D;
    rtvDesc.subresourceRange.layerCount = 1;
    rtvDesc.subresourceRange.mipLevelCount = 1;
    rtv = device->createTextureView(srcTexture, rtvDesc);

    {
        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = device->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        auto resourceEncoder = commandBuffer->encodeResourceCommands();
        ClearValue clearValue = {};
        clearValue.color.floatValues[0] = 0.5f;
        clearValue.color.floatValues[1] = 1.0f;
        clearValue.color.floatValues[2] = 0.2f;
        clearValue.color.floatValues[3] = 0.1f;
        resourceEncoder->clearResourceView(
            rtv,
            &clearValue,
            ClearResourceViewFlags::FloatClearValues);
        resourceEncoder->textureBarrier(
            srcTexture,
            ResourceState::RenderTarget,
            ResourceState::CopySource);
        resourceEncoder->endEncoding();

        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);

        queue->waitOnHost();

        Slang::ComPtr<ISlangBlob> blob;
        size_t rowPitch, pixelSize;
        device->readTextureResource(
            srcTexture,
            ResourceState::CopySource,
            blob.writeRef(),
            &rowPitch,
            &pixelSize);
        float* data = (float*)blob->getBufferPointer();
        for (int i = 0; i < 4; i++)
        {
            SLANG_CHECK(data[i] == clearValue.color.floatValues[i]);
        }
    }
}

SLANG_UNIT_TEST(clearTextureTestVulkan)
{
    runTestImpl(clearTextureTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}
} // namespace gfx_test
