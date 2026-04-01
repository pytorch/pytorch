// metal-swap-chain.cpp
#include "metal-swap-chain.h"

#include "../apple/cocoa-util.h"
#include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

ISwapchain* SwapchainImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ISwapchain)
        return static_cast<ISwapchain*>(this);
    return nullptr;
}

void SwapchainImpl::getWindowSize(int& widthOut, int& heightOut) const
{
    CocoaUtil::getNSWindowContentSize((void*)m_windowHandle.handleValues[0], &widthOut, &heightOut);
}

void SwapchainImpl::createImages()
{
    m_images.setCount(m_desc.imageCount);
    for (GfxCount i = 0; i < m_desc.imageCount; ++i)
    {
        ITextureResource::Desc imageDesc = {};
        imageDesc.allowedStates = ResourceStateSet(
            ResourceState::Present,
            ResourceState::RenderTarget,
            ResourceState::CopyDestination,
            ResourceState::CopySource);
        imageDesc.type = IResource::Type::Texture2D;
        imageDesc.arraySize = 0;
        imageDesc.format = m_desc.format;
        imageDesc.size.width = m_desc.width;
        imageDesc.size.height = m_desc.height;
        imageDesc.size.depth = 1;
        imageDesc.numMipLevels = 1;
        imageDesc.defaultState = ResourceState::Present;
        m_device->createTextureResource(
            imageDesc,
            nullptr,
            (gfx::ITextureResource**)m_images[i].writeRef());
    }
}

SwapchainImpl::~SwapchainImpl()
{
    m_images.clear();
    CocoaUtil::destroyMetalLayer(m_metalLayer);
}

Result SwapchainImpl::init(DeviceImpl* device, const ISwapchain::Desc& desc, WindowHandle window)
{
    m_device = device;
    m_desc = desc;
    m_windowHandle = window;
    m_metalFormat = MetalUtil::translatePixelFormat(desc.format);
    m_currentImageIndex = -1;

    getWindowSize(m_desc.width, m_desc.height);

    m_metalLayer = (CA::MetalLayer*)CocoaUtil::createMetalLayer((void*)window.handleValues[0]);
    if (!m_metalLayer)
    {
        return SLANG_FAIL;
    }
    m_metalLayer->setPixelFormat(m_metalFormat);
    m_metalLayer->setDevice(m_device->m_device.get());
    m_metalLayer->setDrawableSize(CGSize{(float)m_desc.width, (float)m_desc.height});
    // We need to be able to copy from a texture.
    m_metalLayer->setFramebufferOnly(false);

    createImages();

    return SLANG_OK;
}

Result SwapchainImpl::getImage(GfxIndex index, ITextureResource** outResource)
{
    if (index < 0 || index >= m_desc.imageCount)
        return SLANG_FAIL;
    returnComPtr(outResource, m_images[index]);
    return SLANG_OK;
}

Result SwapchainImpl::resize(GfxCount width, GfxCount height)
{
    m_currentImageIndex = -1;
    m_currentDrawable.reset();
    getWindowSize(m_desc.width, m_desc.height);
    m_metalLayer->setDrawableSize(CGSize{(float)m_desc.width, (float)m_desc.height});
    createImages();
    return SLANG_OK;
}

Result SwapchainImpl::present()
{
    AUTORELEASEPOOL

    if (!m_currentDrawable)
    {
        return SLANG_FAIL;
    }

    MTL::CommandBuffer* commandBuffer = m_device->m_commandQueue->commandBuffer();
    MTL::BlitCommandEncoder* encoder = commandBuffer->blitCommandEncoder();
    encoder->copyFromTexture(
        m_images[m_currentImageIndex]->m_texture.get(),
        m_currentDrawable->texture());
    encoder->endEncoding();
    commandBuffer->presentDrawable(m_currentDrawable.get());
    commandBuffer->commit();
    m_currentDrawable.reset();
    return SLANG_OK;

    // // TODO: Expose controls via some other means
    // static uint32_t frameCount = 0;
    // static uint32_t maxFrameCount = 32;
    // ++frameCount;
    // if (m_device->captureEnabled() && frameCount == maxFrameCount)
    // {
    //     MTL::CaptureManager* captureManager = MTL::CaptureManager::sharedCaptureManager();
    //     captureManager->stopCapture();
    //     exit(1);
    // }
    // return SLANG_OK;
}

int SwapchainImpl::acquireNextImage()
{
    AUTORELEASEPOOL

    m_currentDrawable = NS::RetainPtr(m_metalLayer->nextDrawable());
    if (m_currentDrawable)
    {
        m_currentImageIndex = (m_currentImageIndex + 1) % m_desc.imageCount;
    }
    else
    {
        m_currentImageIndex = -1;
    }
    return m_currentImageIndex;
}

Result SwapchainImpl::setFullScreenMode(bool mode)
{
    return SLANG_E_NOT_AVAILABLE;
}

} // namespace metal
} // namespace gfx
