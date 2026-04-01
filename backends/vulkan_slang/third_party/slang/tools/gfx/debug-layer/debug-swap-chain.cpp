// debug-swap-chain.cpp
#include "debug-swap-chain.h"

#include "debug-command-queue.h"
#include "debug-helper-functions.h"
#include "debug-texture.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

const ISwapchain::Desc& DebugSwapchain::getDesc()
{
    SLANG_GFX_API_FUNC;
    desc = baseObject->getDesc();
    desc.queue = queue.Ptr();
    return desc;
}

Result DebugSwapchain::getImage(GfxIndex index, ITextureResource** outResource)
{
    SLANG_GFX_API_FUNC;
    maybeRebuildImageList();
    if (index > (GfxCount)m_images.getCount())
    {
        GFX_DIAGNOSE_ERROR_FORMAT(
            "`index`(%d) must not exceed total number of images (%d) in the swapchain.",
            index,
            (uint32_t)m_images.getCount());
    }
    returnComPtr(outResource, m_images[index]);
    return SLANG_OK;
}

Result DebugSwapchain::present()
{
    SLANG_GFX_API_FUNC;
    return baseObject->present();
}

int DebugSwapchain::acquireNextImage()
{
    SLANG_GFX_API_FUNC;
    return baseObject->acquireNextImage();
}

Result DebugSwapchain::resize(GfxCount width, GfxCount height)
{
    SLANG_GFX_API_FUNC;
    for (auto& image : m_images)
    {
        if (image->debugGetReferenceCount() != 1)
        {
            // Only warn here because tools like NSight might keep
            // an additional reference to swapchain images.
            GFX_DIAGNOSE_WARNING("all swapchain images must be released before calling resize().");
            break;
        }
    }
    m_images.clearAndDeallocate();
    return baseObject->resize(width, height);
}

bool DebugSwapchain::isOccluded()
{
    SLANG_GFX_API_FUNC;
    return baseObject->isOccluded();
}

Result DebugSwapchain::setFullScreenMode(bool mode)
{
    SLANG_GFX_API_FUNC;
    return baseObject->setFullScreenMode(mode);
}

void DebugSwapchain::maybeRebuildImageList()
{
    SLANG_GFX_API_FUNC;
    if (m_images.getCount() != 0)
        return;
    m_images.clearAndDeallocate();
    for (GfxIndex i = 0; i < baseObject->getDesc().imageCount; i++)
    {
        RefPtr<DebugTextureResource> image = new DebugTextureResource();
        baseObject->getImage(i, image->baseObject.writeRef());
        m_images.add(image);
    }
}

} // namespace debug
} // namespace gfx
