// vk-swap-chain.cpp
#include "vk-swap-chain.h"

#include "../apple/cocoa-util.h"
#include "vk-util.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

ISwapchain* SwapchainImpl::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ISwapchain)
        return static_cast<ISwapchain*>(this);
    return nullptr;
}

void SwapchainImpl::destroySwapchainAndImages()
{
    m_api->vkQueueWaitIdle(m_queue->m_queue);
    if (m_swapChain != VK_NULL_HANDLE)
    {
        m_api->vkDestroySwapchainKHR(m_api->m_device, m_swapChain, nullptr);
        m_swapChain = VK_NULL_HANDLE;
    }

    // Mark that it is no longer used
    m_images.clear();
}

void SwapchainImpl::getWindowSize(int* widthOut, int* heightOut) const
{
#if SLANG_WINDOWS_FAMILY
    RECT rc;
    ::GetClientRect((HWND)m_windowHandle.handleValues[0], &rc);
    *widthOut = rc.right - rc.left;
    *heightOut = rc.bottom - rc.top;
#elif SLANG_APPLE_FAMILY
    CocoaUtil::getNSWindowContentSize((void*)m_windowHandle.handleValues[0], widthOut, heightOut);
#elif defined(SLANG_ENABLE_XLIB)
    XWindowAttributes winAttr = {};
    XGetWindowAttributes(
        (Display*)m_windowHandle.handleValues[0],
        (Window)m_windowHandle.handleValues[1],
        &winAttr);

    *widthOut = winAttr.width;
    *heightOut = winAttr.height;
#else
    *widthOut = 0;
    *heightOut = 0;
#endif
}

Result SwapchainImpl::createSwapchainAndImages()
{
    int width, height;
    getWindowSize(&width, &height);

    VkExtent2D imageExtent = {};
    imageExtent.width = width;
    imageExtent.height = height;

    m_desc.width = width;
    m_desc.height = height;

    // catch this before throwing error
    if (width == 0 || height == 0)
    {
        return SLANG_FAIL;
    }

    // It is necessary to query the caps -> otherwise the LunarG verification layer will
    // issue an error
    {
        VkSurfaceCapabilitiesKHR surfaceCaps;

        SLANG_VK_RETURN_ON_FAIL(m_api->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            m_api->m_physicalDevice,
            m_surface,
            &surfaceCaps));
    }

    VkPresentModeKHR presentMode;
    List<VkPresentModeKHR> presentModes;
    uint32_t numPresentModes = 0;
    m_api->vkGetPhysicalDeviceSurfacePresentModesKHR(
        m_api->m_physicalDevice,
        m_surface,
        &numPresentModes,
        nullptr);
    presentModes.setCount(numPresentModes);
    m_api->vkGetPhysicalDeviceSurfacePresentModesKHR(
        m_api->m_physicalDevice,
        m_surface,
        &numPresentModes,
        presentModes.getBuffer());

    {
        int numCheckPresentOptions = 3;
        VkPresentModeKHR presentOptions[] = {
            VK_PRESENT_MODE_IMMEDIATE_KHR,
            VK_PRESENT_MODE_MAILBOX_KHR,
            VK_PRESENT_MODE_FIFO_KHR};
        if (m_desc.enableVSync)
        {
            presentOptions[0] = VK_PRESENT_MODE_FIFO_KHR;
            presentOptions[1] = VK_PRESENT_MODE_IMMEDIATE_KHR;
            presentOptions[2] = VK_PRESENT_MODE_MAILBOX_KHR;
        }

        presentMode = VK_PRESENT_MODE_MAX_ENUM_KHR; // Invalid

        // Find the first option that's available on the device
        for (int j = 0; j < numCheckPresentOptions; j++)
        {
            if (presentModes.indexOf(presentOptions[j]) != Index(-1))
            {
                presentMode = presentOptions[j];
                break;
            }
        }

        if (presentMode == VK_PRESENT_MODE_MAX_ENUM_KHR)
        {
            return SLANG_FAIL;
        }
    }

    VkSwapchainKHR oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainCreateInfoKHR swapchainDesc = {};
    swapchainDesc.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainDesc.surface = m_surface;
    swapchainDesc.minImageCount = m_desc.imageCount;
    swapchainDesc.imageFormat = m_vkformat;
    swapchainDesc.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainDesc.imageExtent = imageExtent;
    swapchainDesc.imageArrayLayers = 1;
    swapchainDesc.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchainDesc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainDesc.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchainDesc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainDesc.presentMode = presentMode;
    swapchainDesc.clipped = VK_TRUE;
    swapchainDesc.oldSwapchain = oldSwapchain;

    SLANG_VK_RETURN_ON_FAIL(
        m_api->vkCreateSwapchainKHR(m_api->m_device, &swapchainDesc, nullptr, &m_swapChain));

    uint32_t numSwapChainImages = 0;
    m_api->vkGetSwapchainImagesKHR(m_api->m_device, m_swapChain, &numSwapChainImages, nullptr);
    List<VkImage> vkImages;
    {
        vkImages.setCount(numSwapChainImages);
        m_api->vkGetSwapchainImagesKHR(
            m_api->m_device,
            m_swapChain,
            &numSwapChainImages,
            vkImages.getBuffer());
    }

    for (GfxIndex i = 0; i < m_desc.imageCount; i++)
    {
        ITextureResource::Desc imageDesc = {};
        imageDesc.allowedStates = ResourceStateSet(
            ResourceState::Present,
            ResourceState::RenderTarget,
            ResourceState::CopyDestination);
        imageDesc.type = IResource::Type::Texture2D;
        imageDesc.arraySize = 0;
        imageDesc.format = m_desc.format;
        imageDesc.size.width = m_desc.width;
        imageDesc.size.height = m_desc.height;
        imageDesc.size.depth = 1;
        imageDesc.numMipLevels = 1;
        imageDesc.defaultState = ResourceState::Present;
        RefPtr<TextureResourceImpl> image = new TextureResourceImpl(imageDesc, m_renderer);
        image->m_image = vkImages[i];
        image->m_imageMemory = 0;
        image->m_vkformat = m_vkformat;
        image->m_isWeakImageReference = true;
        m_images.add(image);
    }
    return SLANG_OK;
}

SwapchainImpl::~SwapchainImpl()
{
    destroySwapchainAndImages();
    if (m_surface)
    {
        m_api->vkDestroySurfaceKHR(m_api->m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
    m_renderer->m_api.vkDestroySemaphore(m_renderer->m_api.m_device, m_nextImageSemaphore, nullptr);
#if SLANG_APPLE_FAMILY
    CocoaUtil::destroyMetalLayer(m_metalLayer);
#endif
}

Index SwapchainImpl::_indexOfFormat(List<VkSurfaceFormatKHR>& formatsIn, VkFormat format)
{
    const Index numFormats = formatsIn.getCount();
    const VkSurfaceFormatKHR* formats = formatsIn.getBuffer();

    for (Index i = 0; i < numFormats; ++i)
    {
        if (formats[i].format == format)
        {
            return i;
        }
    }
    return -1;
}

Result SwapchainImpl::init(DeviceImpl* renderer, const ISwapchain::Desc& desc, WindowHandle window)
{
    m_desc = desc;
    m_renderer = renderer;
    m_api = &renderer->m_api;
    m_queue = static_cast<CommandQueueImpl*>(desc.queue);
    m_windowHandle = window;

    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    SLANG_VK_RETURN_ON_FAIL(renderer->m_api.vkCreateSemaphore(
        renderer->m_api.m_device,
        &semaphoreCreateInfo,
        nullptr,
        &m_nextImageSemaphore));

    m_queue = static_cast<CommandQueueImpl*>(desc.queue);

    // Make sure it's not set initially
    m_vkformat = VK_FORMAT_UNDEFINED;

#if SLANG_WINDOWS_FAMILY
    VkWin32SurfaceCreateInfoKHR surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceCreateInfo.hinstance = ::GetModuleHandle(nullptr);
    surfaceCreateInfo.hwnd = (HWND)window.handleValues[0];
    SLANG_VK_RETURN_ON_FAIL(
        m_api->vkCreateWin32SurfaceKHR(m_api->m_instance, &surfaceCreateInfo, nullptr, &m_surface));
#elif SLANG_APPLE_FAMILY
    m_metalLayer = CocoaUtil::createMetalLayer((void*)window.handleValues[0]);
    VkMetalSurfaceCreateInfoEXT surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_METAL_SURFACE_CREATE_INFO_EXT;
    surfaceCreateInfo.pLayer = (CAMetalLayer*)m_metalLayer;
    SLANG_VK_RETURN_ON_FAIL(
        m_api->vkCreateMetalSurfaceEXT(m_api->m_instance, &surfaceCreateInfo, nullptr, &m_surface));
#elif SLANG_ENABLE_XLIB
    VkXlibSurfaceCreateInfoKHR surfaceCreateInfo = {};
    surfaceCreateInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
    surfaceCreateInfo.dpy = (Display*)window.handleValues[0];
    surfaceCreateInfo.window = (Window)window.handleValues[1];
    SLANG_VK_RETURN_ON_FAIL(
        m_api->vkCreateXlibSurfaceKHR(m_api->m_instance, &surfaceCreateInfo, nullptr, &m_surface));
#else
    return SLANG_E_NOT_AVAILABLE;
#endif

    VkBool32 supported = false;
    m_api->vkGetPhysicalDeviceSurfaceSupportKHR(
        m_api->m_physicalDevice,
        renderer->m_queueFamilyIndex,
        m_surface,
        &supported);

    uint32_t numSurfaceFormats = 0;
    List<VkSurfaceFormatKHR> surfaceFormats;
    m_api->vkGetPhysicalDeviceSurfaceFormatsKHR(
        m_api->m_physicalDevice,
        m_surface,
        &numSurfaceFormats,
        nullptr);
    surfaceFormats.setCount(int(numSurfaceFormats));
    m_api->vkGetPhysicalDeviceSurfaceFormatsKHR(
        m_api->m_physicalDevice,
        m_surface,
        &numSurfaceFormats,
        surfaceFormats.getBuffer());

    // Look for a suitable format
    List<VkFormat> formats;
    formats.add(VulkanUtil::getVkFormat(desc.format));
    // HACK! To check for a different format if couldn't be found
    if (desc.format == Format::R8G8B8A8_UNORM)
    {
        formats.add(VK_FORMAT_B8G8R8A8_UNORM);
    }

    for (Index i = 0; i < formats.getCount(); ++i)
    {
        VkFormat format = formats[i];
        if (_indexOfFormat(surfaceFormats, format) >= 0)
        {
            m_vkformat = format;
        }
    }

    if (m_vkformat == VK_FORMAT_UNDEFINED)
    {
        return SLANG_FAIL;
    }

    // Save the desc
    m_desc = desc;
    if (m_desc.format == Format::R8G8B8A8_UNORM && m_vkformat == VK_FORMAT_B8G8R8A8_UNORM)
    {
        m_desc.format = Format::B8G8R8A8_UNORM;
    }

    createSwapchainAndImages();
    return SLANG_OK;
}

Result SwapchainImpl::getImage(GfxIndex index, ITextureResource** outResource)
{
    if (m_images.getCount() <= (Index)index)
        return SLANG_FAIL;
    returnComPtr(outResource, m_images[index]);
    return SLANG_OK;
}

Result SwapchainImpl::resize(GfxCount width, GfxCount height)
{
    SLANG_UNUSED(width);
    SLANG_UNUSED(height);
    destroySwapchainAndImages();
    return createSwapchainAndImages();
}

Result SwapchainImpl::present()
{
    // If there are pending fence wait operations, flush them as an
    // empty vkQueueSubmit.
    if (m_queue->m_pendingWaitFences.getCount() != 0)
    {
        m_queue->queueSubmitImpl(0, nullptr, nullptr, 0);
    }

    uint32_t swapChainIndices[] = {uint32_t(m_currentImageIndex)};

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapChain;
    presentInfo.pImageIndices = swapChainIndices;
    Array<VkSemaphore, 2> waitSemaphores;
    for (auto s : m_queue->m_pendingWaitSemaphores)
    {
        if (s != VK_NULL_HANDLE)
        {
            waitSemaphores.add(s);
        }
    }
    m_queue->m_pendingWaitSemaphores[0] = VK_NULL_HANDLE;
    m_queue->m_pendingWaitSemaphores[1] = VK_NULL_HANDLE;
    presentInfo.waitSemaphoreCount = (uint32_t)waitSemaphores.getCount();
    if (presentInfo.waitSemaphoreCount)
    {
        presentInfo.pWaitSemaphores = waitSemaphores.getBuffer();
    }
    if (m_currentImageIndex != -1)
        m_api->vkQueuePresentKHR(m_queue->m_queue, &presentInfo);
    return SLANG_OK;
}

int SwapchainImpl::acquireNextImage()
{
    if (!m_images.getCount())
    {
        m_queue->m_pendingWaitSemaphores[1] = VK_NULL_HANDLE;
        return -1;
    }

    m_currentImageIndex = -1;
    VkResult result = m_api->vkAcquireNextImageKHR(
        m_api->m_device,
        m_swapChain,
        UINT64_MAX,
        m_nextImageSemaphore,
        VK_NULL_HANDLE,
        (uint32_t*)&m_currentImageIndex);

    if (result != VK_SUCCESS
#if SLANG_APPLE_FAMILY
        && result != VK_SUBOPTIMAL_KHR
#endif
    )
    {
        m_currentImageIndex = -1;
        destroySwapchainAndImages();
        return m_currentImageIndex;
    }
    // Make the queue's next submit wait on `m_nextImageSemaphore`.
    m_queue->m_pendingWaitSemaphores[1] = m_nextImageSemaphore;
    return m_currentImageIndex;
}

Result SwapchainImpl::setFullScreenMode(bool mode)
{
    return SLANG_FAIL;
}

} // namespace vk
} // namespace gfx
