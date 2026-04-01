// vk-swap-chain.h
#pragma once

#include "vk-base.h"
#include "vk-command-queue.h"
#include "vk-device.h"
#include "vk-texture.h"

namespace gfx
{

using namespace Slang;

namespace vk
{

class SwapchainImpl : public ISwapchain, public ComObject
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    ISwapchain* getInterface(const Guid& guid);

public:
    VkSwapchainKHR m_swapChain;
    VkSurfaceKHR m_surface;
    VkSemaphore m_nextImageSemaphore; // Semaphore to signal after `acquireNextImage`.
    ISwapchain::Desc m_desc;
    VkFormat m_vkformat;
    RefPtr<CommandQueueImpl> m_queue;
    ShortList<RefPtr<TextureResourceImpl>> m_images;
    RefPtr<DeviceImpl> m_renderer;
    VulkanApi* m_api;
    uint32_t m_currentImageIndex = 0;
    WindowHandle m_windowHandle;
#if SLANG_APPLE_FAMILY
    void* m_metalLayer;
#endif

    void destroySwapchainAndImages();

    void getWindowSize(int* widthOut, int* heightOut) const;

    Result createSwapchainAndImages();

public:
    ~SwapchainImpl();

    static Index _indexOfFormat(List<VkSurfaceFormatKHR>& formatsIn, VkFormat format);

    Result init(DeviceImpl* renderer, const ISwapchain::Desc& desc, WindowHandle window);

    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() override { return m_desc; }
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getImage(GfxIndex index, ITextureResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL resize(GfxCount width, GfxCount height) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL present() override;
    virtual SLANG_NO_THROW int SLANG_MCALL acquireNextImage() override;
    virtual SLANG_NO_THROW bool SLANG_MCALL isOccluded() override { return false; }
    virtual SLANG_NO_THROW Result SLANG_MCALL setFullScreenMode(bool mode) override;
};

} // namespace vk
} // namespace gfx
