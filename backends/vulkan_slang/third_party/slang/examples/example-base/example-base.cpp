#include "example-base.h"

#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace Slang;
using namespace gfx;

Slang::Result WindowedAppBase::initializeBase(
    const char* title,
    int width,
    int height,
    DeviceType deviceType)
{
    // Initialize the rendering layer.
#ifdef _DEBUG
    // Enable debug layer in debug config.
    gfxEnableDebugLayer();
#endif
    IDevice::Desc deviceDesc = {};
    deviceDesc.deviceType = deviceType;
    gfx::Result res = gfxCreateDevice(&deviceDesc, gDevice.writeRef());
    if (SLANG_FAILED(res))
        return res;

    ICommandQueue::Desc queueDesc = {};
    queueDesc.type = ICommandQueue::QueueType::Graphics;
    gQueue = gDevice->createCommandQueue(queueDesc);

    windowWidth = width;
    windowHeight = height;

    IFramebufferLayout::TargetLayout renderTargetLayout = {gfx::Format::R8G8B8A8_UNORM, 1};
    IFramebufferLayout::TargetLayout depthLayout = {gfx::Format::D32_FLOAT, 1};
    IFramebufferLayout::Desc framebufferLayoutDesc;
    framebufferLayoutDesc.renderTargetCount = 1;
    framebufferLayoutDesc.renderTargets = &renderTargetLayout;
    framebufferLayoutDesc.depthStencil = &depthLayout;
    SLANG_RETURN_ON_FAIL(
        gDevice->createFramebufferLayout(framebufferLayoutDesc, gFramebufferLayout.writeRef()));

    // Do not create swapchain and windows in test mode, because there won't be any display.
    if (!isTestMode())
    {
        // Create a window for our application to render into.
        //
        platform::WindowDesc windowDesc;
        windowDesc.title = title;
        windowDesc.width = width;
        windowDesc.height = height;
        windowDesc.style = platform::WindowStyle::Default;
        gWindow = platform::Application::createWindow(windowDesc);
        gWindow->events.mainLoop = [this]() { mainLoop(); };
        gWindow->events.sizeChanged = Slang::Action<>(this, &WindowedAppBase::windowSizeChanged);

        auto deviceInfo = gDevice->getDeviceInfo();
        Slang::StringBuilder titleSb;
        titleSb << title << " (" << deviceInfo.apiName << ": " << deviceInfo.adapterName << ")";
        gWindow->setText(titleSb.getBuffer());

        // Create swapchain and framebuffers.
        gfx::ISwapchain::Desc swapchainDesc = {};
        swapchainDesc.format = gfx::Format::R8G8B8A8_UNORM;
        swapchainDesc.width = width;
        swapchainDesc.height = height;
        swapchainDesc.imageCount = kSwapchainImageCount;
        swapchainDesc.queue = gQueue;
        gfx::WindowHandle windowHandle = gWindow->getNativeHandle().convert<gfx::WindowHandle>();
        gSwapchain = gDevice->createSwapchain(swapchainDesc, windowHandle);
        createSwapchainFramebuffers();
    }
    else
    {
        createOfflineFramebuffers();
    }

    for (uint32_t i = 0; i < kSwapchainImageCount; i++)
    {
        gfx::ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096 * 1024;
        auto transientHeap = gDevice->createTransientResourceHeap(transientHeapDesc);
        gTransientHeaps.add(transientHeap);
    }

    gfx::IRenderPassLayout::Desc renderPassDesc = {};
    renderPassDesc.framebufferLayout = gFramebufferLayout;
    renderPassDesc.renderTargetCount = 1;
    IRenderPassLayout::TargetAccessDesc renderTargetAccess = {};
    IRenderPassLayout::TargetAccessDesc depthStencilAccess = {};
    renderTargetAccess.loadOp = IRenderPassLayout::TargetLoadOp::Clear;
    renderTargetAccess.storeOp = IRenderPassLayout::TargetStoreOp::Store;
    renderTargetAccess.initialState = ResourceState::Undefined;
    renderTargetAccess.finalState = ResourceState::Present;
    depthStencilAccess.loadOp = IRenderPassLayout::TargetLoadOp::Clear;
    depthStencilAccess.storeOp = IRenderPassLayout::TargetStoreOp::Store;
    depthStencilAccess.initialState = ResourceState::DepthWrite;
    depthStencilAccess.finalState = ResourceState::DepthWrite;
    renderPassDesc.renderTargetAccess = &renderTargetAccess;
    renderPassDesc.depthStencilAccess = &depthStencilAccess;
    gRenderPass = gDevice->createRenderPassLayout(renderPassDesc);

    return SLANG_OK;
}

void WindowedAppBase::mainLoop()
{
    int frameBufferIndex = gSwapchain->acquireNextImage();

    gTransientHeaps[frameBufferIndex]->synchronizeAndReset();
    renderFrame(frameBufferIndex);
    gTransientHeaps[frameBufferIndex]->finish();
}

void WindowedAppBase::offlineRender()
{
    gTransientHeaps[0]->synchronizeAndReset();
    renderFrame(0);
    gTransientHeaps[0]->finish();
}

void WindowedAppBase::createFramebuffers(
    uint32_t width,
    uint32_t height,
    gfx::Format colorFormat,
    uint32_t frameBufferCount)
{
    for (uint32_t i = 0; i < frameBufferCount; i++)
    {
        gfx::ITextureResource::Desc depthBufferDesc;
        depthBufferDesc.type = IResource::Type::Texture2D;
        depthBufferDesc.size.width = width;
        depthBufferDesc.size.height = height;
        depthBufferDesc.size.depth = 1;
        depthBufferDesc.format = gfx::Format::D32_FLOAT;
        depthBufferDesc.defaultState = ResourceState::DepthWrite;
        depthBufferDesc.allowedStates = ResourceStateSet(ResourceState::DepthWrite);
        ClearValue depthClearValue = {};
        depthBufferDesc.optimalClearValue = &depthClearValue;
        ComPtr<gfx::ITextureResource> depthBufferResource =
            gDevice->createTextureResource(depthBufferDesc, nullptr);

        ComPtr<gfx::ITextureResource> colorBuffer;
        if (isTestMode())
        {
            gfx::ITextureResource::Desc colorBufferDesc;
            colorBufferDesc.type = IResource::Type::Texture2D;
            colorBufferDesc.size.width = width;
            colorBufferDesc.size.height = height;
            colorBufferDesc.size.depth = 1;
            colorBufferDesc.format = colorFormat;
            colorBufferDesc.defaultState = ResourceState::RenderTarget;
            colorBufferDesc.allowedStates =
                ResourceStateSet(ResourceState::RenderTarget, ResourceState::CopyDestination);
            colorBuffer = gDevice->createTextureResource(colorBufferDesc, nullptr);
        }
        else
        {
            gSwapchain->getImage(i, colorBuffer.writeRef());
        }

        gfx::IResourceView::Desc colorBufferViewDesc;
        memset(&colorBufferViewDesc, 0, sizeof(colorBufferViewDesc));
        colorBufferViewDesc.format = colorFormat;
        colorBufferViewDesc.renderTarget.shape = gfx::IResource::Type::Texture2D;
        colorBufferViewDesc.type = gfx::IResourceView::Type::RenderTarget;
        ComPtr<gfx::IResourceView> rtv =
            gDevice->createTextureView(colorBuffer.get(), colorBufferViewDesc);

        gfx::IResourceView::Desc depthBufferViewDesc;
        memset(&depthBufferViewDesc, 0, sizeof(depthBufferViewDesc));
        depthBufferViewDesc.format = gfx::Format::D32_FLOAT;
        depthBufferViewDesc.renderTarget.shape = gfx::IResource::Type::Texture2D;
        depthBufferViewDesc.type = gfx::IResourceView::Type::DepthStencil;
        ComPtr<gfx::IResourceView> dsv =
            gDevice->createTextureView(depthBufferResource.get(), depthBufferViewDesc);

        gfx::IFramebuffer::Desc framebufferDesc;
        framebufferDesc.renderTargetCount = 1;
        framebufferDesc.depthStencilView = dsv.get();
        framebufferDesc.renderTargetViews = rtv.readRef();
        framebufferDesc.layout = gFramebufferLayout;
        ComPtr<gfx::IFramebuffer> frameBuffer = gDevice->createFramebuffer(framebufferDesc);

        gFramebuffers.add(frameBuffer);
    }
}

void WindowedAppBase::createOfflineFramebuffers()
{
    gFramebuffers.clear();
    createFramebuffers(windowWidth, windowHeight, gfx::Format::R8G8B8A8_UNORM, 1);
}

void WindowedAppBase::createSwapchainFramebuffers()
{
    gFramebuffers.clear();
    createFramebuffers(
        gSwapchain->getDesc().width,
        gSwapchain->getDesc().height,
        gSwapchain->getDesc().format,
        kSwapchainImageCount);
}

ComPtr<gfx::IResourceView> WindowedAppBase::createTextureFromFile(
    String fileName,
    int& textureWidth,
    int& textureHeight)
{
    int channelsInFile = 0;
    auto textureContent =
        stbi_load(fileName.getBuffer(), &textureWidth, &textureHeight, &channelsInFile, 4);
    gfx::ITextureResource::Desc textureDesc = {};
    textureDesc.allowedStates.add(ResourceState::ShaderResource);
    textureDesc.format = gfx::Format::R8G8B8A8_UNORM;
    textureDesc.numMipLevels = Math::Log2Ceil(Math::Min(textureWidth, textureHeight)) + 1;
    textureDesc.type = gfx::IResource::Type::Texture2D;
    textureDesc.size.width = textureWidth;
    textureDesc.size.height = textureHeight;
    textureDesc.size.depth = 1;
    List<gfx::ITextureResource::SubresourceData> subresData;
    List<List<uint32_t>> mipMapData;
    mipMapData.setCount(textureDesc.numMipLevels);
    subresData.setCount(textureDesc.numMipLevels);
    mipMapData[0].setCount(textureWidth * textureHeight);
    memcpy(mipMapData[0].getBuffer(), textureContent, textureWidth * textureHeight * 4);
    stbi_image_free(textureContent);
    subresData[0].data = mipMapData[0].getBuffer();
    subresData[0].strideY = textureWidth * 4;
    subresData[0].strideZ = textureWidth * textureHeight * 4;

    // Build mipmaps.
    struct RGBA
    {
        uint8_t v[4];
    };
    auto castToRGBA = [](uint32_t v)
    {
        RGBA result;
        memcpy(&result, &v, 4);
        return result;
    };
    auto castToUint = [](RGBA v)
    {
        uint32_t result;
        memcpy(&result, &v, 4);
        return result;
    };

    int lastMipWidth = textureWidth;
    int lastMipHeight = textureHeight;
    for (int m = 1; m < textureDesc.numMipLevels; m++)
    {
        auto lastMipmapData = mipMapData[m - 1].getBuffer();
        int w = lastMipWidth / 2;
        int h = lastMipHeight / 2;
        mipMapData[m].setCount(w * h);
        subresData[m].data = mipMapData[m].getBuffer();
        subresData[m].strideY = w * 4;
        subresData[m].strideZ = h * w * 4;
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                auto pix1 = castToRGBA(lastMipmapData[(y * 2) * lastMipWidth + (x * 2)]);
                auto pix2 = castToRGBA(lastMipmapData[(y * 2) * lastMipWidth + (x * 2 + 1)]);
                auto pix3 = castToRGBA(lastMipmapData[(y * 2 + 1) * lastMipWidth + (x * 2)]);
                auto pix4 = castToRGBA(lastMipmapData[(y * 2 + 1) * lastMipWidth + (x * 2 + 1)]);
                RGBA pix;
                for (int c = 0; c < 4; c++)
                {
                    pix.v[c] =
                        (uint8_t)(((uint32_t)pix1.v[c] + pix2.v[c] + pix3.v[c] + pix4.v[c]) / 4);
                }
                mipMapData[m][y * w + x] = castToUint(pix);
            }
        }
        lastMipWidth = w;
        lastMipHeight = h;
    }

    auto texture = gDevice->createTextureResource(textureDesc, subresData.getBuffer());

    gfx::IResourceView::Desc viewDesc = {};
    viewDesc.type = gfx::IResourceView::Type::ShaderResource;
    return gDevice->createTextureView(texture.get(), viewDesc);
}

void WindowedAppBase::windowSizeChanged()
{
    // Wait for the GPU to finish.
    gQueue->waitOnHost();

    auto clientRect = gWindow->getClientRect();
    if (clientRect.width > 0 && clientRect.height > 0)
    {
        // Free all framebuffers before resizing swapchain.
        gFramebuffers = decltype(gFramebuffers)();

        // Resize swapchain.
        if (gSwapchain->resize(clientRect.width, clientRect.height) == SLANG_OK)
        {
            // Recreate framebuffers for each swapchain back buffer image.
            createSwapchainFramebuffers();
            windowWidth = clientRect.width;
            windowHeight = clientRect.height;
        }
    }
}

int64_t getCurrentTime()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

int64_t getTimerFrequency()
{
    return std::chrono::high_resolution_clock::period::den;
}

class DebugCallback : public IDebugCallback
{
public:
    virtual SLANG_NO_THROW void SLANG_MCALL
    handleMessage(DebugMessageType type, DebugMessageSource source, const char* message) override
    {
        const char* typeStr = "";
        switch (type)
        {
        case DebugMessageType::Info:
            typeStr = "INFO: ";
            break;
        case DebugMessageType::Warning:
            typeStr = "WARNING: ";
            break;
        case DebugMessageType::Error:
            typeStr = "ERROR: ";
            break;
        default:
            break;
        }
        const char* sourceStr = "[GraphicsLayer]: ";
        switch (source)
        {
        case DebugMessageSource::Slang:
            sourceStr = "[Slang]: ";
            break;
        case DebugMessageSource::Driver:
            sourceStr = "[Driver]: ";
            break;
        }
        printf("%s%s%s\n", sourceStr, typeStr, message);
#ifdef _WIN32
        OutputDebugStringA(sourceStr);
        OutputDebugStringA(typeStr);
        OutputDebugStringW(String(message).toWString());
        OutputDebugStringW(L"\n");
#endif
    }
};

void initDebugCallback()
{
    static DebugCallback callback = {};
    gfxSetDebugCallback(&callback);
}

#ifdef _WIN32
void _Win32OutputDebugString(const char* str)
{
    OutputDebugStringW(Slang::String(str).toWString().begin());
}
#endif
