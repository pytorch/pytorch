#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "platform/window.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace gfx;
using namespace Slang;

namespace gfx_test
{
struct Vertex
{
    float position[3];
};

static const int kVertexCount = 3;
static const Vertex kVertexData[kVertexCount] = {
    // Triangle 1
    {0, 0, 1},
    {4, 0, 1},
    {0, 4, 1},
};

struct SwapchainResizeTest
{
    IDevice* device;
    UnitTestContext* context;

    RefPtr<platform::Window> window;
    ComPtr<ICommandQueue> queue;
    ComPtr<ISwapchain> swapchain;

    ComPtr<ITransientResourceHeap> transientHeap;
    ComPtr<gfx::IFramebufferLayout> framebufferLayout;
    ComPtr<IPipelineState> pipelineState;
    ComPtr<IRenderPassLayout> renderPass;
    List<ComPtr<IFramebuffer>> framebuffers;

    ComPtr<IBufferResource> vertexBuffer;

    GfxCount width = 500;
    GfxCount height = 500;
    static const int kSwapchainImageCount = 2;
    const Format desiredFormat = Format::R8G8B8A8_UNORM;

    void init(IDevice* device, UnitTestContext* context)
    {
        this->device = device;
        this->context = context;
    }

    void createSwapchainFramebuffers()
    {
        framebuffers.clear();
        for (GfxIndex i = 0; i < kSwapchainImageCount; ++i)
        {
            ComPtr<ITextureResource> colorBuffer;
            swapchain->getImage(i, colorBuffer.writeRef());

            gfx::IResourceView::Desc colorBufferViewDesc;
            memset(&colorBufferViewDesc, 0, sizeof(colorBufferViewDesc));
            colorBufferViewDesc.format = swapchain->getDesc().format;
            colorBufferViewDesc.renderTarget.shape = gfx::IResource::Type::Texture2D;
            colorBufferViewDesc.type = gfx::IResourceView::Type::RenderTarget;
            auto rtv = device->createTextureView(colorBuffer.get(), colorBufferViewDesc);

            gfx::IFramebuffer::Desc framebufferDesc;
            framebufferDesc.renderTargetCount = 1;
            framebufferDesc.depthStencilView = nullptr;
            framebufferDesc.renderTargetViews = rtv.readRef();
            framebufferDesc.layout = framebufferLayout;
            ComPtr<IFramebuffer> framebuffer;
            GFX_CHECK_CALL_ABORT(
                device->createFramebuffer(framebufferDesc, framebuffer.writeRef()));

            framebuffers.add(framebuffer);
        }
    }

    void createRequiredResources()
    {
        platform::Application::init();

        platform::WindowDesc windowDesc;
        windowDesc.title = "";
        windowDesc.width = width;
        windowDesc.height = height;
        windowDesc.style = platform::WindowStyle::Default;
        window = platform::Application::createWindow(windowDesc);

        ICommandQueue::Desc queueDesc = {};
        queueDesc.type = ICommandQueue::QueueType::Graphics;
        queue = device->createCommandQueue(queueDesc);

        ISwapchain::Desc swapchainDesc = {};
        swapchainDesc.format = desiredFormat;
        swapchainDesc.width = width;
        swapchainDesc.height = height;
        swapchainDesc.imageCount = kSwapchainImageCount;
        swapchainDesc.queue = queue;
        WindowHandle windowHandle = window->getNativeHandle().convert<WindowHandle>();
        auto createSwapchainResult =
            device->createSwapchain(swapchainDesc, windowHandle, swapchain.writeRef());
        if (SLANG_FAILED(createSwapchainResult))
        {
            SLANG_IGNORE_TEST;
        }

        VertexStreamDesc vertexStreams[] = {
            {sizeof(Vertex), InputSlotClass::PerVertex, 0},
        };

        InputElementDesc inputElements[] = {
            // Vertex buffer data
            {"POSITIONA", 0, Format::R32G32B32_FLOAT, offsetof(Vertex, position), 0},
        };
        IInputLayout::Desc inputLayoutDesc = {};
        inputLayoutDesc.inputElementCount = SLANG_COUNT_OF(inputElements);
        inputLayoutDesc.inputElements = inputElements;
        inputLayoutDesc.vertexStreamCount = SLANG_COUNT_OF(vertexStreams);
        inputLayoutDesc.vertexStreams = vertexStreams;
        auto inputLayout = device->createInputLayout(inputLayoutDesc);
        SLANG_CHECK_ABORT(inputLayout != nullptr);

        IBufferResource::Desc vertexBufferDesc;
        vertexBufferDesc.type = IResource::Type::Buffer;
        vertexBufferDesc.sizeInBytes = kVertexCount * sizeof(Vertex);
        vertexBufferDesc.defaultState = ResourceState::VertexBuffer;
        vertexBuffer = device->createBufferResource(vertexBufferDesc, &kVertexData[0]);
        SLANG_CHECK_ABORT(vertexBuffer != nullptr);

        ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096 * 1024;
        GFX_CHECK_CALL_ABORT(
            device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

        ComPtr<IShaderProgram> shaderProgram;
        slang::ProgramLayout* slangReflection;
        GFX_CHECK_CALL_ABORT(loadGraphicsProgram(
            device,
            shaderProgram,
            "swapchain-shader",
            "vertexMain",
            "fragmentMain",
            slangReflection));

        IFramebufferLayout::TargetLayout targetLayout;
        targetLayout.format = swapchain->getDesc().format;
        targetLayout.sampleCount = 1;

        IFramebufferLayout::Desc framebufferLayoutDesc;
        framebufferLayoutDesc.renderTargetCount = 1;
        framebufferLayoutDesc.renderTargets = &targetLayout;
        framebufferLayout = device->createFramebufferLayout(framebufferLayoutDesc);
        SLANG_CHECK_ABORT(framebufferLayout != nullptr);

        GraphicsPipelineStateDesc pipelineDesc = {};
        pipelineDesc.program = shaderProgram.get();
        pipelineDesc.inputLayout = inputLayout;
        pipelineDesc.framebufferLayout = framebufferLayout;
        pipelineDesc.depthStencil.depthTestEnable = false;
        pipelineDesc.depthStencil.depthWriteEnable = false;
        GFX_CHECK_CALL_ABORT(
            device->createGraphicsPipelineState(pipelineDesc, pipelineState.writeRef()));

        IRenderPassLayout::Desc renderPassDesc = {};
        renderPassDesc.framebufferLayout = framebufferLayout;
        renderPassDesc.renderTargetCount = 1;
        IRenderPassLayout::TargetAccessDesc renderTargetAccess = {};
        renderTargetAccess.loadOp = IRenderPassLayout::TargetLoadOp::Clear;
        renderTargetAccess.storeOp = IRenderPassLayout::TargetStoreOp::Store;
        renderTargetAccess.initialState = ResourceState::Undefined;
        renderTargetAccess.finalState = ResourceState::Present;
        renderPassDesc.renderTargetAccess = &renderTargetAccess;
        GFX_CHECK_CALL_ABORT(device->createRenderPassLayout(renderPassDesc, renderPass.writeRef()));

        createSwapchainFramebuffers();
    }

    void renderFrame(GfxIndex framebufferIndex)
    {
        auto commandBuffer = transientHeap->createCommandBuffer();

        auto encoder =
            commandBuffer->encodeRenderCommands(renderPass, framebuffers[framebufferIndex]);
        auto rootObject = encoder->bindPipeline(pipelineState);

        gfx::Viewport viewport = {};
        viewport.maxZ = 1.0f;
        viewport.extentX = (float)width;
        viewport.extentY = (float)height;
        encoder->setViewportAndScissor(viewport);

        encoder->setVertexBuffer(0, vertexBuffer);
        encoder->setPrimitiveTopology(PrimitiveTopology::TriangleList);

        swapchain->acquireNextImage();
        encoder->draw(kVertexCount);
        encoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        swapchain->present();
    }

    void run()
    {
        createRequiredResources();
        // Render for 5 frames then resize the swapchain and render for another 5 frames to ensure
        // the swapchain remains usable after resizing.
        for (GfxIndex i = 0; i < 5; ++i)
        {
            renderFrame(i % kSwapchainImageCount);
        }
        queue->waitOnHost();

        framebuffers = decltype(framebuffers)();
        GFX_CHECK_CALL(swapchain->resize(700, 700));
        createSwapchainFramebuffers();
        width = 700;
        height = 700;

        for (GfxIndex i = 0; i < 5; ++i)
        {
            renderFrame(i % kSwapchainImageCount);
        }
        queue->waitOnHost();
    }
};

void swapchainResizeTestImpl(IDevice* device, UnitTestContext* context)
{
    SwapchainResizeTest t;
    t.init(device, context);
    t.run();
}

SLANG_UNIT_TEST(swapchainResizeD3D12)
{
    runTestImpl(swapchainResizeTestImpl, unitTestContext, RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(swapchainResizeVulkan)
{
    runTestImpl(swapchainResizeTestImpl, unitTestContext, RenderApiFlag::Vulkan);
}

} // namespace gfx_test
