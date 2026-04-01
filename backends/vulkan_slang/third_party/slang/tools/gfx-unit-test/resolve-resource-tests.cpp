#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

#if SLANG_WINDOWS_FAMILY
#include <d3d12.h>
#endif

using namespace Slang;
using namespace gfx;

namespace
{
using namespace gfx_test;

struct Vertex
{
    float position[3];
    float color[3];
};

static const int kVertexCount = 12;
static const Vertex kVertexData[kVertexCount] = {
    // Triangle 1
    {{0, 0, 0.5}, {1, 0, 0}},
    {{1, 1, 0.5}, {1, 0, 0}},
    {{-1, 1, 0.5}, {1, 0, 0}},

    // Triangle 2
    {{-1, 1, 0.5}, {0, 1, 0}},
    {{0, 0, 0.5}, {0, 1, 0}},
    {{-1, -1, 0.5}, {0, 1, 0}},

    // Triangle 3
    {{-1, -1, 0.5}, {0, 0, 1}},
    {{0, 0, 0.5}, {0, 0, 1}},
    {{1, -1, 0.5}, {0, 0, 1}},

    // Triangle 4
    {{1, -1, 0.5}, {0, 0, 0}},
    {{0, 0, 0.5}, {0, 0, 0}},
    {{1, 1, 0.5}, {0, 0, 0}},
};

const int kWidth = 256;
const int kHeight = 256;
Format format = Format::R32G32B32A32_FLOAT;

ComPtr<IBufferResource> createVertexBuffer(IDevice* device)
{
    IBufferResource::Desc vertexBufferDesc;
    vertexBufferDesc.type = IResource::Type::Buffer;
    vertexBufferDesc.sizeInBytes = kVertexCount * sizeof(Vertex);
    vertexBufferDesc.defaultState = ResourceState::VertexBuffer;
    vertexBufferDesc.allowedStates = ResourceState::VertexBuffer;
    ComPtr<IBufferResource> vertexBuffer =
        device->createBufferResource(vertexBufferDesc, &kVertexData[0]);
    SLANG_CHECK_ABORT(vertexBuffer != nullptr);
    return vertexBuffer;
}

struct BaseResolveResourceTest
{
    IDevice* device;
    UnitTestContext* context;

    ComPtr<ITextureResource> msaaTexture;
    ComPtr<ITextureResource> dstTexture;

    ComPtr<ITransientResourceHeap> transientHeap;
    ComPtr<IPipelineState> pipelineState;
    ComPtr<IRenderPassLayout> renderPass;
    ComPtr<IFramebuffer> framebuffer;

    ComPtr<IBufferResource> vertexBuffer;

    struct TextureInfo
    {
        ITextureResource::Extents extent;
        int numMipLevels;
        int arraySize;
        ITextureResource::SubresourceData const* initData;
    };

    void init(IDevice* device, UnitTestContext* context)
    {
        this->device = device;
        this->context = context;
    }

    void createRequiredResources(
        TextureInfo msaaTextureInfo,
        TextureInfo dstTextureInfo,
        Format format)
    {
        VertexStreamDesc vertexStreams[] = {
            {sizeof(Vertex), InputSlotClass::PerVertex, 0},
        };

        InputElementDesc inputElements[] = {
            // Vertex buffer data
            {"POSITION", 0, Format::R32G32B32_FLOAT, offsetof(Vertex, position), 0},
            {"COLOR", 0, Format::R32G32B32_FLOAT, offsetof(Vertex, color), 0},
        };

        ITextureResource::Desc msaaTexDesc = {};
        msaaTexDesc.type = IResource::Type::Texture2D;
        msaaTexDesc.numMipLevels = dstTextureInfo.numMipLevels;
        msaaTexDesc.arraySize = dstTextureInfo.arraySize;
        msaaTexDesc.size = dstTextureInfo.extent;
        msaaTexDesc.defaultState = ResourceState::RenderTarget;
        msaaTexDesc.allowedStates =
            ResourceStateSet(ResourceState::RenderTarget, ResourceState::ResolveSource);
        msaaTexDesc.format = format;
        msaaTexDesc.sampleDesc.numSamples = 4;

        GFX_CHECK_CALL_ABORT(device->createTextureResource(
            msaaTexDesc,
            msaaTextureInfo.initData,
            msaaTexture.writeRef()));

        ITextureResource::Desc dstTexDesc = {};
        dstTexDesc.type = IResource::Type::Texture2D;
        dstTexDesc.numMipLevels = dstTextureInfo.numMipLevels;
        dstTexDesc.arraySize = dstTextureInfo.arraySize;
        dstTexDesc.size = dstTextureInfo.extent;
        dstTexDesc.defaultState = ResourceState::ResolveDestination;
        dstTexDesc.allowedStates =
            ResourceStateSet(ResourceState::ResolveDestination, ResourceState::CopySource);
        dstTexDesc.format = format;

        GFX_CHECK_CALL_ABORT(device->createTextureResource(
            dstTexDesc,
            dstTextureInfo.initData,
            dstTexture.writeRef()));

        IInputLayout::Desc inputLayoutDesc = {};
        inputLayoutDesc.inputElementCount = SLANG_COUNT_OF(inputElements);
        inputLayoutDesc.inputElements = inputElements;
        inputLayoutDesc.vertexStreamCount = SLANG_COUNT_OF(vertexStreams);
        inputLayoutDesc.vertexStreams = vertexStreams;
        auto inputLayout = device->createInputLayout(inputLayoutDesc);
        SLANG_CHECK_ABORT(inputLayout != nullptr);

        vertexBuffer = createVertexBuffer(device);

        ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096;
        GFX_CHECK_CALL_ABORT(
            device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

        ComPtr<IShaderProgram> shaderProgram;
        slang::ProgramLayout* slangReflection;
        GFX_CHECK_CALL_ABORT(loadGraphicsProgram(
            device,
            shaderProgram,
            "resolve-resource-shader",
            "vertexMain",
            "fragmentMain",
            slangReflection));

        IFramebufferLayout::TargetLayout targetLayout;
        targetLayout.format = format;
        targetLayout.sampleCount = 4;

        IFramebufferLayout::Desc framebufferLayoutDesc;
        framebufferLayoutDesc.renderTargetCount = 1;
        framebufferLayoutDesc.renderTargets = &targetLayout;
        ComPtr<gfx::IFramebufferLayout> framebufferLayout =
            device->createFramebufferLayout(framebufferLayoutDesc);
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
        renderTargetAccess.initialState = ResourceState::RenderTarget;
        renderTargetAccess.finalState = ResourceState::ResolveSource;
        renderPassDesc.renderTargetAccess = &renderTargetAccess;
        GFX_CHECK_CALL_ABORT(device->createRenderPassLayout(renderPassDesc, renderPass.writeRef()));

        gfx::IResourceView::Desc colorBufferViewDesc;
        memset(&colorBufferViewDesc, 0, sizeof(colorBufferViewDesc));
        colorBufferViewDesc.format = format;
        colorBufferViewDesc.renderTarget.shape = gfx::IResource::Type::Texture2D;
        colorBufferViewDesc.type = gfx::IResourceView::Type::RenderTarget;
        auto rtv = device->createTextureView(msaaTexture, colorBufferViewDesc);

        gfx::IFramebuffer::Desc framebufferDesc;
        framebufferDesc.renderTargetCount = 1;
        framebufferDesc.depthStencilView = nullptr;
        framebufferDesc.renderTargetViews = rtv.readRef();
        framebufferDesc.layout = framebufferLayout;
        GFX_CHECK_CALL_ABORT(device->createFramebuffer(framebufferDesc, framebuffer.writeRef()));
    }

    void submitGPUWork(
        SubresourceRange msaaSubresource,
        SubresourceRange dstSubresource,
        ITextureResource::Extents extent)
    {
        Slang::ComPtr<ITransientResourceHeap> transientHeap;
        ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096;
        GFX_CHECK_CALL_ABORT(
            device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = device->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        auto renderEncoder = commandBuffer->encodeRenderCommands(renderPass, framebuffer);
        auto rootObject = renderEncoder->bindPipeline(pipelineState);

        gfx::Viewport viewport = {};
        viewport.maxZ = 1.0f;
        viewport.extentX = kWidth;
        viewport.extentY = kHeight;
        renderEncoder->setViewportAndScissor(viewport);

        renderEncoder->setVertexBuffer(0, vertexBuffer);
        renderEncoder->setPrimitiveTopology(PrimitiveTopology::TriangleList);
        renderEncoder->draw(kVertexCount, 0);
        renderEncoder->endEncoding();

        auto resourceEncoder = commandBuffer->encodeResourceCommands();

        resourceEncoder->resolveResource(
            msaaTexture,
            ResourceState::ResolveSource,
            msaaSubresource,
            dstTexture,
            ResourceState::ResolveDestination,
            dstSubresource);
        resourceEncoder->textureSubresourceBarrier(
            dstTexture,
            dstSubresource,
            ResourceState::ResolveDestination,
            ResourceState::CopySource);
        resourceEncoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    void checkTestResults(
        int pixelCount,
        int channelCount,
        const int* testXCoords,
        const int* testYCoords,
        float* testResults)
    {
        // Read texture values back from four specific pixels located within the triangles
        // and compare against expected values (because testing every single pixel will be too long
        // and tedious and requires maintaining reference images).
        ComPtr<ISlangBlob> resultBlob;
        size_t rowPitch = 0;
        size_t pixelSize = 0;
        GFX_CHECK_CALL_ABORT(device->readTextureResource(
            dstTexture,
            ResourceState::CopySource,
            resultBlob.writeRef(),
            &rowPitch,
            &pixelSize));
        auto result = (float*)resultBlob->getBufferPointer();

        int cursor = 0;
        for (int i = 0; i < pixelCount; ++i)
        {
            auto x = testXCoords[i];
            auto y = testYCoords[i];
            auto pixelPtr = result + x * channelCount + y * rowPitch / sizeof(float);
            for (int j = 0; j < channelCount; ++j)
            {
                testResults[cursor] = pixelPtr[j];
                cursor++;
            }
        }

        float expectedResult[] = {0.5f, 0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.0f, 0.0f,
                                  1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f,
                                  0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 1.0f};
        SLANG_CHECK(memcmp(testResults, expectedResult, 128) == 0);
    }
};

// TODO: Add more tests?

struct ResolveResourceSimple : BaseResolveResourceTest
{
    void run()
    {
        ITextureResource::Extents extent = {};
        extent.width = kWidth;
        extent.height = kHeight;
        extent.depth = 1;

        TextureInfo msaaTextureInfo = {extent, 1, 1, nullptr};
        TextureInfo dstTextureInfo = {extent, 1, 1, nullptr};

        createRequiredResources(msaaTextureInfo, dstTextureInfo, format);

        SubresourceRange msaaSubresource = {};
        msaaSubresource.aspectMask = TextureAspect::Color;
        msaaSubresource.mipLevel = 0;
        msaaSubresource.mipLevelCount = 1;
        msaaSubresource.baseArrayLayer = 0;
        msaaSubresource.layerCount = 1;

        SubresourceRange dstSubresource = {};
        dstSubresource.aspectMask = TextureAspect::Color;
        dstSubresource.mipLevel = 0;
        dstSubresource.mipLevelCount = 1;
        dstSubresource.baseArrayLayer = 0;
        dstSubresource.layerCount = 1;

        submitGPUWork(msaaSubresource, dstSubresource, extent);

        const int kPixelCount = 8;
        const int kChannelCount = 4;
        int testXCoords[kPixelCount] = {64, 127, 191, 64, 191, 64, 127, 191};
        int testYCoords[kPixelCount] = {64, 64, 64, 127, 127, 191, 191, 191};
        float testResults[kPixelCount * kChannelCount];

        checkTestResults(kPixelCount, kChannelCount, testXCoords, testYCoords, testResults);
    }
};

template<typename T>
void resolveResourceTestImpl(IDevice* device, UnitTestContext* context)
{
    T test;
    test.init(device, context);
    test.run();
}
} // namespace

namespace gfx_test
{
SLANG_UNIT_TEST(resolveResourceSimpleD3D12)
{
    runTestImpl(
        resolveResourceTestImpl<ResolveResourceSimple>,
        unitTestContext,
        Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(resolveResourceSimpleVulkan)
{
    runTestImpl(
        resolveResourceTestImpl<ResolveResourceSimple>,
        unitTestContext,
        Slang::RenderApiFlag::Vulkan);
}
} // namespace gfx_test
