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
struct BaseTextureViewTest
{
    IDevice* device;
    UnitTestContext* context;

    IResourceView::Type viewType;
    size_t alignedRowStride;

    RefPtr<TextureInfo> textureInfo;
    RefPtr<ValidationTextureFormatBase> validationFormat;

    ComPtr<ITextureResource> texture;
    ComPtr<IResourceView> textureView;
    ComPtr<IBufferResource> resultsBuffer;
    ComPtr<IResourceView> bufferView;

    ComPtr<ISamplerState> sampler;

    const void* expectedTextureData;

    void init(
        IDevice* device,
        UnitTestContext* context,
        Format format,
        RefPtr<ValidationTextureFormatBase> validationFormat,
        IResourceView::Type viewType,
        IResource::Type type)
    {
        this->device = device;
        this->context = context;
        this->validationFormat = validationFormat;
        this->viewType = viewType;

        this->textureInfo = new TextureInfo();
        this->textureInfo->format = format;
        this->textureInfo->textureType = type;
    }

    ResourceState getDefaultResourceStateForViewType(IResourceView::Type type)
    {
        switch (type)
        {
        case IResourceView::Type::RenderTarget:
            return ResourceState::RenderTarget;
        case IResourceView::Type::DepthStencil:
            return ResourceState::DepthWrite;
        case IResourceView::Type::ShaderResource:
            return ResourceState::ShaderResource;
        case IResourceView::Type::UnorderedAccess:
            return ResourceState::UnorderedAccess;
        case IResourceView::Type::AccelerationStructure:
            return ResourceState::AccelerationStructure;
        default:
            return ResourceState::Undefined;
        }
    }

    String getShaderEntryPoint()
    {
        String base = "resourceViewTest";
        String shape;
        String view;

        switch (textureInfo->textureType)
        {
        case IResource::Type::Texture1D:
            shape = "1D";
            break;
        case IResource::Type::Texture2D:
            shape = "2D";
            break;
        case IResource::Type::Texture3D:
            shape = "3D";
            break;
        case IResource::Type::TextureCube:
            shape = "Cube";
            break;
        default:
            assert(!"Invalid texture shape");
            SLANG_CHECK_ABORT(false);
        }

        switch (viewType)
        {
        case IResourceView::Type::RenderTarget:
            view = "Render";
            break;
        case IResourceView::Type::DepthStencil:
            view = "Depth";
            break;
        case IResourceView::Type::ShaderResource:
            view = "Shader";
            break;
        case IResourceView::Type::UnorderedAccess:
            view = "Unordered";
            break;
        case IResourceView::Type::AccelerationStructure:
            view = "Accel";
            break;
        default:
            assert(!"Invalid resource view");
            SLANG_CHECK_ABORT(false);
        }

        return base + shape + view;
    }
};

// used for shaderresource and unorderedaccess
struct ShaderAndUnorderedTests : BaseTextureViewTest
{
    void createRequiredResources()
    {
        ITextureResource::Desc textureDesc = {};
        textureDesc.type = textureInfo->textureType;
        textureDesc.numMipLevels = textureInfo->mipLevelCount;
        textureDesc.arraySize = textureInfo->arrayLayerCount;
        textureDesc.size = textureInfo->extents;
        textureDesc.defaultState = getDefaultResourceStateForViewType(viewType);
        textureDesc.allowedStates = ResourceStateSet(
            textureDesc.defaultState,
            ResourceState::CopySource,
            ResourceState::CopyDestination);
        textureDesc.format = textureInfo->format;

        GFX_CHECK_CALL_ABORT(device->createTextureResource(
            textureDesc,
            textureInfo->subresourceDatas.getBuffer(),
            texture.writeRef()));

        IResourceView::Desc textureViewDesc = {};
        textureViewDesc.type = viewType;
        textureViewDesc.format =
            textureDesc.format; // TODO: Handle typeless formats - gfxIsTypelessFormat(format) ?
                                // convertTypelessFormat(format) : format;
        GFX_CHECK_CALL_ABORT(
            device->createTextureView(texture, textureViewDesc, textureView.writeRef()));

        auto texelSize = getTexelSize(textureInfo->format);
        size_t alignment;
        device->getTextureRowAlignment(&alignment);
        alignedRowStride =
            (textureInfo->extents.width * texelSize + alignment - 1) & ~(alignment - 1);
        IBufferResource::Desc bufferDesc = {};
        // All of the values read back from the shader will be uint32_t
        bufferDesc.sizeInBytes = textureDesc.size.width * textureDesc.size.height *
                                 textureDesc.size.depth * texelSize * sizeof(uint32_t);
        bufferDesc.format = Format::Unknown;
        bufferDesc.elementSize = sizeof(uint32_t);
        bufferDesc.defaultState = ResourceState::UnorderedAccess;
        bufferDesc.allowedStates = ResourceStateSet(
            bufferDesc.defaultState,
            ResourceState::CopyDestination,
            ResourceState::CopySource);
        bufferDesc.memoryType = MemoryType::DeviceLocal;

        GFX_CHECK_CALL_ABORT(
            device->createBufferResource(bufferDesc, nullptr, resultsBuffer.writeRef()));

        IResourceView::Desc bufferViewDesc = {};
        bufferViewDesc.type = IResourceView::Type::UnorderedAccess;
        bufferViewDesc.format = Format::Unknown;
        GFX_CHECK_CALL_ABORT(device->createBufferView(
            resultsBuffer,
            nullptr,
            bufferViewDesc,
            bufferView.writeRef()));
    }

    void submitShaderWork(const char* entryPoint)
    {
        Slang::ComPtr<ITransientResourceHeap> transientHeap;
        ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096;
        GFX_CHECK_CALL_ABORT(
            device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

        ComPtr<IShaderProgram> shaderProgram;
        slang::ProgramLayout* slangReflection;
        GFX_CHECK_CALL_ABORT(loadComputeProgram(
            device,
            shaderProgram,
            "trivial-copy-textures",
            entryPoint,
            slangReflection));

        ComputePipelineStateDesc pipelineDesc = {};
        pipelineDesc.program = shaderProgram.get();
        ComPtr<gfx::IPipelineState> pipelineState;
        GFX_CHECK_CALL_ABORT(
            device->createComputePipelineState(pipelineDesc, pipelineState.writeRef()));

        // We have done all the set up work, now it is time to start recording a command buffer for
        // GPU execution.
        {
            ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
            auto queue = device->createCommandQueue(queueDesc);

            auto commandBuffer = transientHeap->createCommandBuffer();
            auto encoder = commandBuffer->encodeComputeCommands();

            auto rootObject = encoder->bindPipeline(pipelineState);

            ShaderCursor entryPointCursor(
                rootObject->getEntryPoint(0)); // get a cursor the the first entry-point.

            auto width = textureInfo->extents.width;
            auto height = textureInfo->extents.height;
            auto depth = textureInfo->extents.depth;

            entryPointCursor["width"].setData(width);
            entryPointCursor["height"].setData(height);
            entryPointCursor["depth"].setData(depth);

            // Bind texture view to the entry point
            entryPointCursor["resourceView"].setResource(
                textureView); // TODO: Bind nullptr and make sure it doesn't splut - should be 0
                              // everywhere
            entryPointCursor["testResults"].setResource(bufferView);

            if (sampler)
                entryPointCursor["sampler"].setSampler(
                    sampler); // TODO: Bind nullptr and make sure it doesn't splut

            auto bufferElementCount = width * height * depth;
            encoder->dispatchCompute(bufferElementCount, 1, 1);
            encoder->endEncoding();
            commandBuffer->close();
            queue->executeCommandBuffer(commandBuffer);
            queue->waitOnHost();
        }
    }

    void validateTextureValues(ValidationTextureData actual, ValidationTextureData original)
    {
        // TODO: needs to be extended to cover mip levels and array layers
        for (GfxIndex x = 0; x < actual.extents.width; ++x)
        {
            for (GfxIndex y = 0; y < actual.extents.height; ++y)
            {
                for (GfxIndex z = 0; z < actual.extents.depth; ++z)
                {
                    auto actualBlock = (uint8_t*)actual.getBlockAt(x, y, z);
                    for (Int i = 0; i < 4; ++i)
                    {
                        SLANG_CHECK(actualBlock[i] == 1);
                    }
                }
            }
        }
    }

    void checkTestResults()
    {
        // Shader resources are read-only, so we don't need to check that writes to the resource
        // were correct.
        if (viewType != IResourceView::Type::ShaderResource)
        {
            ComPtr<ISlangBlob> textureBlob;
            size_t rowPitch;
            size_t pixelSize;
            GFX_CHECK_CALL_ABORT(device->readTextureResource(
                texture,
                ResourceState::CopySource,
                textureBlob.writeRef(),
                &rowPitch,
                &pixelSize));
            auto textureValues = (uint8_t*)textureBlob->getBufferPointer();

            ValidationTextureData textureResults;
            textureResults.extents = textureInfo->extents;
            textureResults.textureData = textureValues;
            textureResults.strides.x = (uint32_t)pixelSize;
            textureResults.strides.y = (uint32_t)rowPitch;
            textureResults.strides.z = textureResults.extents.height * textureResults.strides.y;

            ValidationTextureData originalData;
            originalData.extents = textureInfo->extents;
            originalData.textureData = textureInfo->subresourceDatas.getBuffer();
            originalData.strides.x = (uint32_t)pixelSize;
            originalData.strides.y = textureInfo->extents.width * originalData.strides.x;
            originalData.strides.z = textureInfo->extents.height * originalData.strides.y;

            validateTextureValues(textureResults, originalData);
        }

        ComPtr<ISlangBlob> bufferBlob;
        GFX_CHECK_CALL_ABORT(device->readBufferResource(
            resultsBuffer,
            0,
            resultsBuffer->getDesc()->sizeInBytes,
            bufferBlob.writeRef()));
        auto results = (uint32_t*)bufferBlob->getBufferPointer();

        auto elementCount = textureInfo->extents.width * textureInfo->extents.height *
                            textureInfo->extents.depth * 4;
        auto castedTextureData = (uint8_t*)expectedTextureData;
        for (Int i = 0; i < elementCount; ++i)
        {
            SLANG_CHECK(results[i] == castedTextureData[i]);
        }
    }

    void run()
    {
        // TODO: Should test with samplers
        //             ISamplerState::Desc samplerDesc;
        //             sampler = device->createSamplerState(samplerDesc);

        // TODO: Should test multiple mip levels and array layers
        textureInfo->extents.width = 4;
        textureInfo->extents.height =
            (textureInfo->textureType == IResource::Type::Texture1D) ? 1 : 4;
        textureInfo->extents.depth =
            (textureInfo->textureType != IResource::Type::Texture3D) ? 1 : 2;
        textureInfo->mipLevelCount = 1;
        textureInfo->arrayLayerCount = 1;
        generateTextureData(textureInfo, validationFormat);

        // We need to save the pointer to the original texture data for results checking because the
        // texture will be overwritten during testing (if the texture can be written to).
        expectedTextureData = textureInfo->subresourceDatas[getSubresourceIndex(0, 1, 0)].data;

        createRequiredResources();
        auto entryPointName = getShaderEntryPoint();
        // printf("%s\n", entryPointName.getBuffer());
        submitShaderWork(entryPointName.getBuffer());

        checkTestResults();
    }
};

// used for rendertarget and depthstencil
struct RenderTargetTests : BaseTextureViewTest
{
    struct Vertex
    {
        float position[3];
        float color[3];
    };

    const int kVertexCount = 12;
    const Vertex kVertexData[12] = {
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

    int sampleCount = 1;

    ComPtr<ITransientResourceHeap> transientHeap;
    ComPtr<IPipelineState> pipelineState;
    ComPtr<IRenderPassLayout> renderPass;
    ComPtr<IFramebuffer> framebuffer;

    ComPtr<ITextureResource> sampledTexture;
    ComPtr<IBufferResource> vertexBuffer;

    void createRequiredResources()
    {
        IBufferResource::Desc vertexBufferDesc;
        vertexBufferDesc.type = IResource::Type::Buffer;
        vertexBufferDesc.sizeInBytes = kVertexCount * sizeof(Vertex);
        vertexBufferDesc.defaultState = ResourceState::VertexBuffer;
        vertexBufferDesc.allowedStates = ResourceState::VertexBuffer;
        vertexBuffer = device->createBufferResource(vertexBufferDesc, &kVertexData[0]);
        SLANG_CHECK_ABORT(vertexBuffer != nullptr);

        VertexStreamDesc vertexStreams[] = {
            {sizeof(Vertex), InputSlotClass::PerVertex, 0},
        };

        InputElementDesc inputElements[] = {
            // Vertex buffer data
            {"POSITION", 0, Format::R32G32B32_FLOAT, offsetof(Vertex, position), 0},
            {"COLOR", 0, Format::R32G32B32_FLOAT, offsetof(Vertex, color), 0},
        };

        ITextureResource::Desc sampledTexDesc = {};
        sampledTexDesc.type = textureInfo->textureType;
        sampledTexDesc.numMipLevels = textureInfo->mipLevelCount;
        sampledTexDesc.arraySize = textureInfo->arrayLayerCount;
        sampledTexDesc.size = textureInfo->extents;
        sampledTexDesc.defaultState = getDefaultResourceStateForViewType(viewType);
        sampledTexDesc.allowedStates = ResourceStateSet(
            sampledTexDesc.defaultState,
            ResourceState::ResolveSource,
            ResourceState::CopySource);
        sampledTexDesc.format = textureInfo->format;
        sampledTexDesc.sampleDesc.numSamples = sampleCount;

        GFX_CHECK_CALL_ABORT(device->createTextureResource(
            sampledTexDesc,
            textureInfo->subresourceDatas.getBuffer(),
            sampledTexture.writeRef()));

        ITextureResource::Desc texDesc = {};
        texDesc.type = textureInfo->textureType;
        texDesc.numMipLevels = textureInfo->mipLevelCount;
        texDesc.arraySize = textureInfo->arrayLayerCount;
        texDesc.size = textureInfo->extents;
        texDesc.defaultState = ResourceState::ResolveDestination;
        texDesc.allowedStates =
            ResourceStateSet(ResourceState::ResolveDestination, ResourceState::CopySource);
        texDesc.format = textureInfo->format;

        GFX_CHECK_CALL_ABORT(device->createTextureResource(
            texDesc,
            textureInfo->subresourceDatas.getBuffer(),
            texture.writeRef()));

        IInputLayout::Desc inputLayoutDesc = {};
        inputLayoutDesc.inputElementCount = SLANG_COUNT_OF(inputElements);
        inputLayoutDesc.inputElements = inputElements;
        inputLayoutDesc.vertexStreamCount = SLANG_COUNT_OF(vertexStreams);
        inputLayoutDesc.vertexStreams = vertexStreams;
        auto inputLayout = device->createInputLayout(inputLayoutDesc);
        SLANG_CHECK_ABORT(inputLayout != nullptr);

        ITransientResourceHeap::Desc transientHeapDesc = {};
        transientHeapDesc.constantBufferSize = 4096;
        GFX_CHECK_CALL_ABORT(
            device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

        ComPtr<IShaderProgram> shaderProgram;
        slang::ProgramLayout* slangReflection;
        GFX_CHECK_CALL_ABORT(loadGraphicsProgram(
            device,
            shaderProgram,
            "trivial-copy-textures",
            "vertexMain",
            "fragmentMain",
            slangReflection));

        IFramebufferLayout::TargetLayout targetLayout;
        targetLayout.format = textureInfo->format;
        targetLayout.sampleCount = sampleCount;

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
        renderTargetAccess.initialState = getDefaultResourceStateForViewType(viewType);
        renderTargetAccess.finalState = ResourceState::ResolveSource;
        renderPassDesc.renderTargetAccess = &renderTargetAccess;
        GFX_CHECK_CALL_ABORT(device->createRenderPassLayout(renderPassDesc, renderPass.writeRef()));

        gfx::IResourceView::Desc colorBufferViewDesc;
        memset(&colorBufferViewDesc, 0, sizeof(colorBufferViewDesc));
        colorBufferViewDesc.format = textureInfo->format;
        colorBufferViewDesc.renderTarget.shape = textureInfo->textureType; // TODO: TextureCube?
        colorBufferViewDesc.type = viewType;
        auto rtv = device->createTextureView(sampledTexture, colorBufferViewDesc);

        gfx::IFramebuffer::Desc framebufferDesc;
        framebufferDesc.renderTargetCount = 1;
        framebufferDesc.depthStencilView = nullptr;
        framebufferDesc.renderTargetViews = rtv.readRef();
        framebufferDesc.layout = framebufferLayout;
        GFX_CHECK_CALL_ABORT(device->createFramebuffer(framebufferDesc, framebuffer.writeRef()));

        auto texelSize = getTexelSize(textureInfo->format);
        size_t alignment;
        device->getTextureRowAlignment(&alignment);
        alignedRowStride =
            (textureInfo->extents.width * texelSize + alignment - 1) & ~(alignment - 1);
    }

    void submitShaderWork(const char* entryPointName)
    {
        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = device->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        auto renderEncoder = commandBuffer->encodeRenderCommands(renderPass, framebuffer);
        auto rootObject = renderEncoder->bindPipeline(pipelineState);

        gfx::Viewport viewport = {};
        viewport.maxZ = (float)textureInfo->extents.depth;
        viewport.extentX = (float)textureInfo->extents.width;
        viewport.extentY = (float)textureInfo->extents.height;
        renderEncoder->setViewportAndScissor(viewport);

        renderEncoder->setVertexBuffer(0, vertexBuffer);
        renderEncoder->setPrimitiveTopology(PrimitiveTopology::TriangleList);
        renderEncoder->draw(kVertexCount, 0);
        renderEncoder->endEncoding();

        auto resourceEncoder = commandBuffer->encodeResourceCommands();

        if (sampleCount > 1)
        {
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

            resourceEncoder->resolveResource(
                sampledTexture,
                ResourceState::ResolveSource,
                msaaSubresource,
                texture,
                ResourceState::ResolveDestination,
                dstSubresource);
            resourceEncoder->textureBarrier(
                texture,
                ResourceState::ResolveDestination,
                ResourceState::CopySource);
        }
        else
        {
            resourceEncoder->textureBarrier(
                sampledTexture,
                ResourceState::ResolveSource,
                ResourceState::CopySource);
        }
        resourceEncoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    // TODO: Should take a value indicating the slice that was rendered into
    // TODO: Needs to handle either the correct slice or array layer (will not always check z)
    void validateTextureValues(ValidationTextureData actual)
    {
        for (GfxIndex x = 0; x < actual.extents.width; ++x)
        {
            for (GfxIndex y = 0; y < actual.extents.height; ++y)
            {
                for (GfxIndex z = 0; z < actual.extents.depth; ++z)
                {
                    auto actualBlock = (float*)actual.getBlockAt(x, y, z);
                    for (Int i = 0; i < 4; ++i)
                    {
                        if (z == 0)
                        {
                            // Slice being rendered into
                            SLANG_CHECK(actualBlock[i] == (float)i + 1);
                        }
                        else
                        {
                            SLANG_CHECK(actualBlock[i] == 0.0f);
                        }
                    }
                }
            }
        }
    }

    void checkTestResults()
    {
        ComPtr<ISlangBlob> textureBlob;
        size_t rowPitch;
        size_t pixelSize;
        if (sampleCount > 1)
        {
            GFX_CHECK_CALL_ABORT(device->readTextureResource(
                texture,
                ResourceState::CopySource,
                textureBlob.writeRef(),
                &rowPitch,
                &pixelSize));
        }
        else
        {
            GFX_CHECK_CALL_ABORT(device->readTextureResource(
                sampledTexture,
                ResourceState::CopySource,
                textureBlob.writeRef(),
                &rowPitch,
                &pixelSize));
        }
        auto textureValues = (float*)textureBlob->getBufferPointer();

        ValidationTextureData textureResults;
        textureResults.extents = textureInfo->extents;
        textureResults.textureData = textureValues;
        textureResults.strides.x = (uint32_t)pixelSize;
        textureResults.strides.y = (uint32_t)rowPitch;
        textureResults.strides.z = textureResults.extents.height * textureResults.strides.y;

        validateTextureValues(textureResults);
    }

    void run()
    {
        auto entryPointName = getShaderEntryPoint();
        //             printf("%s\n", entryPointName.getBuffer());

        // TODO: Sampler state and null state?
        //             ISamplerState::Desc samplerDesc;
        //             sampler = device->createSamplerState(samplerDesc);

        textureInfo->extents.width = 4;
        textureInfo->extents.height =
            (textureInfo->textureType == IResource::Type::Texture1D) ? 1 : 4;
        textureInfo->extents.depth =
            (textureInfo->textureType != IResource::Type::Texture3D) ? 1 : 2;
        textureInfo->mipLevelCount = 1;
        textureInfo->arrayLayerCount = 1;
        generateTextureData(textureInfo, validationFormat);

        // We need to save the pointer to the original texture data for results checking because the
        // texture will be overwritten during testing (if the texture can be written to).
        expectedTextureData = textureInfo->subresourceDatas[getSubresourceIndex(0, 1, 0)].data;

        createRequiredResources();
        submitShaderWork(entryPointName.getBuffer());

        checkTestResults();
    }
};

void shaderAndUnorderedTestImpl(IDevice* device, UnitTestContext* context)
{
    // TODO: Buffer and TextureCube
    for (Int i = 2; i < (int32_t)IResource::Type::TextureCube; ++i)
    {
        for (Int j = 3; j < (int32_t)IResourceView::Type::AccelerationStructure; ++j)
        {
            auto shape = (IResource::Type)i;
            auto view = (IResourceView::Type)j;
            auto format = Format::R8G8B8A8_UINT;
            auto validationFormat = getValidationTextureFormat(format);
            if (!validationFormat)
                SLANG_CHECK_ABORT(false);

            ShaderAndUnorderedTests test;
            test.init(device, context, format, validationFormat, view, shape);
            test.run();
        }
    }
}

void renderTargetTestImpl(IDevice* device, UnitTestContext* context)
{
    // TODO: Buffer and TextureCube
    for (Int i = 2; i < (int32_t)IResource::Type::TextureCube; ++i)
    {
        auto shape = (IResource::Type)i;
        auto view = IResourceView::Type::RenderTarget;
        auto format = Format::R32G32B32A32_FLOAT;
        auto validationFormat = getValidationTextureFormat(format);
        if (!validationFormat)
            SLANG_CHECK_ABORT(false);

        RenderTargetTests test;
        test.init(device, context, format, validationFormat, view, shape);
        test.run();
    }
}

SLANG_UNIT_TEST(shaderAndUnorderedAccessTests)
{
    runTestImpl(shaderAndUnorderedTestImpl, unitTestContext, Slang::RenderApiFlag::D3D12);
    runTestImpl(shaderAndUnorderedTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}

SLANG_UNIT_TEST(renderTargetTests)
{
    runTestImpl(renderTargetTestImpl, unitTestContext, Slang::RenderApiFlag::D3D12);
    runTestImpl(renderTargetTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}
} // namespace gfx_test

// 1D + array + multisample, ditto for 2D, ditto for 3D
// one test with something bound, one test with nothing bound, one test with subset of layers (set
// values in SubresourceRange and assign in desc)
