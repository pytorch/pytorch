#include "core/slang-basic.h"
#include "examples/example-base/example-base.h"
#include "gfx-util/shader-cursor.h"
#include "platform/vector-math.h"
#include "platform/window.h"
#include "slang-com-ptr.h"
#include "slang-gfx.h"
#include "slang.h"

using namespace gfx;
using namespace Slang;

static const ExampleResources resourceBase("autodiff-texture");

struct Vertex
{
    float position[3];
};

static const int kVertexCount = 4;
static const Vertex kVertexData[kVertexCount] = {
    {{0, 0, 0}},
    {{0, 1, 0}},
    {{1, 0, 0}},
    {{1, 1, 0}},
};

struct AutoDiffTexture : public WindowedAppBase
{

    List<uint32_t> mipMapOffset;
    int textureWidth;
    int textureHeight;

    void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
    {
        if (diagnosticsBlob != nullptr)
        {
            printf("%s", (const char*)diagnosticsBlob->getBufferPointer());
        }
    }

    gfx::Result loadRenderProgram(
        gfx::IDevice* device,
        const char* fileName,
        const char* fragmentShader,
        gfx::IShaderProgram** outProgram)
    {
        ComPtr<slang::ISession> slangSession;
        slangSession = device->getSlangSession();

        ComPtr<slang::IBlob> diagnosticsBlob;
        Slang::String path = resourceBase.resolveResource(fileName);
        slang::IModule* module =
            slangSession->loadModule(path.getBuffer(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        if (!module)
            return SLANG_FAIL;

        ComPtr<slang::IEntryPoint> vertexEntryPoint;
        SLANG_RETURN_ON_FAIL(
            module->findEntryPointByName("vertexMain", vertexEntryPoint.writeRef()));
        ComPtr<slang::IEntryPoint> fragmentEntryPoint;
        SLANG_RETURN_ON_FAIL(
            module->findEntryPointByName(fragmentShader, fragmentEntryPoint.writeRef()));

        Slang::List<slang::IComponentType*> componentTypes;
        componentTypes.add(module);
        int entryPointCount = 0;
        int vertexEntryPointIndex = entryPointCount++;
        componentTypes.add(vertexEntryPoint);

        int fragmentEntryPointIndex = entryPointCount++;
        componentTypes.add(fragmentEntryPoint);

        ComPtr<slang::IComponentType> linkedProgram;
        SlangResult result = slangSession->createCompositeComponentType(
            componentTypes.getBuffer(),
            componentTypes.getCount(),
            linkedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        if (isTestMode())
        {
            printEntrypointHashes(componentTypes.getCount() - 1, 1, linkedProgram);
        }

        gfx::IShaderProgram::Desc programDesc = {};
        programDesc.slangGlobalScope = linkedProgram;
        SLANG_RETURN_ON_FAIL(device->createProgram(programDesc, outProgram));

        return SLANG_OK;
    }

    gfx::Result loadComputeProgram(
        gfx::IDevice* device,
        const char* fileName,
        gfx::IShaderProgram** outProgram)
    {
        ComPtr<slang::ISession> slangSession;
        slangSession = device->getSlangSession();

        ComPtr<slang::IBlob> diagnosticsBlob;
        Slang::String path = resourceBase.resolveResource(fileName);
        slang::IModule* module =
            slangSession->loadModule(path.getBuffer(), diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        if (!module)
            return SLANG_FAIL;

        Slang::List<slang::IComponentType*> componentTypes;
        componentTypes.add(module);
        ComPtr<slang::IEntryPoint> computeEntryPoint;
        SLANG_RETURN_ON_FAIL(
            module->findEntryPointByName("computeMain", computeEntryPoint.writeRef()));
        componentTypes.add(computeEntryPoint);

        ComPtr<slang::IComponentType> linkedProgram;
        SlangResult result = slangSession->createCompositeComponentType(
            componentTypes.getBuffer(),
            componentTypes.getCount(),
            linkedProgram.writeRef(),
            diagnosticsBlob.writeRef());
        diagnoseIfNeeded(diagnosticsBlob);
        SLANG_RETURN_ON_FAIL(result);

        if (isTestMode())
        {
            printEntrypointHashes(componentTypes.getCount() - 1, 1, linkedProgram);
        }

        gfx::IShaderProgram::Desc programDesc = {};
        programDesc.slangGlobalScope = linkedProgram;
        SLANG_RETURN_ON_FAIL(device->createProgram(programDesc, outProgram));

        return SLANG_OK;
    }

    ComPtr<gfx::IPipelineState> gRefPipelineState;
    ComPtr<gfx::IPipelineState> gIterPipelineState;
    ComPtr<gfx::IPipelineState> gReconstructPipelineState;
    ComPtr<gfx::IPipelineState> gConvertPipelineState;
    ComPtr<gfx::IPipelineState> gBuildMipPipelineState;
    ComPtr<gfx::IPipelineState> gLearnMipPipelineState;
    ComPtr<gfx::IPipelineState> gDrawQuadPipelineState;

    ComPtr<gfx::ITextureResource> gLearningTexture;
    ComPtr<gfx::IResourceView> gLearningTextureSRV;
    List<ComPtr<gfx::IResourceView>> gLearningTextureUAVs;

    ComPtr<gfx::ITextureResource> gDiffTexture;
    ComPtr<gfx::IResourceView> gDiffTextureSRV;
    List<ComPtr<gfx::IResourceView>> gDiffTextureUAVs;

    ComPtr<gfx::IBufferResource> gVertexBuffer;
    ComPtr<gfx::IResourceView> gTexView;
    ComPtr<gfx::ISamplerState> gSampler;
    ComPtr<gfx::IFramebuffer> gRefFrameBuffer;
    ComPtr<gfx::IFramebuffer> gIterFrameBuffer;

    ComPtr<gfx::ITextureResource> gDepthTexture;
    ComPtr<gfx::IResourceView> gDepthTextureView;

    ComPtr<gfx::IResourceView> gIterImageDSV;

    ComPtr<gfx::ITextureResource> gIterImage;
    ComPtr<gfx::IResourceView> gIterImageSRV;
    ComPtr<gfx::IResourceView> gIterImageRTV;

    ComPtr<gfx::ITextureResource> gRefImage;
    ComPtr<gfx::IResourceView> gRefImageSRV;
    ComPtr<gfx::IResourceView> gRefImageRTV;

    ComPtr<gfx::IBufferResource> gAccumulateBuffer;
    ComPtr<gfx::IBufferResource> gReconstructBuffer;
    ComPtr<gfx::IResourceView> gAccumulateBufferView;
    ComPtr<gfx::IResourceView> gReconstructBufferView;

    ClearValue kClearValue;
    bool resetLearntTexture = false;

    ComPtr<gfx::ITextureResource> createRenderTargetTexture(
        gfx::Format format,
        int w,
        int h,
        int levels)
    {
        gfx::ITextureResource::Desc textureDesc = {};
        textureDesc.allowedStates.add(ResourceState::ShaderResource);
        textureDesc.allowedStates.add(ResourceState::UnorderedAccess);
        textureDesc.allowedStates.add(ResourceState::RenderTarget);
        textureDesc.defaultState = ResourceState::RenderTarget;
        textureDesc.format = format;
        textureDesc.numMipLevels = levels;
        textureDesc.type = gfx::IResource::Type::Texture2D;
        textureDesc.size.width = w;
        textureDesc.size.height = h;
        textureDesc.size.depth = 1;
        textureDesc.optimalClearValue = &kClearValue;
        return gDevice->createTextureResource(textureDesc, nullptr);
    }
    ComPtr<gfx::ITextureResource> createDepthTexture()
    {
        gfx::ITextureResource::Desc textureDesc = {};
        textureDesc.allowedStates.add(ResourceState::DepthWrite);
        textureDesc.defaultState = ResourceState::DepthWrite;
        textureDesc.format = gfx::Format::D32_FLOAT;
        textureDesc.numMipLevels = 1;
        textureDesc.type = gfx::IResource::Type::Texture2D;
        textureDesc.size.width = windowWidth;
        textureDesc.size.height = windowHeight;
        textureDesc.size.depth = 1;
        ClearValue clearValue = {};
        textureDesc.optimalClearValue = &clearValue;
        return gDevice->createTextureResource(textureDesc, nullptr);
    }
    ComPtr<gfx::IFramebuffer> createRenderTargetFramebuffer(IResourceView* tex)
    {
        IFramebuffer::Desc desc = {};
        desc.layout = gFramebufferLayout.get();
        desc.renderTargetCount = 1;
        desc.renderTargetViews = &tex;
        desc.depthStencilView = gDepthTextureView;
        return gDevice->createFramebuffer(desc);
    }
    ComPtr<gfx::IResourceView> createRTV(ITextureResource* tex, Format f)
    {
        IResourceView::Desc rtvDesc = {};
        rtvDesc.type = IResourceView::Type::RenderTarget;
        rtvDesc.subresourceRange.mipLevelCount = 1;
        rtvDesc.format = f;
        rtvDesc.renderTarget.shape = gfx::IResource::Type::Texture2D;
        return gDevice->createTextureView(tex, rtvDesc);
    }
    ComPtr<gfx::IResourceView> createDSV(ITextureResource* tex)
    {
        IResourceView::Desc dsvDesc = {};
        dsvDesc.type = IResourceView::Type::DepthStencil;
        dsvDesc.subresourceRange.mipLevelCount = 1;
        dsvDesc.format = Format::D32_FLOAT;
        dsvDesc.renderTarget.shape = gfx::IResource::Type::Texture2D;
        return gDevice->createTextureView(tex, dsvDesc);
    }
    ComPtr<gfx::IResourceView> createSRV(ITextureResource* tex)
    {
        IResourceView::Desc rtvDesc = {};
        rtvDesc.type = IResourceView::Type::ShaderResource;
        return gDevice->createTextureView(tex, rtvDesc);
    }
    ComPtr<gfx::IPipelineState> createRenderPipelineState(
        IInputLayout* inputLayout,
        IShaderProgram* program)
    {
        GraphicsPipelineStateDesc desc;
        desc.inputLayout = inputLayout;
        desc.program = program;
        desc.rasterizer.cullMode = gfx::CullMode::None;
        desc.framebufferLayout = gFramebufferLayout;
        auto pipelineState = gDevice->createGraphicsPipelineState(desc);
        return pipelineState;
    }
    ComPtr<gfx::IPipelineState> createComputePipelineState(IShaderProgram* program)
    {
        ComputePipelineStateDesc desc = {};
        desc.program = program;
        auto pipelineState = gDevice->createComputePipelineState(desc);
        return pipelineState;
    }
    ComPtr<gfx::IResourceView> createUAV(IBufferResource* buffer)
    {
        IResourceView::Desc desc = {};
        desc.type = IResourceView::Type::UnorderedAccess;
        return gDevice->createBufferView(buffer, nullptr, desc);
    }
    ComPtr<gfx::IResourceView> createUAV(ITextureResource* texture, int level)
    {
        IResourceView::Desc desc = {};
        desc.type = IResourceView::Type::UnorderedAccess;
        desc.subresourceRange.layerCount = 1;
        desc.subresourceRange.mipLevel = level;
        desc.subresourceRange.baseArrayLayer = 0;
        return gDevice->createTextureView(texture, desc);
    }
    Slang::Result initialize()
    {
        SLANG_RETURN_ON_FAIL(initializeBase("autodiff-texture", 1024, 768));
        srand(20421);

        if (!isTestMode())
        {
            gWindow->events.keyPress = [this](platform::KeyEventArgs& e)
            {
                if (e.keyChar == 'R' || e.keyChar == 'r')
                    resetLearntTexture = true;
            };
        }

        kClearValue.color.floatValues[0] = 0.3f;
        kClearValue.color.floatValues[1] = 0.5f;
        kClearValue.color.floatValues[2] = 0.7f;
        kClearValue.color.floatValues[3] = 1.0f;

        platform::Rect clientRect{};
        if (isTestMode())
        {
            clientRect.width = 1024;
            clientRect.height = 768;
        }
        else
        {
            clientRect = getWindow()->getClientRect();
        }

        windowWidth = clientRect.width;
        windowHeight = clientRect.height;

        InputElementDesc inputElements[] = {
            {"POSITION", 0, Format::R32G32B32_FLOAT, offsetof(Vertex, position)}};
        auto inputLayout = gDevice->createInputLayout(sizeof(Vertex), &inputElements[0], 1);
        if (!inputLayout)
            return SLANG_FAIL;

        IBufferResource::Desc vertexBufferDesc;
        vertexBufferDesc.type = IResource::Type::Buffer;
        vertexBufferDesc.sizeInBytes = kVertexCount * sizeof(Vertex);
        vertexBufferDesc.defaultState = ResourceState::VertexBuffer;
        gVertexBuffer = gDevice->createBufferResource(vertexBufferDesc, &kVertexData[0]);
        if (!gVertexBuffer)
            return SLANG_FAIL;

        {
            ComPtr<IShaderProgram> shaderProgram;
            SLANG_RETURN_ON_FAIL(loadRenderProgram(
                gDevice,
                "train.slang",
                "fragmentMain",
                shaderProgram.writeRef()));
            gRefPipelineState = createRenderPipelineState(inputLayout, shaderProgram);
        }
        {
            ComPtr<IShaderProgram> shaderProgram;
            SLANG_RETURN_ON_FAIL(loadRenderProgram(
                gDevice,
                "train.slang",
                "diffFragmentMain",
                shaderProgram.writeRef()));
            gIterPipelineState = createRenderPipelineState(inputLayout, shaderProgram);
        }
        {
            ComPtr<IShaderProgram> shaderProgram;
            SLANG_RETURN_ON_FAIL(loadRenderProgram(
                gDevice,
                "draw-quad.slang",
                "fragmentMain",
                shaderProgram.writeRef()));
            gDrawQuadPipelineState = createRenderPipelineState(inputLayout, shaderProgram);
        }
        {
            ComPtr<IShaderProgram> shaderProgram;
            SLANG_RETURN_ON_FAIL(
                loadComputeProgram(gDevice, "reconstruct.slang", shaderProgram.writeRef()));
            gReconstructPipelineState = createComputePipelineState(shaderProgram);
        }
        {
            ComPtr<IShaderProgram> shaderProgram;
            SLANG_RETURN_ON_FAIL(
                loadComputeProgram(gDevice, "convert.slang", shaderProgram.writeRef()));
            gConvertPipelineState = createComputePipelineState(shaderProgram);
        }
        {
            ComPtr<IShaderProgram> shaderProgram;
            SLANG_RETURN_ON_FAIL(
                loadComputeProgram(gDevice, "buildmip.slang", shaderProgram.writeRef()));
            gBuildMipPipelineState = createComputePipelineState(shaderProgram);
        }
        {
            ComPtr<IShaderProgram> shaderProgram;
            SLANG_RETURN_ON_FAIL(
                loadComputeProgram(gDevice, "learnmip.slang", shaderProgram.writeRef()));
            gLearnMipPipelineState = createComputePipelineState(shaderProgram);
        }

        Slang::String imagePath = resourceBase.resolveResource("checkerboard.jpg");
        gTexView = createTextureFromFile(imagePath.getBuffer(), textureWidth, textureHeight);
        initMipOffsets(textureWidth, textureHeight);

        gfx::IBufferResource::Desc bufferDesc = {};
        bufferDesc.allowedStates.add(ResourceState::ShaderResource);
        bufferDesc.allowedStates.add(ResourceState::UnorderedAccess);
        bufferDesc.allowedStates.add(ResourceState::General);
        bufferDesc.sizeInBytes = mipMapOffset.getLast() * sizeof(uint32_t);
        bufferDesc.type = IResource::Type::Buffer;
        gAccumulateBuffer = gDevice->createBufferResource(bufferDesc);
        gReconstructBuffer = gDevice->createBufferResource(bufferDesc);

        gAccumulateBufferView = createUAV(gAccumulateBuffer);
        gReconstructBufferView = createUAV(gReconstructBuffer);

        int mipCount = 1 + Math::Log2Ceil(Math::Max(textureWidth, textureHeight));
        gLearningTexture = createRenderTargetTexture(
            Format::R32G32B32A32_FLOAT,
            textureWidth,
            textureHeight,
            mipCount);
        gLearningTextureSRV = createSRV(gLearningTexture);
        for (int i = 0; i < mipCount; i++)
            gLearningTextureUAVs.add(createUAV(gLearningTexture, i));

        gDiffTexture = createRenderTargetTexture(
            Format::R32G32B32A32_FLOAT,
            textureWidth,
            textureHeight,
            mipCount);
        gDiffTextureSRV = createSRV(gDiffTexture);
        for (int i = 0; i < mipCount; i++)
            gDiffTextureUAVs.add(createUAV(gDiffTexture, i));

        gfx::ISamplerState::Desc samplerDesc = {};
        // samplerDesc.maxLOD = 0.0f;
        gSampler = gDevice->createSamplerState(samplerDesc);

        gDepthTexture = createDepthTexture();
        gDepthTextureView = createDSV(gDepthTexture);

        gRefImage = createRenderTargetTexture(Format::R8G8B8A8_UNORM, windowWidth, windowHeight, 1);
        gRefImageRTV = createRTV(gRefImage, Format::R8G8B8A8_UNORM);
        gRefImageSRV = createSRV(gRefImage);

        gIterImage =
            createRenderTargetTexture(Format::R8G8B8A8_UNORM, windowWidth, windowHeight, 1);
        gIterImageRTV = createRTV(gIterImage, Format::R8G8B8A8_UNORM);
        gIterImageSRV = createSRV(gIterImage);

        gRefFrameBuffer = createRenderTargetFramebuffer(gRefImageRTV);
        gIterFrameBuffer = createRenderTargetFramebuffer(gIterImageRTV);

        {
            ComPtr<ICommandBuffer> commandBuffer = gTransientHeaps[0]->createCommandBuffer();
            auto encoder = commandBuffer->encodeResourceCommands();
            encoder->textureBarrier(
                gLearningTexture,
                ResourceState::RenderTarget,
                ResourceState::UnorderedAccess);
            encoder->textureBarrier(
                gDiffTexture,
                ResourceState::RenderTarget,
                ResourceState::UnorderedAccess);
            encoder->textureBarrier(
                gRefImage,
                ResourceState::RenderTarget,
                ResourceState::ShaderResource);
            encoder->textureBarrier(
                gIterImage,
                ResourceState::RenderTarget,
                ResourceState::ShaderResource);
            for (int i = 0; i < gLearningTextureUAVs.getCount(); i++)
            {
                ClearValue clearValue = {};
                encoder->clearResourceView(
                    gLearningTextureUAVs[i],
                    &clearValue,
                    ClearResourceViewFlags::None);
                encoder->clearResourceView(
                    gDiffTextureUAVs[i],
                    &clearValue,
                    ClearResourceViewFlags::None);
            }
            encoder->textureBarrier(
                gLearningTexture,
                ResourceState::UnorderedAccess,
                ResourceState::ShaderResource);

            encoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
        }

        return SLANG_OK;
    }

    void initMipOffsets(int w, int h)
    {
        int layers = 1 + Math::Log2Ceil(Math::Max(w, h));
        uint32_t offset = 0;
        for (int i = 0; i < layers; i++)
        {
            auto lw = Math::Max(1, w >> i);
            auto lh = Math::Max(1, h >> i);
            mipMapOffset.add(offset);
            offset += lw * lh * 4;
        }
        mipMapOffset.add(offset);
    }

    glm::mat4x4 getTransformMatrix()
    {
        float rotX = (rand() / (float)RAND_MAX) * 0.3f;
        float rotY = (rand() / (float)RAND_MAX) * 0.2f;
        glm::mat4x4 matProj = glm::perspectiveRH_ZO(
            glm::radians(60.0f),
            (float)windowWidth / (float)windowHeight,
            0.1f,
            1000.0f);
        auto identity = glm::mat4(1.0f);
        auto translate = glm::translate(
            identity,
            glm::vec3(
                -0.6f + 0.2f * (rand() / (float)RAND_MAX),
                -0.6f + 0.2f * (rand() / (float)RAND_MAX),
                -1.0f));
        auto rot = glm::rotate(translate, -glm::pi<float>() * rotX, glm::vec3(1.0f, 0.0f, 0.0f));
        rot = glm::rotate(rot, -glm::pi<float>() * rotY, glm::vec3(0.0f, 1.0f, 0.0f));
        auto transformMatrix = matProj * rot;
        transformMatrix = glm::transpose(transformMatrix);
        return transformMatrix;
    }

    template<typename SetupPipelineFunc>
    void renderImage(
        int transientHeapIndex,
        IFramebuffer* fb,
        const SetupPipelineFunc& setupPipeline)
    {
        ComPtr<ICommandBuffer> commandBuffer =
            gTransientHeaps[transientHeapIndex]->createCommandBuffer();
        auto renderEncoder = commandBuffer->encodeRenderCommands(gRenderPass, fb);

        gfx::Viewport viewport = {};
        viewport.maxZ = 1.0f;
        viewport.extentX = (float)windowWidth;
        viewport.extentY = (float)windowHeight;
        renderEncoder->setViewportAndScissor(viewport);

        setupPipeline(renderEncoder);

        renderEncoder->setVertexBuffer(0, gVertexBuffer);
        renderEncoder->setPrimitiveTopology(PrimitiveTopology::TriangleStrip);

        renderEncoder->draw(4);
        renderEncoder->endEncoding();
        commandBuffer->close();
        gQueue->executeCommandBuffer(commandBuffer);
    }

    void renderReferenceImage(int transientHeapIndex, glm::mat4x4 transformMatrix)
    {
        {
            ComPtr<ICommandBuffer> commandBuffer =
                gTransientHeaps[transientHeapIndex]->createCommandBuffer();
            auto encoder = commandBuffer->encodeResourceCommands();
            encoder->textureBarrier(
                gRefImage,
                ResourceState::ShaderResource,
                ResourceState::RenderTarget);
            encoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
        }

        renderImage(
            transientHeapIndex,
            gRefFrameBuffer,
            [&](IRenderCommandEncoder* encoder)
            {
                auto rootObject = encoder->bindPipeline(gRefPipelineState);
                ShaderCursor rootCursor(rootObject);
                rootCursor["Uniforms"]["modelViewProjection"].setData(
                    &transformMatrix,
                    sizeof(float) * 16);
                rootCursor["Uniforms"]["bwdTexture"]["texture"].setResource(gTexView);
                rootCursor["Uniforms"]["sampler"].setSampler(gSampler);
                rootCursor["Uniforms"]["mipOffset"].setData(
                    mipMapOffset.getBuffer(),
                    sizeof(uint32_t) * mipMapOffset.getCount());
                rootCursor["Uniforms"]["texRef"].setResource(gTexView);
                rootCursor["Uniforms"]["bwdTexture"]["accumulateBuffer"].setResource(
                    gAccumulateBufferView);
            });
    }

    virtual void renderFrame(int frameBufferIndex) override
    {
        static uint32_t frameCount = 0;
        frameCount++;
        auto transformMatrix = getTransformMatrix();
        renderReferenceImage(frameBufferIndex, transformMatrix);

        // Barriers.
        {
            ComPtr<ICommandBuffer> commandBuffer =
                gTransientHeaps[frameBufferIndex]->createCommandBuffer();
            auto resEncoder = commandBuffer->encodeResourceCommands();
            ClearValue clearValue = {};
            resEncoder->bufferBarrier(
                gAccumulateBuffer,
                ResourceState::Undefined,
                ResourceState::UnorderedAccess);
            resEncoder->bufferBarrier(
                gReconstructBuffer,
                ResourceState::Undefined,
                ResourceState::UnorderedAccess);
            resEncoder->textureBarrier(
                gRefImage,
                ResourceState::Present,
                ResourceState::ShaderResource);
            resEncoder->textureBarrier(
                gIterImage,
                ResourceState::ShaderResource,
                ResourceState::RenderTarget);
            resEncoder->clearResourceView(
                gAccumulateBufferView,
                &clearValue,
                ClearResourceViewFlags::None);
            resEncoder->clearResourceView(
                gReconstructBufferView,
                &clearValue,
                ClearResourceViewFlags::None);
            if (resetLearntTexture)
            {
                resEncoder->textureBarrier(
                    gLearningTexture,
                    ResourceState::ShaderResource,
                    ResourceState::UnorderedAccess);
                for (Index i = 0; i < gLearningTextureUAVs.getCount(); i++)
                    resEncoder->clearResourceView(
                        gLearningTextureUAVs[i],
                        &clearValue,
                        ClearResourceViewFlags::None);
                resEncoder->textureBarrier(
                    gLearningTexture,
                    ResourceState::UnorderedAccess,
                    ResourceState::ShaderResource);
                resetLearntTexture = false;
            }
            resEncoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
        }

        // Render image using backward propagate shader to obtain texture-space gradients.
        renderImage(
            frameBufferIndex,
            gIterFrameBuffer,
            [&](IRenderCommandEncoder* encoder)
            {
                auto rootObject = encoder->bindPipeline(gIterPipelineState);
                ShaderCursor rootCursor(rootObject);

                rootCursor["Uniforms"]["modelViewProjection"].setData(
                    &transformMatrix,
                    sizeof(float) * 16);
                rootCursor["Uniforms"]["bwdTexture"]["texture"].setResource(gLearningTextureSRV);
                rootCursor["Uniforms"]["sampler"].setSampler(gSampler);
                rootCursor["Uniforms"]["mipOffset"].setData(
                    mipMapOffset.getBuffer(),
                    sizeof(uint32_t) * mipMapOffset.getCount());
                rootCursor["Uniforms"]["texRef"].setResource(gRefImageSRV);
                rootCursor["Uniforms"]["bwdTexture"]["accumulateBuffer"].setResource(
                    gAccumulateBufferView);
                rootCursor["Uniforms"]["bwdTexture"]["minLOD"].setData(5.0);
            });

        // Propagete gradients through mip map layers from top (lowest res) to bottom (highest res).
        {
            ComPtr<ICommandBuffer> commandBuffer =
                gTransientHeaps[frameBufferIndex]->createCommandBuffer();
            auto encoder = commandBuffer->encodeComputeCommands();
            encoder->textureBarrier(
                gLearningTexture,
                ResourceState::ShaderResource,
                ResourceState::UnorderedAccess);
            auto rootObject = encoder->bindPipeline(gReconstructPipelineState);
            for (int i = (int)mipMapOffset.getCount() - 2; i >= 0; i--)
            {
                ShaderCursor rootCursor(rootObject);
                rootCursor["Uniforms"]["mipOffset"].setData(
                    mipMapOffset.getBuffer(),
                    sizeof(uint32_t) * mipMapOffset.getCount());
                rootCursor["Uniforms"]["dstLayer"].setData(i);
                rootCursor["Uniforms"]["layerCount"].setData(mipMapOffset.getCount() - 1);
                rootCursor["Uniforms"]["width"].setData(textureWidth);
                rootCursor["Uniforms"]["height"].setData(textureHeight);
                rootCursor["Uniforms"]["accumulateBuffer"].setResource(gAccumulateBufferView);
                rootCursor["Uniforms"]["dstBuffer"].setResource(gReconstructBufferView);
                encoder->dispatchCompute(
                    ((textureWidth >> i) + 15) / 16,
                    ((textureHeight >> i) + 15) / 16,
                    1);
                encoder->bufferBarrier(
                    gReconstructBuffer,
                    ResourceState::UnorderedAccess,
                    ResourceState::UnorderedAccess);
            }

            // Convert bottom layer mip from buffer to texture.
            rootObject = encoder->bindPipeline(gConvertPipelineState);
            ShaderCursor rootCursor(rootObject);
            rootCursor["Uniforms"]["mipOffset"].setData(
                mipMapOffset.getBuffer(),
                sizeof(uint32_t) * mipMapOffset.getCount());
            rootCursor["Uniforms"]["dstLayer"].setData(0);
            rootCursor["Uniforms"]["width"].setData(textureWidth);
            rootCursor["Uniforms"]["height"].setData(textureHeight);
            rootCursor["Uniforms"]["srcBuffer"].setResource(gReconstructBufferView);
            rootCursor["Uniforms"]["dstTexture"].setResource(gDiffTextureUAVs[0]);
            encoder->dispatchCompute((textureWidth + 15) / 16, (textureHeight + 15) / 16, 1);
            encoder->textureBarrier(
                gDiffTexture,
                ResourceState::UnorderedAccess,
                ResourceState::UnorderedAccess);

            // Build higher level mip map layers.
            rootObject = encoder->bindPipeline(gBuildMipPipelineState);
            for (int i = 1; i < (int)mipMapOffset.getCount() - 1; i++)
            {
                ShaderCursor rootCursor(rootObject);
                rootCursor["Uniforms"]["dstWidth"].setData(textureWidth >> i);
                rootCursor["Uniforms"]["dstHeight"].setData(textureHeight >> i);
                rootCursor["Uniforms"]["srcTexture"].setResource(gDiffTextureUAVs[i - 1]);
                rootCursor["Uniforms"]["dstTexture"].setResource(gDiffTextureUAVs[i]);
                encoder->dispatchCompute(
                    ((textureWidth >> i) + 15) / 16,
                    ((textureHeight >> i) + 15) / 16,
                    1);
                encoder->textureBarrier(
                    gDiffTexture,
                    ResourceState::UnorderedAccess,
                    ResourceState::UnorderedAccess);
            }

            // Accumulate gradients to learnt texture.
            rootObject = encoder->bindPipeline(gLearnMipPipelineState);
            for (int i = 0; i < (int)mipMapOffset.getCount() - 1; i++)
            {
                ShaderCursor rootCursor(rootObject);
                rootCursor["Uniforms"]["dstWidth"].setData(textureWidth >> i);
                rootCursor["Uniforms"]["dstHeight"].setData(textureHeight >> i);
                rootCursor["Uniforms"]["learningRate"].setData(0.1f);
                rootCursor["Uniforms"]["srcTexture"].setResource(gDiffTextureUAVs[i]);
                rootCursor["Uniforms"]["dstTexture"].setResource(gLearningTextureUAVs[i]);
                encoder->dispatchCompute(
                    ((textureWidth >> i) + 15) / 16,
                    ((textureHeight >> i) + 15) / 16,
                    1);
            }
            encoder->textureBarrier(
                gLearningTexture,
                ResourceState::UnorderedAccess,
                ResourceState::ShaderResource);
            encoder->textureBarrier(
                gIterImage,
                ResourceState::Present,
                ResourceState::ShaderResource);

            encoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
        }

        // Draw currently learnt texture.
        {
            ComPtr<ICommandBuffer> commandBuffer =
                gTransientHeaps[frameBufferIndex]->createCommandBuffer();
            auto renderEncoder =
                commandBuffer->encodeRenderCommands(gRenderPass, gFramebuffers[frameBufferIndex]);
            drawTexturedQuad(renderEncoder, 0, 0, textureWidth, textureHeight, gLearningTextureSRV);
            int refImageWidth = windowWidth - textureWidth - 10;
            int refImageHeight = refImageWidth * windowHeight / windowWidth;
            drawTexturedQuad(
                renderEncoder,
                textureWidth + 10,
                0,
                refImageWidth,
                refImageHeight,
                gRefImageSRV);
            drawTexturedQuad(
                renderEncoder,
                textureWidth + 10,
                refImageHeight + 10,
                refImageWidth,
                refImageHeight,
                gIterImageSRV);
            renderEncoder->endEncoding();
            commandBuffer->close();
            gQueue->executeCommandBuffer(commandBuffer);
        }

        if (!isTestMode())
        {
            gSwapchain->present();
        }
    }

    void drawTexturedQuad(
        IRenderCommandEncoder* renderEncoder,
        int x,
        int y,
        int w,
        int h,
        IResourceView* srv)
    {
        gfx::Viewport viewport = {};
        viewport.maxZ = 1.0f;
        viewport.extentX = (float)windowWidth;
        viewport.extentY = (float)windowHeight;
        renderEncoder->setViewportAndScissor(viewport);

        auto root = renderEncoder->bindPipeline(gDrawQuadPipelineState);
        ShaderCursor rootCursor(root);
        rootCursor["Uniforms"]["x"].setData(x);
        rootCursor["Uniforms"]["y"].setData(y);
        rootCursor["Uniforms"]["width"].setData(w);
        rootCursor["Uniforms"]["height"].setData(h);
        rootCursor["Uniforms"]["viewWidth"].setData(windowWidth);
        rootCursor["Uniforms"]["viewHeight"].setData(windowHeight);
        rootCursor["Uniforms"]["texture"].setResource(srv);
        rootCursor["Uniforms"]["sampler"].setSampler(gSampler);
        renderEncoder->setVertexBuffer(0, gVertexBuffer);
        renderEncoder->setPrimitiveTopology(PrimitiveTopology::TriangleStrip);
        renderEncoder->draw(4);
    }
};

EXAMPLE_MAIN(innerMain<AutoDiffTexture>);
