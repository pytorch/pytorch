#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace gfx;

namespace gfx_test
{
static ComPtr<IBufferResource> createBuffer(IDevice* device, uint32_t content)
{
    ComPtr<IBufferResource> buffer;
    IBufferResource::Desc bufferDesc = {};
    bufferDesc.sizeInBytes = sizeof(uint32_t);
    bufferDesc.format = gfx::Format::Unknown;
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.allowedStates = ResourceStateSet(
        ResourceState::ShaderResource,
        ResourceState::UnorderedAccess,
        ResourceState::CopyDestination,
        ResourceState::CopySource);
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBufferResource> numbersBuffer;
    GFX_CHECK_CALL_ABORT(
        device->createBufferResource(bufferDesc, (void*)&content, buffer.writeRef()));

    return buffer;
}
void samplerArrayTestImpl(IDevice* device, UnitTestContext* context)
{
    Slang::ComPtr<ITransientResourceHeap> transientHeap;
    ITransientResourceHeap::Desc transientHeapDesc = {};
    transientHeapDesc.constantBufferSize = 4096;
    GFX_CHECK_CALL_ABORT(
        device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

    ComPtr<IShaderProgram> shaderProgram;
    slang::ProgramLayout* slangReflection;
    GFX_CHECK_CALL_ABORT(
        loadComputeProgram(device, shaderProgram, "sampler-array", "computeMain", slangReflection));

    ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<gfx::IPipelineState> pipelineState;
    GFX_CHECK_CALL_ABORT(
        device->createComputePipelineState(pipelineDesc, pipelineState.writeRef()));

    Slang::List<ComPtr<ISamplerState>> samplers;
    Slang::List<ComPtr<IResourceView>> srvs;
    ComPtr<IResourceView> uav;
    ComPtr<ITextureResource> texture;
    ComPtr<IBufferResource> buffer = createBuffer(device, 0);

    {
        IResourceView::Desc viewDesc = {};
        viewDesc.type = IResourceView::Type::UnorderedAccess;
        viewDesc.format = Format::Unknown;
        GFX_CHECK_CALL_ABORT(device->createBufferView(buffer, nullptr, viewDesc, uav.writeRef()));
    }
    {
        ITextureResource::Desc textureDesc = {};
        textureDesc.type = IResource::Type::Texture2D;
        textureDesc.format = Format::R8G8B8A8_UNORM;
        textureDesc.size.width = 2;
        textureDesc.size.height = 2;
        textureDesc.size.depth = 1;
        textureDesc.numMipLevels = 2;
        textureDesc.memoryType = MemoryType::DeviceLocal;
        textureDesc.defaultState = ResourceState::ShaderResource;
        textureDesc.allowedStates.add(ResourceState::CopyDestination);
        uint32_t data[] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        ITextureResource::SubresourceData subResourceData[2] = {{data, 8, 16}, {data, 8, 16}};
        GFX_CHECK_CALL_ABORT(
            device->createTextureResource(textureDesc, subResourceData, texture.writeRef()));
    }
    for (uint32_t i = 0; i < 32; i++)
    {
        ComPtr<IResourceView> srv;
        IResourceView::Desc viewDesc = {};
        viewDesc.type = IResourceView::Type::ShaderResource;
        viewDesc.format = Format::R8G8B8A8_UNORM;
        viewDesc.subresourceRange.layerCount = 1;
        viewDesc.subresourceRange.mipLevelCount = 1;
        GFX_CHECK_CALL_ABORT(device->createTextureView(texture, viewDesc, srv.writeRef()));
        srvs.add(srv);
    }

    for (uint32_t i = 0; i < 32; i++)
    {
        ISamplerState::Desc desc = {};
        ComPtr<ISamplerState> sampler;
        GFX_CHECK_CALL_ABORT(device->createSamplerState(desc, sampler.writeRef()));
        samplers.add(sampler);
    }

    ComPtr<IShaderObject> rootObject;
    device->createMutableRootShaderObject(shaderProgram, rootObject.writeRef());

    ComPtr<IShaderObject> g;
    device->createMutableShaderObject(
        slangReflection->findTypeByName("S0"),
        ShaderObjectContainerType::None,
        g.writeRef());

    ComPtr<IShaderObject> s1;
    device->createMutableShaderObject(
        slangReflection->findTypeByName("S1"),
        ShaderObjectContainerType::None,
        s1.writeRef());

    {
        auto cursor = ShaderCursor(s1);
        for (uint32_t i = 0; i < 32; i++)
        {
            cursor["samplers"][i].setSampler(samplers[i]);
            cursor["tex"][i].setResource(srvs[i]);
        }
        cursor["data"].setData(1.0f);
    }

    {
        auto cursor = ShaderCursor(g);
        cursor["s"].setObject(s1);
        cursor["data"].setData(2.0f);
    }

    {
        auto cursor = ShaderCursor(rootObject);
        cursor["g"].setObject(g);
        cursor["buffer"].setResource(uav);
    }

    {
        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = device->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        {
            auto encoder = commandBuffer->encodeComputeCommands();
            encoder->bindPipelineWithRootObject(pipelineState, rootObject);
            encoder->dispatchCompute(1, 1, 1);
            encoder->endEncoding();
        }

        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    compareComputeResult(device, buffer, Slang::makeArray<float>(4.0f));
}

SLANG_UNIT_TEST(samplerArrayVulkan)
{
    runTestImpl(samplerArrayTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}
} // namespace gfx_test
