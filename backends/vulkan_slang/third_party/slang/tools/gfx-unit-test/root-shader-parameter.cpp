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
void rootShaderParameterTestImpl(IDevice* device, UnitTestContext* context)
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
        "root-shader-parameter",
        "computeMain",
        slangReflection));

    ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<gfx::IPipelineState> pipelineState;
    GFX_CHECK_CALL_ABORT(
        device->createComputePipelineState(pipelineDesc, pipelineState.writeRef()));

    Slang::List<ComPtr<IBufferResource>> buffers;
    Slang::List<ComPtr<IResourceView>> srvs, uavs;

    for (uint32_t i = 0; i < 9; i++)
    {
        buffers.add(createBuffer(device, i == 0 ? 10 : i));

        ComPtr<IResourceView> bufferView;
        IResourceView::Desc viewDesc = {};
        viewDesc.type = IResourceView::Type::UnorderedAccess;
        viewDesc.format = Format::Unknown;
        GFX_CHECK_CALL_ABORT(
            device->createBufferView(buffers[i], nullptr, viewDesc, bufferView.writeRef()));
        uavs.add(bufferView);

        viewDesc.type = IResourceView::Type::ShaderResource;
        viewDesc.format = Format::Unknown;
        GFX_CHECK_CALL_ABORT(
            device->createBufferView(buffers[i], nullptr, viewDesc, bufferView.writeRef()));
        srvs.add(bufferView);
    }

    ComPtr<IShaderObject> rootObject;
    device->createMutableRootShaderObject(shaderProgram, rootObject.writeRef());

    ComPtr<IShaderObject> g, s1, s2;
    device->createMutableShaderObject(
        slangReflection->findTypeByName("S0"),
        ShaderObjectContainerType::None,
        g.writeRef());
    device->createMutableShaderObject(
        slangReflection->findTypeByName("S1"),
        ShaderObjectContainerType::None,
        s1.writeRef());
    device->createMutableShaderObject(
        slangReflection->findTypeByName("S1"),
        ShaderObjectContainerType::None,
        s2.writeRef());

    {
        auto cursor = ShaderCursor(s1);
        cursor["c0"].setResource(srvs[2]);
        cursor["c1"].setResource(uavs[3]);
        cursor["c2"].setResource(srvs[4]);
    }
    {
        auto cursor = ShaderCursor(s2);
        cursor["c0"].setResource(srvs[5]);
        cursor["c1"].setResource(uavs[6]);
        cursor["c2"].setResource(srvs[7]);
    }
    {
        auto cursor = ShaderCursor(g);
        cursor["b0"].setResource(srvs[0]);
        cursor["b1"].setResource(srvs[1]);
        cursor["s1"].setObject(s1);
        cursor["s2"].setObject(s2);
    }
    {
        auto cursor = ShaderCursor(rootObject);
        cursor["g"].setObject(g);
        cursor["buffer"].setResource(uavs[8]);
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

    compareComputeResult(
        device,
        buffers[8],
        Slang::makeArray<uint32_t>(10 - 1 + 2 - 3 + 4 + 5 - 6 + 7));
}

SLANG_UNIT_TEST(rootShaderParameterD3D12)
{
    runTestImpl(rootShaderParameterTestImpl, unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(rootShaderParameterVulkan)
{
    runTestImpl(rootShaderParameterTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}
} // namespace gfx_test
