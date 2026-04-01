#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace gfx;

namespace gfx_test
{
void existingDeviceHandleTestImpl(IDevice* device, UnitTestContext* context)
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
        "compute-trivial",
        "computeMain",
        slangReflection));

    ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<gfx::IPipelineState> pipelineState;
    GFX_CHECK_CALL_ABORT(
        device->createComputePipelineState(pipelineDesc, pipelineState.writeRef()));

    const int numberCount = 4;
    float initialData[] = {0.0f, 1.0f, 2.0f, 3.0f};
    IBufferResource::Desc bufferDesc = {};
    bufferDesc.sizeInBytes = numberCount * sizeof(float);
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
        device->createBufferResource(bufferDesc, (void*)initialData, numbersBuffer.writeRef()));

    ComPtr<IResourceView> bufferView;
    IResourceView::Desc viewDesc = {};
    viewDesc.type = IResourceView::Type::UnorderedAccess;
    viewDesc.format = Format::Unknown;
    GFX_CHECK_CALL_ABORT(
        device->createBufferView(numbersBuffer, nullptr, viewDesc, bufferView.writeRef()));

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = device->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        auto encoder = commandBuffer->encodeComputeCommands();

        auto rootObject = encoder->bindPipeline(pipelineState);

        ShaderCursor rootCursor(rootObject);
        // Bind buffer view to the root.
        rootCursor.getPath("buffer").setResource(bufferView);

        encoder->dispatchCompute(1, 1, 1);
        encoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    compareComputeResult(device, numbersBuffer, Slang::makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}

void existingDeviceHandleTestAPI(UnitTestContext* context, Slang::RenderApiFlag::Enum api)
{
    if ((api & context->enabledApis) == 0)
    {
        SLANG_IGNORE_TEST;
    }
    Slang::ComPtr<IDevice> device;
    IDevice::Desc deviceDesc = {};
    switch (api)
    {
    case Slang::RenderApiFlag::D3D12:
        deviceDesc.deviceType = gfx::DeviceType::DirectX12;
        break;
    case Slang::RenderApiFlag::Vulkan:
        deviceDesc.deviceType = gfx::DeviceType::Vulkan;
        break;
    case Slang::RenderApiFlag::CUDA:
        deviceDesc.deviceType = gfx::DeviceType::CUDA;
        break;
    default:
        SLANG_IGNORE_TEST;
    }
    deviceDesc.slang.slangGlobalSession = context->slangGlobalSession;
    const char* searchPaths[] = {"", "../../tools/gfx-unit-test", "tools/gfx-unit-test"};
    deviceDesc.slang.searchPathCount = (SlangInt)SLANG_COUNT_OF(searchPaths);
    deviceDesc.slang.searchPaths = searchPaths;
    auto createDeviceResult = gfxCreateDevice(&deviceDesc, device.writeRef());
    if (SLANG_FAILED(createDeviceResult) || !device)
    {
        SLANG_IGNORE_TEST;
    }

    IDevice::InteropHandles handles;
    GFX_CHECK_CALL_ABORT(device->getNativeDeviceHandles(&handles));
    Slang::ComPtr<IDevice> testDevice;
    IDevice::Desc testDeviceDesc = deviceDesc;
    testDeviceDesc.existingDeviceHandles.handles[0] = handles.handles[0];
    if (api == Slang::RenderApiFlag::Vulkan)
    {
        testDeviceDesc.existingDeviceHandles.handles[1] = handles.handles[1];
        testDeviceDesc.existingDeviceHandles.handles[2] = handles.handles[2];
    }
    auto createTestDeviceResult = gfxCreateDevice(&testDeviceDesc, testDevice.writeRef());
    if (SLANG_FAILED(createTestDeviceResult) || !device)
    {
        SLANG_IGNORE_TEST;
    }

    existingDeviceHandleTestImpl(device, context);
}

SLANG_UNIT_TEST(existingDeviceHandleD3D12)
{
    return existingDeviceHandleTestAPI(unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(existingDeviceHandleVulkan)
{
    return existingDeviceHandleTestAPI(unitTestContext, Slang::RenderApiFlag::Vulkan);
}
#if SLANG_WIN64
SLANG_UNIT_TEST(existingDeviceHandleCUDA)
{
    return existingDeviceHandleTestAPI(unitTestContext, Slang::RenderApiFlag::CUDA);
}
#endif
} // namespace gfx_test
