#include "core/slang-basic.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace gfx;

namespace gfx_test
{
void sharedBufferTestImpl(IDevice* srcDevice, IDevice* dstDevice, UnitTestContext* context)
{
    // Create a shareable buffer using srcDevice, get its handle, then create a buffer using the
    // handle using dstDevice. Read back the buffer and check that its contents are correct.
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
    bufferDesc.isShared = true;

    ComPtr<IBufferResource> srcBuffer;
    GFX_CHECK_CALL_ABORT(
        srcDevice->createBufferResource(bufferDesc, (void*)initialData, srcBuffer.writeRef()));

    InteropHandle sharedHandle;
    GFX_CHECK_CALL_ABORT(srcBuffer->getSharedHandle(&sharedHandle));
    ComPtr<IBufferResource> dstBuffer;
    GFX_CHECK_CALL_ABORT(
        dstDevice->createBufferFromSharedHandle(sharedHandle, bufferDesc, dstBuffer.writeRef()));
    // Reading back the buffer from srcDevice to make sure it's been filled in before reading
    // anything back from dstDevice
    // TODO: Implement actual synchronization (and not this hacky solution)
    compareComputeResult(srcDevice, srcBuffer, Slang::makeArray<float>(0.0f, 1.0f, 2.0f, 3.0f));

    InteropHandle testHandle;
    GFX_CHECK_CALL_ABORT(dstBuffer->getNativeResourceHandle(&testHandle));
    IBufferResource::Desc* testDesc = dstBuffer->getDesc();
    SLANG_CHECK(testDesc->elementSize == sizeof(float));
    SLANG_CHECK(testDesc->sizeInBytes == numberCount * sizeof(float));
    compareComputeResult(dstDevice, dstBuffer, Slang::makeArray<float>(0.0f, 1.0f, 2.0f, 3.0f));

    // Check that dstBuffer can be successfully used in a compute dispatch using dstDevice.
    Slang::ComPtr<ITransientResourceHeap> transientHeap;
    ITransientResourceHeap::Desc transientHeapDesc = {};
    transientHeapDesc.constantBufferSize = 4096;
    GFX_CHECK_CALL_ABORT(
        dstDevice->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

    ComPtr<IShaderProgram> shaderProgram;
    slang::ProgramLayout* slangReflection;
    GFX_CHECK_CALL_ABORT(loadComputeProgram(
        dstDevice,
        shaderProgram,
        "compute-trivial",
        "computeMain",
        slangReflection));

    ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<gfx::IPipelineState> pipelineState;
    GFX_CHECK_CALL_ABORT(
        dstDevice->createComputePipelineState(pipelineDesc, pipelineState.writeRef()));

    ComPtr<IResourceView> bufferView;
    IResourceView::Desc viewDesc = {};
    viewDesc.type = IResourceView::Type::UnorderedAccess;
    viewDesc.format = Format::Unknown;
    GFX_CHECK_CALL_ABORT(
        dstDevice->createBufferView(dstBuffer, nullptr, viewDesc, bufferView.writeRef()));

    {
        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = dstDevice->createCommandQueue(queueDesc);

        auto commandBuffer = transientHeap->createCommandBuffer();
        auto encoder = commandBuffer->encodeComputeCommands();

        auto rootObject = encoder->bindPipeline(pipelineState);

        ShaderCursor rootCursor(rootObject);
        // Bind buffer view to the entry point.
        rootCursor.getPath("buffer").setResource(bufferView);

        encoder->dispatchCompute(1, 1, 1);
        encoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    compareComputeResult(dstDevice, dstBuffer, Slang::makeArray<float>(1.0f, 2.0f, 3.0f, 4.0f));
}

void sharedBufferTestAPI(
    UnitTestContext* context,
    Slang::RenderApiFlag::Enum srcApi,
    Slang::RenderApiFlag::Enum dstApi)
{
    auto srcDevice = createTestingDevice(context, srcApi);
    auto dstDevice = createTestingDevice(context, dstApi);
    if (!srcDevice || !dstDevice)
    {
        SLANG_IGNORE_TEST;
    }

    sharedBufferTestImpl(srcDevice, dstDevice, context);
}
#if SLANG_WIN64
SLANG_UNIT_TEST(sharedBufferD3D12ToCUDA)
{
    sharedBufferTestAPI(unitTestContext, Slang::RenderApiFlag::D3D12, Slang::RenderApiFlag::CUDA);
}

SLANG_UNIT_TEST(sharedBufferVulkanToCUDA)
{
    sharedBufferTestAPI(unitTestContext, Slang::RenderApiFlag::Vulkan, Slang::RenderApiFlag::CUDA);
}
#endif
} // namespace gfx_test
