// main.cpp
#include "slang-com-ptr.h"
#include "slang.h"

#include <string>
using Slang::ComPtr;

#include "core/slang-basic.h"
#include "examples/example-base/example-base.h"
#include "gfx-util/shader-cursor.h"
#include "gpu-printing.h"
#include "platform/window.h"
#include "slang-gfx.h"

using namespace gfx;

static const ExampleResources resourceBase("gpu-printing");

ComPtr<slang::ISession> createSlangSession(gfx::IDevice* device)
{
    ComPtr<slang::ISession> slangSession = device->getSlangSession();
    return slangSession;
}

ComPtr<slang::IModule> compileShaderModuleFromFile(
    slang::ISession* slangSession,
    char const* filePath)
{
    ComPtr<slang::IModule> slangModule;
    ComPtr<slang::IBlob> diagnosticBlob;
    Slang::String path = resourceBase.resolveResource(filePath);
    slangModule = slangSession->loadModule(path.getBuffer(), diagnosticBlob.writeRef());
    diagnoseIfNeeded(diagnosticBlob);

    return slangModule;
}

struct ExampleProgram : public TestBase
{
    int gWindowWidth = 640;
    int gWindowHeight = 480;

    ComPtr<gfx::IDevice> gDevice;

    ComPtr<slang::ISession> gSlangSession;
    ComPtr<slang::IModule> gSlangModule;
    ComPtr<gfx::IShaderProgram> gProgram;

    ComPtr<gfx::IPipelineState> gPipelineState;

    Slang::Dictionary<int, std::string> gHashedStrings;

    GPUPrinting gGPUPrinting;

    ComPtr<gfx::IShaderProgram> loadComputeProgram(
        slang::IModule* slangModule,
        char const* entryPointName)
    {
        ComPtr<slang::IEntryPoint> entryPoint;
        slangModule->findEntryPointByName(entryPointName, entryPoint.writeRef());

        ComPtr<slang::IComponentType> linkedProgram;
        entryPoint->link(linkedProgram.writeRef());

        if (isTestMode())
        {
            printEntrypointHashes(1, 1, linkedProgram);
        }

        gGPUPrinting.loadStrings(linkedProgram->getLayout());

        gfx::IShaderProgram::Desc programDesc = {};
        programDesc.slangGlobalScope = linkedProgram;

        auto shaderProgram = gDevice->createProgram(programDesc);

        return shaderProgram;
    }

    Result execute(int argc, char* argv[])
    {
        parseOption(argc, argv);
        IDevice::Desc deviceDesc;
        Result res = gfxCreateDevice(&deviceDesc, gDevice.writeRef());
        if (SLANG_FAILED(res))
            return res;

        Slang::String path = resourceBase.resolveResource("kernels.slang");

        gSlangSession = createSlangSession(gDevice);
        gSlangModule = compileShaderModuleFromFile(gSlangSession, path.getBuffer());
        if (!gSlangModule)
            return SLANG_FAIL;

        gProgram = loadComputeProgram(gSlangModule, "computeMain");
        if (!gProgram)
            return SLANG_FAIL;

        ComputePipelineStateDesc desc;
        desc.program = gProgram;
        auto pipelineState = gDevice->createComputePipelineState(desc);
        if (!pipelineState)
            return SLANG_FAIL;

        gPipelineState = pipelineState;

        size_t printBufferSize = 4 * 1024; // use a small-ish (4KB) buffer for print output

        IBufferResource::Desc printBufferDesc = {};
        printBufferDesc.type = IResource::Type::Buffer;
        printBufferDesc.sizeInBytes = printBufferSize;
        printBufferDesc.elementSize = sizeof(uint32_t);
        printBufferDesc.defaultState = ResourceState::UnorderedAccess;
        printBufferDesc.allowedStates = ResourceStateSet(
            ResourceState::CopySource,
            ResourceState::CopyDestination,
            ResourceState::UnorderedAccess);
        printBufferDesc.memoryType = MemoryType::DeviceLocal;
        auto printBuffer = gDevice->createBufferResource(printBufferDesc);

        IResourceView::Desc printBufferViewDesc = {};
        printBufferViewDesc.type = IResourceView::Type::UnorderedAccess;
        printBufferViewDesc.format = Format::Unknown;
        auto printBufferView = gDevice->createBufferView(printBuffer, nullptr, printBufferViewDesc);

        ITransientResourceHeap::Desc transientResourceHeapDesc = {};
        transientResourceHeapDesc.constantBufferSize = 256;
        auto transientHeap = gDevice->createTransientResourceHeap(transientResourceHeapDesc);

        ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
        auto queue = gDevice->createCommandQueue(queueDesc);
        auto commandBuffer = transientHeap->createCommandBuffer();
        auto encoder = commandBuffer->encodeComputeCommands();
        auto rootShaderObject = encoder->bindPipeline(gPipelineState);
        auto cursor = ShaderCursor(rootShaderObject);
        cursor["gPrintBuffer"].setResource(printBufferView);
        encoder->dispatchCompute(1, 1, 1);
        encoder->bufferBarrier(
            printBuffer,
            ResourceState::UnorderedAccess,
            ResourceState::CopySource);
        encoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);

        ComPtr<ISlangBlob> blob;
        gDevice->readBufferResource(printBuffer, 0, printBufferSize, blob.writeRef());

        gGPUPrinting.processGPUPrintCommands(blob->getBufferPointer(), printBufferSize);

        return SLANG_OK;
    }
};

int exampleMain(int argc, char** argv)
{
    ExampleProgram app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}
