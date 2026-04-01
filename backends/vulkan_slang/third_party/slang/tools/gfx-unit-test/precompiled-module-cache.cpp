#include "core/slang-basic.h"
#include "core/slang-blob.h"
#include "core/slang-io.h"
#include "core/slang-memory-file-system.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

#include <mutex>
using namespace gfx;

namespace gfx_test
{
// Test that precompiled module cache is working.

Slang::ComPtr<slang::ISession> createSession(gfx::IDevice* device, ISlangFileSystemExt* fileSys)
{
    static std::mutex m;
    std::lock_guard<std ::mutex> lock(m);

    Slang::ComPtr<slang::ISession> slangSession;
    device->getSlangSession(slangSession.writeRef());
    slang::SessionDesc sessionDesc = {};
    sessionDesc.searchPathCount = 1;
    const char* searchPath = "cache/";
    sessionDesc.searchPaths = &searchPath;
    sessionDesc.targetCount = 1;
    sessionDesc.compilerOptionEntryCount = 1;
    slang::CompilerOptionEntry entry;
    entry.name = slang::CompilerOptionName::UseUpToDateBinaryModule;
    entry.value.kind = slang::CompilerOptionValueKind::Int;
    entry.value.intValue0 = 1;
    sessionDesc.compilerOptionEntries = &entry;
    slang::TargetDesc targetDesc = {};
    switch (device->getDeviceInfo().deviceType)
    {
    case gfx::DeviceType::DirectX12:
        targetDesc.format = SLANG_DXIL;
        targetDesc.profile = device->getSlangSession()->getGlobalSession()->findProfile("sm_6_1");
        break;
    case gfx::DeviceType::Vulkan:
        targetDesc.format = SLANG_SPIRV;
        targetDesc.profile = device->getSlangSession()->getGlobalSession()->findProfile("GLSL_460");
        break;
    }
    sessionDesc.targets = &targetDesc;
    sessionDesc.fileSystem = fileSys;
    auto globalSession = slangSession->getGlobalSession();
    globalSession->createSession(sessionDesc, slangSession.writeRef());
    return slangSession;
}

static Slang::Result precompileProgram(
    gfx::IDevice* device,
    ISlangMutableFileSystem* fileSys,
    const char* shaderModuleName)
{
    Slang::ComPtr<slang::ISession> slangSession = createSession(device, fileSys);

    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slang::IModule* module = slangSession->loadModule(shaderModuleName, diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!module)
        return SLANG_FAIL;

    // Write loaded modules to memory file system.
    for (SlangInt i = 0; i < slangSession->getLoadedModuleCount(); i++)
    {
        auto module = slangSession->getLoadedModule(i);
        auto path = module->getFilePath();
        if (path)
        {
            auto name = module->getName();
            ComPtr<ISlangBlob> outBlob;
            module->serialize(outBlob.writeRef());
            fileSys->saveFileBlob(
                (Slang::String("cache/") + Slang::String(name) + ".slang-module").getBuffer(),
                outBlob);
        }
    }
    return SLANG_OK;
}

void precompiledModuleCacheTestImpl(IDevice* device, UnitTestContext* context)
{
    Slang::ComPtr<ITransientResourceHeap> transientHeap;
    ITransientResourceHeap::Desc transientHeapDesc = {};
    transientHeapDesc.constantBufferSize = 4096;
    GFX_CHECK_CALL_ABORT(
        device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

    // First, Initialize our file system.
    ComPtr<ISlangMutableFileSystem> memoryFileSystem =
        ComPtr<ISlangMutableFileSystem>(new Slang::MemoryFileSystem());
    memoryFileSystem->createDirectory("cache");

    const char* moduleSrc = R"(
            import "precompiled-module-imported";

            // Main entry-point. 

            using namespace ns;

            [shader("compute")]
            [numthreads(4, 1, 1)]
            void computeMain(
                uint3 sv_dispatchThreadID : SV_DispatchThreadID,
                uniform RWStructuredBuffer <float> buffer)
            {
                buffer[sv_dispatchThreadID.x] = helperFunc() + helperFunc1();
            }
        )";
    memoryFileSystem->saveFile("precompiled-module.slang", moduleSrc, strlen(moduleSrc));

    const char* moduleSrc2 = R"(
            module "precompiled-module-imported";

            __include "precompiled-module-included.slang";

            namespace ns
            {
                public int helperFunc()
                {
                    return 1;
                }
            }
        )";
    memoryFileSystem->saveFile("precompiled-module-imported.slang", moduleSrc2, strlen(moduleSrc2));
    const char* moduleSrc3 = R"(
            implementing "precompiled-module-imported";

            namespace ns
            {
                public int helperFunc1()
                {
                    return 2;
                }
            }
        )";
    memoryFileSystem->saveFile("precompiled-module-included.slang", moduleSrc3, strlen(moduleSrc3));

    // Precompile a module.
    ComPtr<IShaderProgram> shaderProgram;
    slang::ProgramLayout* slangReflection;
    GFX_CHECK_CALL_ABORT(
        precompileProgram(device, memoryFileSystem.get(), "precompiled-module-imported"));

    // Next, load the precompiled slang program.
    Slang::ComPtr<slang::ISession> slangSession = createSession(device, memoryFileSystem);
    ComPtr<ISlangBlob> binaryBlob;
    memoryFileSystem->loadFile(
        "cache/precompiled-module-imported.slang-module",
        binaryBlob.writeRef());
    auto upToDate =
        slangSession->isBinaryModuleUpToDate("precompiled-module-imported.slang", binaryBlob);
    SLANG_CHECK(upToDate); // The module should be up-to-date.

    GFX_CHECK_CALL_ABORT(loadComputeProgram(
        device,
        slangSession,
        shaderProgram,
        "precompiled-module",
        "computeMain",
        slangReflection));

    ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<gfx::IPipelineState> pipelineState;
    GFX_CHECK_CALL_ABORT(
        device->createComputePipelineState(pipelineDesc, pipelineState.writeRef()));

    const int numberCount = 4;
    float initialData[] = {0.0f, 0.0f, 0.0f, 0.0f};
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

        ShaderCursor entryPointCursor(
            rootObject->getEntryPoint(0)); // get a cursor the the first entry-point.
        // Bind buffer view to the entry point.
        entryPointCursor.getPath("buffer").setResource(bufferView);

        encoder->dispatchCompute(1, 1, 1);
        encoder->endEncoding();
        commandBuffer->close();
        queue->executeCommandBuffer(commandBuffer);
        queue->waitOnHost();
    }

    compareComputeResult(device, numbersBuffer, Slang::makeArray<float>(3.0f, 3.0f, 3.0f, 3.0f));

    // Now we change the source and check if the precompiled module is still up-to-date.
    const char* moduleSrc4 = R"(
            implementing "precompiled-module-imported";
            namespace ns {
                public int helperFunc1() {
                    return 2;
                }
            }
        )";
    memoryFileSystem->saveFile("precompiled-module-included.slang", moduleSrc4, strlen(moduleSrc4));

    slangSession = createSession(device, memoryFileSystem);
    upToDate =
        slangSession->isBinaryModuleUpToDate("precompiled-module-imported.slang", binaryBlob);
    SLANG_CHECK(!upToDate); // The module should not be up-to-date because the source has changed.
}

SLANG_UNIT_TEST(precompiledModuleCacheD3D12)
{
    runTestImpl(precompiledModuleCacheTestImpl, unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(precompiledModuleCacheVulkan)
{
    runTestImpl(precompiledModuleCacheTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
