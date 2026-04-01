#include "core/slang-basic.h"
#include "core/slang-blob.h"
#include "gfx-test-util.h"
#include "gfx-util/shader-cursor.h"
#include "slang-gfx.h"
#include "unit-test/slang-unit-test.h"

using namespace gfx;

namespace gfx_test
{
static Slang::Result loadProgram(
    gfx::IDevice* device,
    Slang::ComPtr<gfx::IShaderProgram>& outShaderProgram,
    slang::ProgramLayout*& slangReflection,
    bool linkSpecialization = false)
{
    const char* moduleInterfaceSrc = R"(
            interface IFoo
            {
                static const int offset;
                [mutating] void setValue(float v);
                float getValue();
                property float val2{get;set;}
            }
            struct FooImpl : IFoo
            {
                float val;
                static const int offset = -1;
                [mutating] void setValue(float v) { val = v; }
                float getValue() { return val + 1.0; }
                property float val2 {
                    get { return val + 2.0; }
                    set { val = newValue; }
                }
            };
            struct BarImpl : IFoo
            {
                float val;
                static const int offset = 2;
                [mutating] void setValue(float v) { val = v; }
                float getValue() { return val + 1.0; }
                property float val2 {
                    get { return val; }
                    set { val = newValue; }
                }
            };
        )";
    const char* module0Src = R"(
            import ifoo;
            extern struct Foo : IFoo = FooImpl;
            extern static const float c = 0.0;
            [numthreads(1,1,1)]
            void computeMain(uniform RWStructuredBuffer<float> buffer)
            {
                Foo foo;
                foo.setValue(3.0);
                buffer[0] = foo.getValue() + foo.val2 + Foo.offset + c;
            }
        )";
    const char* module1Src = R"(
            import ifoo;
            export struct Foo : IFoo = BarImpl;
            export static const float c = 1.0;
        )";
    Slang::ComPtr<slang::ISession> slangSession;
    SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    auto moduleInterfaceBlob =
        Slang::UnownedRawBlob::create(moduleInterfaceSrc, strlen(moduleInterfaceSrc));
    auto module0Blob = Slang::UnownedRawBlob::create(module0Src, strlen(module0Src));
    auto module1Blob = Slang::UnownedRawBlob::create(module1Src, strlen(module1Src));
    slang::IModule* moduleInterface =
        slangSession->loadModuleFromSource("ifoo", "ifoo.slang", moduleInterfaceBlob);
    slang::IModule* module0 = slangSession->loadModuleFromSource("module0", "path0", module0Blob);
    slang::IModule* module1 = slangSession->loadModuleFromSource("module1", "path1", module1Blob);
    ComPtr<slang::IEntryPoint> computeEntryPoint;
    SLANG_RETURN_ON_FAIL(
        module0->findEntryPointByName("computeMain", computeEntryPoint.writeRef()));

    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(moduleInterface);
    componentTypes.add(module0);
    if (linkSpecialization)
        componentTypes.add(module1);
    componentTypes.add(computeEntryPoint);

    Slang::ComPtr<slang::IComponentType> composedProgram;
    SlangResult result = slangSession->createCompositeComponentType(
        componentTypes.getBuffer(),
        componentTypes.getCount(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    SLANG_RETURN_ON_FAIL(result);

    ComPtr<slang::IComponentType> linkedProgram;
    result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    SLANG_RETURN_ON_FAIL(result);

    composedProgram = linkedProgram;
    slangReflection = composedProgram->getLayout();

    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.slangGlobalScope = composedProgram.get();

    auto shaderProgram = device->createProgram(programDesc);

    outShaderProgram = shaderProgram;
    return SLANG_OK;
}

void linkTimeDefaultTestImpl(IDevice* device, UnitTestContext* context)
{
    Slang::ComPtr<ITransientResourceHeap> transientHeap;
    ITransientResourceHeap::Desc transientHeapDesc = {};
    transientHeapDesc.constantBufferSize = 4096;
    GFX_CHECK_CALL_ABORT(
        device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

    // Create pipeline without linking a specialization override module, so we should
    // see the default value of `extern Foo`.
    ComPtr<IShaderProgram> shaderProgram;
    slang::ProgramLayout* slangReflection;
    GFX_CHECK_CALL_ABORT(loadProgram(device, shaderProgram, slangReflection, false));

    ComputePipelineStateDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<gfx::IPipelineState> pipelineState;
    GFX_CHECK_CALL_ABORT(
        device->createComputePipelineState(pipelineDesc, pipelineState.writeRef()));

    // Create pipeline with a specialization override module linked in, so we should
    // see the result of using `Bar` for `extern Foo`.
    ComPtr<IShaderProgram> shaderProgram1;
    GFX_CHECK_CALL_ABORT(loadProgram(device, shaderProgram1, slangReflection, true));

    ComputePipelineStateDesc pipelineDesc1 = {};
    pipelineDesc1.program = shaderProgram1.get();
    ComPtr<gfx::IPipelineState> pipelineState1;
    GFX_CHECK_CALL_ABORT(
        device->createComputePipelineState(pipelineDesc1, pipelineState1.writeRef()));

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

    ICommandQueue::Desc queueDesc = {ICommandQueue::QueueType::Graphics};
    auto queue = device->createCommandQueue(queueDesc);

    // We have done all the set up work, now it is time to start recording a command buffer for
    // GPU execution.
    {
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

    compareComputeResult(device, numbersBuffer, Slang::makeArray<float>(8.0));

    // Now run again with the overrided program.
    {
        auto commandBuffer = transientHeap->createCommandBuffer();
        auto encoder = commandBuffer->encodeComputeCommands();

        auto rootObject = encoder->bindPipeline(pipelineState1);

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

    compareComputeResult(device, numbersBuffer, Slang::makeArray<float>(10.0));
}

SLANG_UNIT_TEST(linkTimeDefaultD3D12)
{
    runTestImpl(linkTimeDefaultTestImpl, unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(linkTimeDefaultVulkan)
{
    runTestImpl(linkTimeDefaultTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
