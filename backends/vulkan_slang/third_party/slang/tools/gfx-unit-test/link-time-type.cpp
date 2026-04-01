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
    slang::ProgramLayout*& slangReflection)
{
    const char* moduleInterfaceSrc = R"(
            interface IBase : IDifferentiable
            {
                [Differentiable]
                __init(int x);
                [Differentiable]
                float getBaseValue();
                [Differentiable]
                static float getBaseValueS();
            }
            interface IFoo : IBase
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
                [Differentiable]
                float getBaseValue() { return val; }
                [Differentiable]
                static float getBaseValueS() { return 0.0; }
                property float val2 {
                    get { return val + 2.0; }
                    set { val = newValue; }
                }
                [Differentiable]
                __init(int x) { val = x; }
            };
        )";
    const char* module0Src = R"(
            import ifoo;
            extern struct Foo : IFoo;

            [numthreads(1,1,1)]
            void computeMain(uniform RWStructuredBuffer<float> buffer)
            {
                Foo foo = Foo(0);
                foo.setValue(3.0);
                buffer[0] = foo.getValue() + foo.val2 + Foo.offset + foo.getBaseValue();
            }
        )";
    const char* module1Src = R"(
            import ifoo;
            export struct Foo : IFoo = FooImpl;)";
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

void linkTimeTypeTestImpl(IDevice* device, UnitTestContext* context)
{
    Slang::ComPtr<ITransientResourceHeap> transientHeap;
    ITransientResourceHeap::Desc transientHeapDesc = {};
    transientHeapDesc.constantBufferSize = 4096;
    GFX_CHECK_CALL_ABORT(
        device->createTransientResourceHeap(transientHeapDesc, transientHeap.writeRef()));

    ComPtr<IShaderProgram> shaderProgram;
    slang::ProgramLayout* slangReflection;
    GFX_CHECK_CALL_ABORT(loadProgram(device, shaderProgram, slangReflection));

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

    compareComputeResult(device, numbersBuffer, Slang::makeArray<float>(11.0));
}

SLANG_UNIT_TEST(linkTimeTypeD3D12)
{
    runTestImpl(linkTimeTypeTestImpl, unitTestContext, Slang::RenderApiFlag::D3D12);
}

SLANG_UNIT_TEST(linkTimeTypeVulkan)
{
    runTestImpl(linkTimeTypeTestImpl, unitTestContext, Slang::RenderApiFlag::Vulkan);
}

} // namespace gfx_test
