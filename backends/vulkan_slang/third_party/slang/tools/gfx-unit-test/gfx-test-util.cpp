#include "gfx-test-util.h"

#include "slang-com-ptr.h"
#include "unit-test/slang-unit-test.h"

#define GFX_ENABLE_RENDERDOC_INTEGRATION 0
#define GFX_ENABLE_SPIRV_DEBUG 0
#if GFX_ENABLE_RENDERDOC_INTEGRATION
#include "external/renderdoc_app.h"

#include <windows.h>
#endif

using Slang::ComPtr;

namespace gfx_test
{
void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
{
    if (diagnosticsBlob != nullptr)
    {
        getTestReporter()->message(
            TestMessageType::Info,
            (const char*)diagnosticsBlob->getBufferPointer());
    }
}

Slang::Result loadComputeProgram(
    gfx::IDevice* device,
    Slang::ComPtr<gfx::IShaderProgram>& outShaderProgram,
    const char* shaderModuleName,
    const char* entryPointName,
    slang::ProgramLayout*& slangReflection)
{
    Slang::ComPtr<slang::ISession> slangSession;
    SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slang::IModule* module = slangSession->loadModule(shaderModuleName, diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!module)
        return SLANG_FAIL;

    ComPtr<slang::IEntryPoint> computeEntryPoint;
    SLANG_RETURN_ON_FAIL(
        module->findEntryPointByName(entryPointName, computeEntryPoint.writeRef()));

    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(module);
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

Slang::Result loadComputeProgram(
    gfx::IDevice* device,
    slang::ISession* slangSession,
    Slang::ComPtr<gfx::IShaderProgram>& outShaderProgram,
    const char* shaderModuleName,
    const char* entryPointName,
    slang::ProgramLayout*& slangReflection,
    PrecompilationMode precompilationMode)
{
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slang::IModule* module = slangSession->loadModule(shaderModuleName, diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!module)
        return SLANG_FAIL;

    ComPtr<slang::IEntryPoint> computeEntryPoint;
    SLANG_RETURN_ON_FAIL(
        module->findEntryPointByName(entryPointName, computeEntryPoint.writeRef()));

    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(module);
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
    if (precompilationMode == PrecompilationMode::ExternalLink)
    {
        programDesc.downstreamLinkMode = gfx::IShaderProgram::DownstreamLinkMode::Deferred;
    }
    else
    {
        programDesc.downstreamLinkMode = gfx::IShaderProgram::DownstreamLinkMode::None;
    }

    auto shaderProgram = device->createProgram(programDesc);

    outShaderProgram = shaderProgram;
    return SLANG_OK;
}

Slang::Result loadComputeProgramFromSource(
    gfx::IDevice* device,
    Slang::ComPtr<gfx::IShaderProgram>& outShaderProgram,
    Slang::String source)
{
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;

    gfx::IShaderProgram::CreateDesc2 programDesc = {};
    programDesc.sourceType = gfx::ShaderModuleSourceType::SlangSource;
    programDesc.sourceData = (void*)source.getBuffer();
    programDesc.sourceDataSize = source.getLength();

    return device->createProgram2(
        programDesc,
        outShaderProgram.writeRef(),
        diagnosticsBlob.writeRef());
}

Slang::Result loadGraphicsProgram(
    gfx::IDevice* device,
    Slang::ComPtr<gfx::IShaderProgram>& outShaderProgram,
    const char* shaderModuleName,
    const char* vertexEntryPointName,
    const char* fragmentEntryPointName,
    slang::ProgramLayout*& slangReflection)
{
    Slang::ComPtr<slang::ISession> slangSession;
    SLANG_RETURN_ON_FAIL(device->getSlangSession(slangSession.writeRef()));
    Slang::ComPtr<slang::IBlob> diagnosticsBlob;
    slang::IModule* module = slangSession->loadModule(shaderModuleName, diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    if (!module)
        return SLANG_FAIL;

    ComPtr<slang::IEntryPoint> vertexEntryPoint;
    SLANG_RETURN_ON_FAIL(
        module->findEntryPointByName(vertexEntryPointName, vertexEntryPoint.writeRef()));

    ComPtr<slang::IEntryPoint> fragmentEntryPoint;
    SLANG_RETURN_ON_FAIL(
        module->findEntryPointByName(fragmentEntryPointName, fragmentEntryPoint.writeRef()));

    Slang::List<slang::IComponentType*> componentTypes;
    componentTypes.add(module);
    componentTypes.add(vertexEntryPoint);
    componentTypes.add(fragmentEntryPoint);

    Slang::ComPtr<slang::IComponentType> composedProgram;
    SlangResult result = slangSession->createCompositeComponentType(
        componentTypes.getBuffer(),
        componentTypes.getCount(),
        composedProgram.writeRef(),
        diagnosticsBlob.writeRef());
    diagnoseIfNeeded(diagnosticsBlob);
    SLANG_RETURN_ON_FAIL(result);
    slangReflection = composedProgram->getLayout();

    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.slangGlobalScope = composedProgram.get();

    auto shaderProgram = device->createProgram(programDesc);

    outShaderProgram = shaderProgram;
    return SLANG_OK;
}

void compareComputeResult(
    gfx::IDevice* device,
    gfx::ITextureResource* texture,
    gfx::ResourceState state,
    void* expectedResult,
    size_t expectedResultRowPitch,
    size_t rowCount)
{
    // Read back the results.
    ComPtr<ISlangBlob> resultBlob;
    size_t rowPitch = 0;
    size_t pixelSize = 0;
    GFX_CHECK_CALL_ABORT(
        device->readTextureResource(texture, state, resultBlob.writeRef(), &rowPitch, &pixelSize));
    // Compare results.
    for (size_t row = 0; row < rowCount; row++)
    {
        SLANG_CHECK(
            memcmp(
                (uint8_t*)resultBlob->getBufferPointer() + rowPitch * row,
                (uint8_t*)expectedResult + expectedResultRowPitch * row,
                expectedResultRowPitch) == 0);
    }
}

void compareComputeResult(
    gfx::IDevice* device,
    gfx::IBufferResource* buffer,
    size_t offset,
    const void* expectedResult,
    size_t expectedBufferSize)
{
    // Read back the results.
    ComPtr<ISlangBlob> resultBlob;
    GFX_CHECK_CALL_ABORT(
        device->readBufferResource(buffer, offset, expectedBufferSize, resultBlob.writeRef()));
    SLANG_CHECK(resultBlob->getBufferSize() == expectedBufferSize);
    // Compare results.
    SLANG_CHECK(
        memcmp(resultBlob->getBufferPointer(), (uint8_t*)expectedResult, expectedBufferSize) == 0);
}

void compareComputeResultFuzzy(
    const float* result,
    float* expectedResult,
    size_t expectedBufferSize)
{
    for (size_t i = 0; i < expectedBufferSize / sizeof(float); ++i)
    {
        SLANG_CHECK(abs(result[i] - expectedResult[i]) <= 0.01);
    }
}

void compareComputeResultFuzzy(
    gfx::IDevice* device,
    gfx::IBufferResource* buffer,
    float* expectedResult,
    size_t expectedBufferSize)
{
    // Read back the results.
    ComPtr<ISlangBlob> resultBlob;
    GFX_CHECK_CALL_ABORT(
        device->readBufferResource(buffer, 0, expectedBufferSize, resultBlob.writeRef()));
    SLANG_CHECK(resultBlob->getBufferSize() == expectedBufferSize);
    // Compare results with a tolerance of 0.01.
    auto result = (float*)resultBlob->getBufferPointer();
    compareComputeResultFuzzy(result, expectedResult, expectedBufferSize);
}

Slang::ComPtr<gfx::IDevice> createTestingDevice(
    UnitTestContext* context,
    Slang::RenderApiFlag::Enum api,
    Slang::List<const char*> additionalSearchPaths,
    gfx::IDevice::ShaderCacheDesc shaderCache)
{
    Slang::ComPtr<gfx::IDevice> device;
    gfx::IDevice::Desc deviceDesc = {};
    switch (api)
    {
    case Slang::RenderApiFlag::D3D11:
        deviceDesc.deviceType = gfx::DeviceType::DirectX11;
        break;
    case Slang::RenderApiFlag::D3D12:
        deviceDesc.deviceType = gfx::DeviceType::DirectX12;
        break;
    case Slang::RenderApiFlag::Vulkan:
        deviceDesc.deviceType = gfx::DeviceType::Vulkan;
        break;
    case Slang::RenderApiFlag::CPU:
        deviceDesc.deviceType = gfx::DeviceType::CPU;
        break;
    case Slang::RenderApiFlag::CUDA:
        deviceDesc.deviceType = gfx::DeviceType::CUDA;
        break;
    default:
        SLANG_IGNORE_TEST
    }
    deviceDesc.slang.slangGlobalSession = context->slangGlobalSession;
    Slang::List<const char*> searchPaths = getSlangSearchPaths();
    searchPaths.addRange(additionalSearchPaths);
    deviceDesc.slang.searchPaths = searchPaths.getBuffer();
    deviceDesc.slang.searchPathCount = (gfx::GfxCount)searchPaths.getCount();
    deviceDesc.shaderCache = shaderCache;

    gfx::D3D12DeviceExtendedDesc extDesc = {};
    extDesc.rootParameterShaderAttributeName = "root";

    gfx::SlangSessionExtendedDesc slangExtDesc = {};
    Slang::List<slang::CompilerOptionEntry> entries;
    slang::CompilerOptionEntry emitSpirvDirectlyEntry;
    emitSpirvDirectlyEntry.name = slang::CompilerOptionName::EmitSpirvDirectly;
    emitSpirvDirectlyEntry.value.intValue0 = 1;
    entries.add(emitSpirvDirectlyEntry);
#if GFX_ENABLE_SPIRV_DEBUG
    slang::CompilerOptionEntry debugLevelCompilerOptionEntry;
    debugLevelCompilerOptionEntry.name = slang::CompilerOptionName::DebugInformation;
    debugLevelCompilerOptionEntry.value.intValue0 = SLANG_DEBUG_INFO_LEVEL_STANDARD;
    entries.add(debugLevelCompilerOptionEntry);
#endif
    slangExtDesc.compilerOptionEntries = entries.getBuffer();
    slangExtDesc.compilerOptionEntryCount = (uint32_t)entries.getCount();

    deviceDesc.extendedDescCount = 2;
    void* extDescPtrs[2] = {&extDesc, &slangExtDesc};
    deviceDesc.extendedDescs = extDescPtrs;

    // TODO: We should also set the debug callback
    // (And in general reduce the differences (and duplication) between
    // here and render-test-main.cpp)
#ifdef _DEBUG
    gfx::gfxEnableDebugLayer();
#endif

    auto createDeviceResult = gfxCreateDevice(&deviceDesc, device.writeRef());
    if (SLANG_FAILED(createDeviceResult))
    {
        SLANG_IGNORE_TEST
    }
    return device;
}

Slang::List<const char*> getSlangSearchPaths()
{
    Slang::List<const char*> searchPaths;
    searchPaths.add("");
    searchPaths.add("../../tools/gfx-unit-test");
    searchPaths.add("tools/gfx-unit-test");
    return searchPaths;
}

#if GFX_ENABLE_RENDERDOC_INTEGRATION
RENDERDOC_API_1_1_2* rdoc_api = NULL;
void initializeRenderDoc()
{
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
    {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void**)&rdoc_api);
        assert(ret == 1);
    }
}
void renderDocBeginFrame()
{
    if (!rdoc_api)
        initializeRenderDoc();
    if (rdoc_api)
        rdoc_api->StartFrameCapture(nullptr, nullptr);
}
void renderDocEndFrame()
{
    if (rdoc_api)
        rdoc_api->EndFrameCapture(nullptr, nullptr);
    _fgetchar();
}
#else
void initializeRenderDoc() {}
void renderDocBeginFrame() {}
void renderDocEndFrame() {}
#endif
} // namespace gfx_test
