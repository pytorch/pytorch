// cpu-device.cpp
#include "cpu-device.h"

#include "cpu-buffer.h"
#include "cpu-pipeline-state.h"
#include "cpu-query.h"
#include "cpu-resource-views.h"
#include "cpu-shader-object.h"
#include "cpu-shader-program.h"
#include "cpu-texture.h"

#include <chrono>

namespace gfx
{
using namespace Slang;

namespace cpu
{
DeviceImpl::~DeviceImpl()
{
    m_currentPipeline = nullptr;
    m_currentRootObject = nullptr;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::initialize(const Desc& desc)
{
    SLANG_RETURN_ON_FAIL(slangContext.initialize(
        desc.slang,
        desc.extendedDescCount,
        desc.extendedDescs,
        SLANG_SHADER_HOST_CALLABLE,
        "sm_5_1",
        makeArray(slang::PreprocessorMacroDesc{"__CPU__", "1"}).getView()));

    SLANG_RETURN_ON_FAIL(RendererBase::initialize(desc));

    // Initialize DeviceInfo
    {
        m_info.deviceType = DeviceType::CPU;
        m_info.bindingStyle = BindingStyle::CUDA;
        m_info.projectionStyle = ProjectionStyle::DirectX;
        m_info.apiName = "CPU";
        static const float kIdentity[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        ::memcpy(m_info.identityProjectionMatrix, kIdentity, sizeof(kIdentity));
        m_info.adapterName = "CPU";
        m_info.timestampFrequency = 1000000000;
    }

    // Can support pointers (or something akin to that)
    {
        m_features.add("has-ptr");
    }

    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createTextureResource(
    const ITextureResource::Desc& desc,
    const ITextureResource::SubresourceData* initData,
    ITextureResource** outResource)
{
    TextureResource::Desc srcDesc = fixupTextureDesc(desc);

    RefPtr<TextureResourceImpl> texture = new TextureResourceImpl(srcDesc);

    SLANG_RETURN_ON_FAIL(texture->init(initData));

    returnComPtr(outResource, texture);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createBufferResource(
    const IBufferResource::Desc& descIn,
    const void* initData,
    IBufferResource** outResource)
{
    auto desc = fixupBufferDesc(descIn);
    RefPtr<BufferResourceImpl> resource = new BufferResourceImpl(desc);
    SLANG_RETURN_ON_FAIL(resource->init());
    if (initData)
    {
        SLANG_RETURN_ON_FAIL(resource->setData(0, desc.sizeInBytes, initData));
    }
    returnComPtr(outResource, resource);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createTextureView(
    ITextureResource* inTexture,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    auto texture = static_cast<TextureResourceImpl*>(inTexture);
    RefPtr<TextureResourceViewImpl> view = new TextureResourceViewImpl(desc, texture);
    returnComPtr(outView, view);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createBufferView(
    IBufferResource* inBuffer,
    IBufferResource* counterBuffer,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    auto buffer = static_cast<BufferResourceImpl*>(inBuffer);
    RefPtr<BufferResourceViewImpl> view = new BufferResourceViewImpl(desc, buffer);
    returnComPtr(outView, view);
    return SLANG_OK;
}

Result DeviceImpl::createShaderObjectLayout(
    slang::ISession* session,
    slang::TypeLayoutReflection* typeLayout,
    ShaderObjectLayoutBase** outLayout)
{
    RefPtr<ShaderObjectLayoutImpl> cpuLayout =
        new ShaderObjectLayoutImpl(this, session, typeLayout);
    returnRefPtrMove(outLayout, cpuLayout);

    return SLANG_OK;
}

Result DeviceImpl::createShaderObject(ShaderObjectLayoutBase* layout, IShaderObject** outObject)
{
    auto cpuLayout = static_cast<ShaderObjectLayoutImpl*>(layout);

    RefPtr<ShaderObjectImpl> result = new ShaderObjectImpl();
    SLANG_RETURN_ON_FAIL(result->init(this, cpuLayout));
    returnComPtr(outObject, result);

    return SLANG_OK;
}

Result DeviceImpl::createMutableShaderObject(
    ShaderObjectLayoutBase* layout,
    IShaderObject** outObject)
{
    auto cpuLayout = static_cast<ShaderObjectLayoutImpl*>(layout);

    RefPtr<MutableShaderObjectImpl> result = new MutableShaderObjectImpl();
    SLANG_RETURN_ON_FAIL(result->init(this, cpuLayout));
    returnComPtr(outObject, result);

    return SLANG_OK;
}

Result DeviceImpl::createRootShaderObject(IShaderProgram* program, ShaderObjectBase** outObject)
{
    auto cpuProgram = static_cast<ShaderProgramImpl*>(program);
    auto cpuProgramLayout = cpuProgram->layout;

    RefPtr<RootShaderObjectImpl> result = new RootShaderObjectImpl();
    SLANG_RETURN_ON_FAIL(result->init(this, cpuProgramLayout));
    returnRefPtrMove(outObject, result);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createProgram(
    const IShaderProgram::Desc& desc,
    IShaderProgram** outProgram,
    ISlangBlob** outDiagnosticBlob)
{
    RefPtr<ShaderProgramImpl> cpuProgram = new ShaderProgramImpl();
    cpuProgram->init(desc);
    auto slangGlobalScope = cpuProgram->linkedProgram;
    if (slangGlobalScope)
    {
        auto slangProgramLayout = slangGlobalScope->getLayout();
        if (!slangProgramLayout)
            return SLANG_FAIL;

        RefPtr<RootShaderObjectLayoutImpl> cpuProgramLayout = new RootShaderObjectLayoutImpl(
            this,
            slangGlobalScope->getSession(),
            slangProgramLayout);
        cpuProgramLayout->m_programLayout = slangProgramLayout;

        cpuProgram->layout = cpuProgramLayout;
    }

    returnComPtr(outProgram, cpuProgram);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL DeviceImpl::createComputePipelineState(
    const ComputePipelineStateDesc& desc,
    IPipelineState** outState)
{
    RefPtr<PipelineStateImpl> state = new PipelineStateImpl();
    state->init(desc);
    returnComPtr(outState, state);
    return Result();
}

SLANG_NO_THROW Result SLANG_MCALL
DeviceImpl::createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outPool)
{
    RefPtr<QueryPoolImpl> pool = new QueryPoolImpl();
    pool->init(desc);
    returnComPtr(outPool, pool);
    return SLANG_OK;
}

void DeviceImpl::writeTimestamp(IQueryPool* pool, GfxIndex index)
{
    static_cast<QueryPoolImpl*>(pool)->m_queries[index] =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

SLANG_NO_THROW const DeviceInfo& SLANG_MCALL DeviceImpl::getDeviceInfo() const
{
    return m_info;
}

SLANG_NO_THROW Result SLANG_MCALL
DeviceImpl::createSamplerState(ISamplerState::Desc const& desc, ISamplerState** outSampler)
{
    SLANG_UNUSED(desc);
    *outSampler = nullptr;
    return SLANG_OK;
}

void* DeviceImpl::map(IBufferResource* buffer, MapFlavor flavor)
{
    SLANG_UNUSED(flavor);
    auto bufferImpl = static_cast<BufferResourceImpl*>(buffer);
    return bufferImpl->m_data;
}
void DeviceImpl::unmap(IBufferResource* buffer, size_t offsetWritten, size_t sizeWritten)
{
    SLANG_UNUSED(buffer);
    SLANG_UNUSED(offsetWritten);
    SLANG_UNUSED(sizeWritten);
}

void DeviceImpl::setPipelineState(IPipelineState* state)
{
    m_currentPipeline = static_cast<PipelineStateImpl*>(state);
}

void DeviceImpl::bindRootShaderObject(IShaderObject* object)
{
    m_currentRootObject = static_cast<RootShaderObjectImpl*>(object);
}

void DeviceImpl::dispatchCompute(int x, int y, int z)
{
    int entryPointIndex = 0;
    int targetIndex = 0;

    // Specialize the compute kernel based on the shader object bindings.
    RefPtr<PipelineStateBase> newPipeline;
    maybeSpecializePipeline(m_currentPipeline, m_currentRootObject, newPipeline);
    m_currentPipeline = static_cast<PipelineStateImpl*>(newPipeline.Ptr());

    auto program = m_currentPipeline->getProgram();
    auto entryPointLayout = m_currentRootObject->getLayout()->getEntryPoint(entryPointIndex);
    auto entryPointName = entryPointLayout->getEntryPointName();

    auto entryPointObject = m_currentRootObject->getEntryPoint(entryPointIndex);

    ComPtr<ISlangSharedLibrary> sharedLibrary;
    ComPtr<ISlangBlob> diagnostics;
    auto compileResult = program->slangGlobalScope->getEntryPointHostCallable(
        entryPointIndex,
        targetIndex,
        sharedLibrary.writeRef(),
        diagnostics.writeRef());
    if (diagnostics)
    {
        getDebugCallback()->handleMessage(
            compileResult == SLANG_OK ? DebugMessageType::Warning : DebugMessageType::Error,
            DebugMessageSource::Slang,
            (char*)diagnostics->getBufferPointer());
    }
    if (SLANG_FAILED(compileResult))
        return;

    auto func = (slang_prelude::ComputeFunc)sharedLibrary->findSymbolAddressByName(entryPointName);

    slang_prelude::ComputeVaryingInput varyingInput;
    varyingInput.startGroupID.x = 0;
    varyingInput.startGroupID.y = 0;
    varyingInput.startGroupID.z = 0;
    varyingInput.endGroupID.x = x;
    varyingInput.endGroupID.y = y;
    varyingInput.endGroupID.z = z;

    auto globalParamsData = m_currentRootObject->getDataBuffer();
    auto entryPointParamsData = entryPointObject->getDataBuffer();
    func(&varyingInput, entryPointParamsData, globalParamsData);
}

void DeviceImpl::copyBuffer(
    IBufferResource* dst,
    size_t dstOffset,
    IBufferResource* src,
    size_t srcOffset,
    size_t size)
{
    auto dstImpl = static_cast<BufferResourceImpl*>(dst);
    auto srcImpl = static_cast<BufferResourceImpl*>(src);
    memcpy((uint8_t*)dstImpl->m_data + dstOffset, (uint8_t*)srcImpl->m_data + srcOffset, size);
}

} // namespace cpu

Result SLANG_MCALL createCPUDevice(const IDevice::Desc* desc, IDevice** outDevice)
{
    RefPtr<cpu::DeviceImpl> result = new cpu::DeviceImpl();
    SLANG_RETURN_ON_FAIL(result->initialize(*desc));
    returnComPtr(outDevice, result);
    return SLANG_OK;
}

} // namespace gfx
