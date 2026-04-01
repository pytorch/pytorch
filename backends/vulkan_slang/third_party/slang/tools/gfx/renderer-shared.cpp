#include "renderer-shared.h"

#include "../../source/core/slang-file-system.h"
#include "../../source/core/slang-stable-hash.h"
#include "core/slang-io.h"
#include "core/slang-token-reader.h"
#include "mutable-shader-object.h"
#include "slang.h"

using namespace Slang;

namespace gfx
{

const Slang::Guid GfxGUID::IID_ISlangUnknown = SLANG_UUID_ISlangUnknown;
const Slang::Guid GfxGUID::IID_IShaderProgram = SLANG_UUID_IShaderProgram;
const Slang::Guid GfxGUID::IID_IInputLayout = SLANG_UUID_IInputLayout;
const Slang::Guid GfxGUID::IID_IPipelineState = SLANG_UUID_IPipelineState;
const Slang::Guid GfxGUID::IID_ITransientResourceHeap = SLANG_UUID_ITransientResourceHeap;
const Slang::Guid GfxGUID::IID_IResourceView = SLANG_UUID_IResourceView;
const Slang::Guid GfxGUID::IID_IFramebuffer = SLANG_UUID_IFrameBuffer;
const Slang::Guid GfxGUID::IID_IFramebufferLayout = SLANG_UUID_IFramebufferLayout;

const Slang::Guid GfxGUID::IID_ISwapchain = SLANG_UUID_ISwapchain;
const Slang::Guid GfxGUID::IID_ISamplerState = SLANG_UUID_ISamplerState;
const Slang::Guid GfxGUID::IID_IResource = SLANG_UUID_IResource;
const Slang::Guid GfxGUID::IID_IBufferResource = SLANG_UUID_IBufferResource;
const Slang::Guid GfxGUID::IID_ITextureResource = SLANG_UUID_ITextureResource;
const Slang::Guid GfxGUID::IID_IDevice = SLANG_UUID_IDevice;
const Slang::Guid GfxGUID::IID_IShaderCache = SLANG_UUID_IShaderCache;
const Slang::Guid GfxGUID::IID_IShaderObject = SLANG_UUID_IShaderObject;

const Slang::Guid GfxGUID::IID_IRenderPassLayout = SLANG_UUID_IRenderPassLayout;
const Slang::Guid GfxGUID::IID_IRayTracingCommandEncoder = IRayTracingCommandEncoder::getTypeGuid();
const Slang::Guid GfxGUID::IID_IResourceCommandEncoder = IResourceCommandEncoder::getTypeGuid();
const Slang::Guid GfxGUID::IID_IComputeCommandEncoder = IComputeCommandEncoder::getTypeGuid();
const Slang::Guid GfxGUID::IID_IRenderCommandEncoder = IRenderCommandEncoder::getTypeGuid();

const Slang::Guid GfxGUID::IID_ICommandBuffer = SLANG_UUID_ICommandBuffer;
const Slang::Guid GfxGUID::IID_ICommandBufferD3D12 = SLANG_UUID_ICommandBufferD3D12;

const Slang::Guid GfxGUID::IID_ICommandQueue = SLANG_UUID_ICommandQueue;
const Slang::Guid GfxGUID::IID_IQueryPool = SLANG_UUID_IQueryPool;
const Slang::Guid GfxGUID::IID_IAccelerationStructure = SLANG_UUID_IAccelerationStructure;
const Slang::Guid GfxGUID::IID_IFence = SLANG_UUID_IFence;
const Slang::Guid GfxGUID::IID_IShaderTable = SLANG_UUID_IShaderTable;
const Slang::Guid GfxGUID::IID_IPipelineCreationAPIDispatcher =
    SLANG_UUID_IPipelineCreationAPIDispatcher;
const Slang::Guid GfxGUID::IID_IVulkanPipelineCreationAPIDispatcher =
    SLANG_UUID_IVulkanPipelineCreationAPIDispatcher;
const Slang::Guid GfxGUID::IID_ITransientResourceHeapD3D12 = SLANG_UUID_ITransientResourceHeapD3D12;


StageType translateStage(SlangStage slangStage)
{
    switch (slangStage)
    {
    default:
        SLANG_ASSERT(!"unhandled case");
        return gfx::StageType::Unknown;

#define CASE(FROM, TO)       \
    case SLANG_STAGE_##FROM: \
        return gfx::StageType::TO

        CASE(VERTEX, Vertex);
        CASE(HULL, Hull);
        CASE(DOMAIN, Domain);
        CASE(GEOMETRY, Geometry);
        CASE(FRAGMENT, Fragment);

        CASE(COMPUTE, Compute);

        CASE(RAY_GENERATION, RayGeneration);
        CASE(INTERSECTION, Intersection);
        CASE(ANY_HIT, AnyHit);
        CASE(CLOSEST_HIT, ClosestHit);
        CASE(MISS, Miss);
        CASE(CALLABLE, Callable);

#undef CASE
    }
}

IFence* FenceBase::getInterface(const Slang::Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IFence)
        return static_cast<IFence*>(this);
    return nullptr;
}

IResource* BufferResource::getInterface(const Slang::Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IResource ||
        guid == GfxGUID::IID_IBufferResource)
        return static_cast<IBufferResource*>(this);
    return nullptr;
}

SLANG_NO_THROW IResource::Type SLANG_MCALL BufferResource::getType()
{
    return m_type;
}
SLANG_NO_THROW IBufferResource::Desc* SLANG_MCALL BufferResource::getDesc()
{
    return &m_desc;
}

Result BufferResource::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->handleValue = 0;
    outHandle->api = InteropHandleAPI::Unknown;
    return SLANG_FAIL;
}

Result BufferResource::getSharedHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Unknown;
    outHandle->handleValue = 0;
    return SLANG_FAIL;
}

IResource* TextureResource::getInterface(const Slang::Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IResource ||
        guid == GfxGUID::IID_ITextureResource)
        return static_cast<ITextureResource*>(this);
    return nullptr;
}

SLANG_NO_THROW IResource::Type SLANG_MCALL TextureResource::getType()
{
    return m_type;
}
SLANG_NO_THROW ITextureResource::Desc* SLANG_MCALL TextureResource::getDesc()
{
    return &m_desc;
}

Result TextureResource::getNativeResourceHandle(InteropHandle* outHandle)
{
    outHandle->handleValue = 0;
    outHandle->api = InteropHandleAPI::Unknown;
    return SLANG_FAIL;
}

Result TextureResource::getSharedHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Unknown;
    outHandle->handleValue = 0;
    return SLANG_OK;
}

StageType mapStage(SlangStage stage)
{
    switch (stage)
    {
    default:
        return StageType::Unknown;

    case SLANG_STAGE_AMPLIFICATION:
        return gfx::StageType::Amplification;
    case SLANG_STAGE_ANY_HIT:
        return gfx::StageType::AnyHit;
    case SLANG_STAGE_CALLABLE:
        return gfx::StageType::Callable;
    case SLANG_STAGE_CLOSEST_HIT:
        return gfx::StageType::ClosestHit;
    case SLANG_STAGE_COMPUTE:
        return gfx::StageType::Compute;
    case SLANG_STAGE_DOMAIN:
        return gfx::StageType::Domain;
    case SLANG_STAGE_FRAGMENT:
        return gfx::StageType::Fragment;
    case SLANG_STAGE_GEOMETRY:
        return gfx::StageType::Geometry;
    case SLANG_STAGE_HULL:
        return gfx::StageType::Hull;
    case SLANG_STAGE_INTERSECTION:
        return gfx::StageType::Intersection;
    case SLANG_STAGE_MESH:
        return gfx::StageType::Mesh;
    case SLANG_STAGE_MISS:
        return gfx::StageType::Miss;
    case SLANG_STAGE_RAY_GENERATION:
        return gfx::StageType::RayGeneration;
    case SLANG_STAGE_VERTEX:
        return gfx::StageType::Vertex;
    }
}

IResourceView* ResourceViewBase::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IResourceView)
        return static_cast<IResourceView*>(this);
    return nullptr;
}

Result ResourceViewBase::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Unknown;
    outHandle->handleValue = 0;
    return SLANG_E_NOT_IMPLEMENTED;
}

ISamplerState* SamplerStateBase::getInterface(const Slang::Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_ISamplerState)
        return static_cast<ISamplerState*>(this);
    return nullptr;
}

Result SamplerStateBase::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Unknown;
    outHandle->handleValue = 0;
    return SLANG_E_NOT_IMPLEMENTED;
}

IAccelerationStructure* AccelerationStructureBase::getInterface(const Slang::Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IResourceView ||
        guid == GfxGUID::IID_IAccelerationStructure)
        return static_cast<IAccelerationStructure*>(this);
    return nullptr;
}

bool _doesValueFitInExistentialPayload(
    slang::TypeLayoutReflection* concreteTypeLayout,
    slang::TypeLayoutReflection* existentialTypeLayout)
{
    // Our task here is to figure out if a value of `concreteTypeLayout`
    // can fit into an existential value using `existentialTypelayout`.

    // We can start by asking how many bytes the concrete type of the object consumes.
    //
    auto concreteValueSize = concreteTypeLayout->getSize();

    // We can also compute how many bytes the existential-type value provides,
    // but we need to remember that the *payload* part of that value comes after
    // the header with RTTI and witness-table IDs, so the payload is 16 bytes
    // smaller than the entire value.
    //
    auto existentialValueSize = existentialTypeLayout->getSize();
    auto existentialPayloadSize = existentialValueSize - 16;

    // If the concrete type consumes more ordinary bytes than we have in the payload,
    // it cannot possibly fit.
    //
    if (concreteValueSize > existentialPayloadSize)
        return false;

    // It is possible that the ordinary bytes of `concreteTypeLayout` can fit
    // in the payload, but that type might also use storage other than ordinary
    // bytes. In that case, the value would *not* fit, because all the non-ordinary
    // data can't fit in the payload at all.
    //
    auto categoryCount = concreteTypeLayout->getCategoryCount();
    for (unsigned int i = 0; i < categoryCount; ++i)
    {
        auto category = concreteTypeLayout->getCategoryByIndex(i);
        switch (category)
        {
        // We want to ignore any ordinary/uniform data usage, since that
        // was already checked above.
        //
        case slang::ParameterCategory::Uniform:
            break;

        // Any other kind of data consumed means the value cannot possibly fit.
        default:
            return false;

            // TODO: Are there any cases of resource usage that need to be ignored here?
            // E.g., if the sub-object contains its own existential-type fields (which
            // get reflected as consuming "existential value" storage) should that be
            // ignored?
        }
    }

    // If we didn't reject the concrete type above for either its ordinary
    // data or some use of non-ordinary data, then it seems like it must fit.
    //
    return true;
}

IShaderProgram* ShaderProgramBase::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IShaderProgram)
        return static_cast<IShaderProgram*>(this);
    return nullptr;
}

IInputLayout* InputLayoutBase::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IInputLayout)
        return static_cast<IInputLayout*>(this);
    return nullptr;
}

IFramebufferLayout* FramebufferLayoutBase::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IFramebufferLayout)
        return static_cast<IFramebufferLayout*>(this);
    return nullptr;
}

IFramebuffer* FramebufferBase::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IFramebuffer)
        return static_cast<IFramebuffer*>(this);
    return nullptr;
}

IQueryPool* QueryPoolBase::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IQueryPool)
        return static_cast<IQueryPool*>(this);
    return nullptr;
}

IPipelineState* PipelineStateBase::getInterface(const Guid& guid)
{
    if (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IPipelineState)
        return static_cast<IPipelineState*>(this);
    return nullptr;
}

Result PipelineStateBase::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::Unknown;
    outHandle->handleValue = 0;
    return SLANG_E_NOT_IMPLEMENTED;
}

void PipelineStateBase::initializeBase(const PipelineStateDesc& inDesc)
{
    desc = inDesc;

    auto program = desc.getProgram();
    m_program = program;
    isSpecializable = false;
    if (program->slangGlobalScope && program->slangGlobalScope->getSpecializationParamCount() != 0)
        isSpecializable = true;
    for (auto& entryPoint : program->slangEntryPoints)
    {
        if (entryPoint->getSpecializationParamCount() != 0)
        {
            isSpecializable = true;
            break;
        }
    }
    // Hold a strong reference to inputLayout and framebufferLayout objects to prevent it from
    // destruction.
    if (inDesc.type == PipelineType::Graphics)
    {
        inputLayout = static_cast<InputLayoutBase*>(inDesc.graphics.inputLayout);
        framebufferLayout = static_cast<FramebufferLayoutBase*>(inDesc.graphics.framebufferLayout);
    }
}

Result RendererBase::getEntryPointCodeFromShaderCache(
    slang::IComponentType* program,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    slang::IBlob** outCode,
    slang::IBlob** outDiagnostics)
{
    // Immediately call getEntryPointCode if no shader cache has been initialized
    if (!persistentShaderCache)
    {
        return program->getEntryPointCode(entryPointIndex, targetIndex, outCode, outDiagnostics);
    }

    // Hash all relevant state for generating the entry point shader code to use as a key
    // for the shader cache.
    ComPtr<ISlangBlob> hashBlob;
    program->getEntryPointHash(entryPointIndex, targetIndex, hashBlob.writeRef());
    PersistentCache::Key cacheKey(hashBlob);

    // Query the shader cache.
    ComPtr<ISlangBlob> codeBlob;
    if (persistentShaderCache->readEntry(cacheKey, codeBlob.writeRef()) != SLANG_OK)
    {
        // No cached entry found. Generate the code and add it to the cache.
        SLANG_RETURN_ON_FAIL(program->getEntryPointCode(
            entryPointIndex,
            targetIndex,
            codeBlob.writeRef(),
            outDiagnostics));
        persistentShaderCache->writeEntry(cacheKey, codeBlob);
    }

    *outCode = codeBlob.detach();
    return SLANG_OK;
}

SlangResult RendererBase::queryInterface(SlangUUID const& uuid, void** outObject)
{
    // Only return the shader cache interface if it is enabled.
    if (uuid == GfxGUID::IID_IShaderCache && persistentShaderCache)
    {
        *outObject = static_cast<IShaderCache*>(this);
        addRef();
        return SLANG_OK;
    }

    if (IDevice* device_ptr = getInterface(uuid))
    {
        *outObject = device_ptr;
        addRef();
        return SLANG_OK;
    }
    return SLANG_E_NO_INTERFACE;
}

IDevice* gfx::RendererBase::getInterface(const Guid& guid)
{
    return (guid == GfxGUID::IID_ISlangUnknown || guid == GfxGUID::IID_IDevice)
               ? static_cast<IDevice*>(this)
               : nullptr;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::initialize(const Desc& desc)
{
    // We only want to initialize the shader cache if a shader cache path was provided.
    if (desc.shaderCache.shaderCachePath)
    {
        PersistentCache::Desc cacheDesc;
        cacheDesc.directory = desc.shaderCache.shaderCachePath;
        cacheDesc.maxEntryCount = desc.shaderCache.maxEntryCount;
        persistentShaderCache = new PersistentCache(cacheDesc);
    }

    if (desc.apiCommandDispatcher)
    {
        if (desc.deviceType == DeviceType::Vulkan)
        {
            desc.apiCommandDispatcher->queryInterface(
                GfxGUID::IID_IVulkanPipelineCreationAPIDispatcher,
                (void**)m_pipelineCreationAPIDispatcher.writeRef());
        }
        else
        {
            desc.apiCommandDispatcher->queryInterface(
                GfxGUID::IID_IPipelineCreationAPIDispatcher,
                (void**)m_pipelineCreationAPIDispatcher.writeRef());
        }
    }
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::getNativeDeviceHandles(InteropHandles* outHandles)
{
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
RendererBase::getFeatures(const char** outFeatures, Size bufferSize, GfxCount* outFeatureCount)
{
    if (bufferSize >= (UInt)m_features.getCount())
    {
        for (Index i = 0; i < m_features.getCount(); i++)
        {
            outFeatures[i] = m_features[i].getUnownedSlice().begin();
        }
    }
    if (outFeatureCount)
        *outFeatureCount = (GfxCount)m_features.getCount();
    return SLANG_OK;
}

SLANG_NO_THROW bool SLANG_MCALL RendererBase::hasFeature(const char* featureName)
{
    return m_features.findFirstIndex([&](Slang::String x) { return x == featureName; }) != -1;
}

Result RendererBase::getFormatSupportedResourceStates(Format format, ResourceStateSet* outStates)
{
    SLANG_UNUSED(format);
    outStates->add(ResourceState::AccelerationStructure);
    outStates->add(ResourceState::AccelerationStructureBuildInput);
    outStates->add(ResourceState::ConstantBuffer);
    outStates->add(ResourceState::CopyDestination);
    outStates->add(ResourceState::CopySource);
    outStates->add(ResourceState::DepthRead);
    outStates->add(ResourceState::DepthWrite);
    outStates->add(ResourceState::IndexBuffer);
    outStates->add(ResourceState::IndirectArgument);
    outStates->add(ResourceState::PreInitialized);
    outStates->add(ResourceState::Present);
    outStates->add(ResourceState::RenderTarget);
    outStates->add(ResourceState::ResolveDestination);
    outStates->add(ResourceState::ResolveSource);
    outStates->add(ResourceState::ShaderResource);
    outStates->add(ResourceState::PixelShaderResource);
    outStates->add(ResourceState::NonPixelShaderResource);
    outStates->add(ResourceState::StreamOutput);
    outStates->add(ResourceState::Undefined);
    outStates->add(ResourceState::UnorderedAccess);
    outStates->add(ResourceState::VertexBuffer);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::getSlangSession(slang::ISession** outSlangSession)
{
    *outSlangSession = slangContext.session.get();
    slangContext.session->addRef();
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createTextureFromNativeHandle(
    InteropHandle handle,
    const ITextureResource::Desc& srcDesc,
    ITextureResource** outResource)
{
    SLANG_UNUSED(handle);
    SLANG_UNUSED(srcDesc);
    SLANG_UNUSED(outResource);
    return SLANG_E_NOT_AVAILABLE;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createTextureFromSharedHandle(
    InteropHandle handle,
    const ITextureResource::Desc& srcDesc,
    const Size size,
    ITextureResource** outResource)
{
    SLANG_UNUSED(handle);
    SLANG_UNUSED(srcDesc);
    SLANG_UNUSED(size);
    SLANG_UNUSED(outResource);
    return SLANG_E_NOT_AVAILABLE;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createBufferFromNativeHandle(
    InteropHandle handle,
    const IBufferResource::Desc& srcDesc,
    IBufferResource** outResource)
{
    SLANG_UNUSED(handle);
    SLANG_UNUSED(srcDesc);
    SLANG_UNUSED(outResource);
    return SLANG_E_NOT_AVAILABLE;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createBufferFromSharedHandle(
    InteropHandle handle,
    const IBufferResource::Desc& srcDesc,
    IBufferResource** outResource)
{
    SLANG_UNUSED(handle);
    SLANG_UNUSED(srcDesc);
    SLANG_UNUSED(outResource);
    return SLANG_E_NOT_AVAILABLE;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createShaderObject(
    slang::TypeReflection* type,
    ShaderObjectContainerType container,
    IShaderObject** outObject)
{
    return createShaderObject2(slangContext.session, type, container, outObject);
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createShaderObject2(
    slang::ISession* slangSession,
    slang::TypeReflection* type,
    ShaderObjectContainerType container,
    IShaderObject** outObject)
{
    RefPtr<ShaderObjectLayoutBase> shaderObjectLayout;
    SLANG_RETURN_ON_FAIL(
        getShaderObjectLayout(slangSession, type, container, shaderObjectLayout.writeRef()));
    return createShaderObject(shaderObjectLayout, outObject);
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createMutableShaderObject(
    slang::TypeReflection* type,
    ShaderObjectContainerType containerType,
    IShaderObject** outObject)
{
    return createMutableShaderObject2(slangContext.session, type, containerType, outObject);
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createMutableShaderObject2(
    slang::ISession* slangSession,
    slang::TypeReflection* type,
    ShaderObjectContainerType containerType,
    IShaderObject** outObject)
{
    RefPtr<ShaderObjectLayoutBase> shaderObjectLayout;
    SLANG_RETURN_ON_FAIL(
        getShaderObjectLayout(slangSession, type, containerType, shaderObjectLayout.writeRef()));
    return createMutableShaderObject(shaderObjectLayout, outObject);
}

Result RendererBase::createProgram2(
    const IShaderProgram::CreateDesc2& desc,
    IShaderProgram** outProgram,
    ISlangBlob** outDiagnostic)
{
    auto slangSession = slangContext.session.get();
    slang::IModule* module = nullptr;
    ComPtr<slang::IBlob> diagnosticsBlob;
    switch (desc.sourceType)
    {
    case ShaderModuleSourceType::SlangSourceFile:
        {
            auto fileName = (char*)desc.sourceData;
            module = slangSession->loadModule(fileName, diagnosticsBlob.writeRef());
            if (!module)
                return SLANG_FAIL;
            break;
        }
    case ShaderModuleSourceType::SlangSource:
        {
            auto hash = getStableHashCode32((char*)desc.sourceData, desc.sourceDataSize);
            auto hashStr = String(hash);
            auto srcBlob = UnownedRawBlob::create(desc.sourceData, desc.sourceDataSize);
            module = slangSession->loadModuleFromSource(
                hashStr.getBuffer(),
                hashStr.getBuffer(),
                srcBlob,
                diagnosticsBlob.writeRef());
            if (!module)
                return SLANG_FAIL;
            break;
        }
    default:
        SLANG_RELEASE_ASSERT(false);
    }

    Slang::List<ComPtr<slang::IComponentType>> componentTypes;
    componentTypes.add(ComPtr<slang::IComponentType>(module));

    if (desc.entryPointCount == 0)
    {
        for (SlangInt32 i = 0; i < module->getDefinedEntryPointCount(); i++)
        {
            ComPtr<slang::IEntryPoint> entryPoint;
            SLANG_RETURN_ON_FAIL(module->getDefinedEntryPoint(i, entryPoint.writeRef()));
            componentTypes.add(ComPtr<slang::IComponentType>(entryPoint.get()));
        }
    }
    else
    {
        for (GfxCount i = 0; i < desc.entryPointCount; i++)
        {
            ComPtr<slang::IEntryPoint> entryPoint;
            SLANG_RETURN_ON_FAIL(
                module->findEntryPointByName(desc.entryPointNames[i], entryPoint.writeRef()));
            componentTypes.add(ComPtr<slang::IComponentType>(entryPoint.get()));
        }
    }

    Slang::List<slang::IComponentType*> rawComponentTypes;
    for (auto& compType : componentTypes)
        rawComponentTypes.add(compType.get());

    ComPtr<slang::IComponentType> linkedProgram;
    SlangResult result = slangSession->createCompositeComponentType(
        rawComponentTypes.getBuffer(),
        rawComponentTypes.getCount(),
        linkedProgram.writeRef(),
        diagnosticsBlob.writeRef());
    SLANG_RETURN_ON_FAIL(result);

    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.slangGlobalScope = linkedProgram;
    SLANG_RETURN_ON_FAIL(createProgram(programDesc, outProgram, outDiagnostic));

    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createShaderObjectFromTypeLayout(
    slang::TypeLayoutReflection* typeLayout,
    IShaderObject** outObject)
{
    RefPtr<ShaderObjectLayoutBase> shaderObjectLayout;
    SLANG_RETURN_ON_FAIL(
        getShaderObjectLayout(slangContext.session, typeLayout, shaderObjectLayout.writeRef()));
    return createShaderObject(shaderObjectLayout, outObject);
}

SLANG_NO_THROW Result SLANG_MCALL RendererBase::createMutableShaderObjectFromTypeLayout(
    slang::TypeLayoutReflection* typeLayout,
    IShaderObject** outObject)
{
    RefPtr<ShaderObjectLayoutBase> shaderObjectLayout;
    SLANG_RETURN_ON_FAIL(
        getShaderObjectLayout(slangContext.session, typeLayout, shaderObjectLayout.writeRef()));
    return createMutableShaderObject(shaderObjectLayout, outObject);
}

Result RendererBase::getAccelerationStructurePrebuildInfo(
    const IAccelerationStructure::BuildInputs& buildInputs,
    IAccelerationStructure::PrebuildInfo* outPrebuildInfo)
{
    SLANG_UNUSED(buildInputs);
    SLANG_UNUSED(outPrebuildInfo);
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::createAccelerationStructure(
    const IAccelerationStructure::CreateDesc& desc,
    IAccelerationStructure** outView)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outView);
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::createShaderTable(const IShaderTable::Desc& desc, IShaderTable** outTable)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outTable);
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::createRayTracingPipelineState(
    const RayTracingPipelineStateDesc& desc,
    IPipelineState** outState)
{
    SLANG_UNUSED(desc);
    SLANG_UNUSED(outState);
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::createMutableRootShaderObject(
    IShaderProgram* program,
    IShaderObject** outObject)
{
    SLANG_UNUSED(program);
    SLANG_UNUSED(outObject);
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::createFence(const IFence::Desc& desc, IFence** outFence)
{
    SLANG_UNUSED(desc);
    *outFence = nullptr;
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::waitForFences(
    GfxCount fenceCount,
    IFence** fences,
    uint64_t* fenceValues,
    bool waitForAll,
    uint64_t timeout)
{
    SLANG_UNUSED(fenceCount);
    SLANG_UNUSED(fences);
    SLANG_UNUSED(fenceValues);
    SLANG_UNUSED(waitForAll);
    SLANG_UNUSED(timeout);
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::getTextureAllocationInfo(
    const ITextureResource::Desc& desc,
    Size* outSize,
    Size* outAlignment)
{
    SLANG_UNUSED(desc);
    *outSize = 0;
    *outAlignment = 0;
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::getTextureRowAlignment(Size* outAlignment)
{
    *outAlignment = 0;
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::getCooperativeVectorProperties(
    CooperativeVectorProperties* properties,
    uint32_t* propertyCount)
{
    *propertyCount = 0;
    return SLANG_E_NOT_AVAILABLE;
}

Result RendererBase::getShaderObjectLayout(
    slang::ISession* session,
    slang::TypeReflection* type,
    ShaderObjectContainerType container,
    ShaderObjectLayoutBase** outLayout)
{
    switch (container)
    {
    case ShaderObjectContainerType::StructuredBuffer:
        type = session->getContainerType(type, slang::ContainerType::StructuredBuffer);
        break;
    case ShaderObjectContainerType::Array:
        type = session->getContainerType(type, slang::ContainerType::UnsizedArray);
        break;
    default:
        break;
    }

    auto typeLayout = session->getTypeLayout(type);
    SLANG_RETURN_ON_FAIL(getShaderObjectLayout(session, typeLayout, outLayout));
    (*outLayout)->m_slangSession = session;
    return SLANG_OK;
}

Result RendererBase::getShaderObjectLayout(
    slang::ISession* session,
    slang::TypeLayoutReflection* typeLayout,
    ShaderObjectLayoutBase** outLayout)
{
    RefPtr<ShaderObjectLayoutBase> shaderObjectLayout;
    if (!m_shaderObjectLayoutCache.tryGetValue(typeLayout, shaderObjectLayout))
    {
        SLANG_RETURN_ON_FAIL(
            createShaderObjectLayout(session, typeLayout, shaderObjectLayout.writeRef()));
        m_shaderObjectLayoutCache.add(typeLayout, shaderObjectLayout);
    }
    *outLayout = shaderObjectLayout.detach();
    return SLANG_OK;
}

Result RendererBase::clearShaderCache()
{
    SLANG_ASSERT(persistentShaderCache);
    return persistentShaderCache->clear();
}

Result RendererBase::getShaderCacheStats(ShaderCacheStats* outStats)
{
    SLANG_ASSERT(persistentShaderCache);
    if (!outStats)
    {
        return SLANG_E_INVALID_ARG;
    }

    const auto& stats = persistentShaderCache->getStats();
    outStats->entryCount = (GfxCount)stats.entryCount;
    outStats->hitCount = (GfxCount)stats.hitCount;
    outStats->missCount = (GfxCount)stats.missCount;
    return SLANG_OK;
}

Result RendererBase::resetShaderCacheStats()
{
    SLANG_ASSERT(persistentShaderCache);
    persistentShaderCache->resetStats();
    return SLANG_OK;
}

ShaderComponentID ShaderCache::getComponentId(slang::TypeReflection* type)
{
    ComponentKey key;
    key.typeName = UnownedStringSlice(type->getName());
    switch (type->getKind())
    {
    case slang::TypeReflection::Kind::Specialized:
        {
            auto baseType = type->getElementType();

            StringBuilder builder;
            builder.append(UnownedTerminatedStringSlice(baseType->getName()));

            auto rawType = (SlangReflectionType*)type;

            builder.appendChar('<');
            SlangInt argCount = spReflectionType_getSpecializedTypeArgCount(rawType);
            for (SlangInt a = 0; a < argCount; ++a)
            {
                if (a != 0)
                    builder.appendChar(',');
                if (auto rawArgType = spReflectionType_getSpecializedTypeArgType(rawType, a))
                {
                    auto argType = (slang::TypeReflection*)rawArgType;
                    builder.append(argType->getName());
                }
            }
            builder.appendChar('>');
            key.typeName = builder.getUnownedSlice();
            key.updateHash();
            return getComponentId(key);
        }
        // TODO: collect specialization arguments and append them to `key`.
        SLANG_UNIMPLEMENTED_X("specialized type");
    default:
        break;
    }
    key.updateHash();
    return getComponentId(key);
}

ShaderComponentID ShaderCache::getComponentId(UnownedStringSlice name)
{
    ComponentKey key;
    key.typeName = name;
    key.updateHash();
    return getComponentId(key);
}

ShaderComponentID ShaderCache::getComponentId(ComponentKey key)
{
    ShaderComponentID componentId = 0;
    if (componentIds.tryGetValue(key, componentId))
        return componentId;
    OwningComponentKey owningTypeKey;
    owningTypeKey.hash = key.hash;
    owningTypeKey.typeName = key.typeName;
    owningTypeKey.specializationArgs.addRange(key.specializationArgs);
    ShaderComponentID resultId = static_cast<ShaderComponentID>(componentIds.getCount());
    componentIds[owningTypeKey] = resultId;
    return resultId;
}

void ShaderCache::addSpecializedPipeline(
    PipelineKey key,
    Slang::RefPtr<PipelineStateBase> specializedPipeline)
{
    specializedPipelines[key] = specializedPipeline;
}

void ShaderObjectLayoutBase::initBase(
    RendererBase* renderer,
    slang::ISession* session,
    slang::TypeLayoutReflection* elementTypeLayout)
{
    m_renderer = renderer;
    m_slangSession = session;
    m_elementTypeLayout = elementTypeLayout;
    m_componentID = m_renderer->shaderCache.getComponentId(m_elementTypeLayout->getType());
}

// Get the final type this shader object represents. If the shader object's type has existential
// fields, this function will return a specialized type using the bound sub-objects' type as
// specialization argument.
Result ShaderObjectBase::getSpecializedShaderObjectType(ExtendedShaderObjectType* outType)
{
    return _getSpecializedShaderObjectType(outType);
}

Result ShaderObjectBase::_getSpecializedShaderObjectType(ExtendedShaderObjectType* outType)
{
    if (shaderObjectType.slangType)
        *outType = shaderObjectType;
    ExtendedShaderObjectTypeList specializationArgs;
    SLANG_RETURN_ON_FAIL(collectSpecializationArgs(specializationArgs));
    if (specializationArgs.getCount() == 0)
    {
        shaderObjectType.componentID = getLayoutBase()->getComponentID();
        shaderObjectType.slangType = getLayoutBase()->getElementTypeLayout()->getType();
    }
    else
    {
        shaderObjectType.slangType = getRenderer()->slangContext.session->specializeType(
            _getElementTypeLayout()->getType(),
            specializationArgs.components.getArrayView().getBuffer(),
            specializationArgs.getCount());
        shaderObjectType.componentID =
            getRenderer()->shaderCache.getComponentId(shaderObjectType.slangType);
    }
    *outType = shaderObjectType;
    return SLANG_OK;
}

Result ShaderObjectBase::setExistentialHeader(
    slang::TypeReflection* existentialType,
    slang::TypeReflection* concreteType,
    ShaderOffset offset)
{
    // The first field of the tuple (offset zero) is the run-time type information
    // (RTTI) ID for the concrete type being stored into the field.
    //
    // TODO: We need to be able to gather the RTTI type ID from `object` and then
    // use `setData(offset, &TypeID, sizeof(TypeID))`.

    // The second field of the tuple (offset 8) is the ID of the "witness" for the
    // conformance of the concrete type to the interface used by this field.
    //
    auto witnessTableOffset = offset;
    witnessTableOffset.uniformOffset += 8;
    //
    // Conformances of a type to an interface are computed and then stored by the
    // Slang runtime, so we can look up the ID for this particular conformance (which
    // will create it on demand).
    //
    // Note: If the type doesn't actually conform to the required interface for
    // this sub-object range, then this is the point where we will detect that
    // fact and error out.
    //
    uint32_t conformanceID = 0xFFFFFFFF;
    SLANG_RETURN_ON_FAIL(getLayoutBase()->m_slangSession->getTypeConformanceWitnessSequentialID(
        concreteType,
        existentialType,
        &conformanceID));
    //
    // Once we have the conformance ID, then we can write it into the object
    // at the required offset.
    //
    SLANG_RETURN_ON_FAIL(setData(witnessTableOffset, &conformanceID, sizeof(conformanceID)));

    return SLANG_OK;
}

ResourceViewBase* SimpleShaderObjectData::getResourceView(
    RendererBase* device,
    slang::TypeLayoutReflection* elementLayout,
    slang::BindingType bindingType)
{
    if (!m_structuredBuffer)
    {
        // Create structured buffer resource if it has not been created.
        IBufferResource::Desc desc = {};
        desc.allowedStates =
            ResourceStateSet(ResourceState::ShaderResource, ResourceState::UnorderedAccess);
        desc.defaultState = ResourceState::ShaderResource;
        desc.elementSize = (int)elementLayout->getSize();
        desc.format = Format::Unknown;
        desc.type = IResource::Type::Buffer;
        desc.sizeInBytes = (Size)m_ordinaryData.getCount();
        ComPtr<IBufferResource> bufferResource;
        SLANG_RETURN_NULL_ON_FAIL(device->createBufferResource(
            desc,
            m_ordinaryData.getBuffer(),
            bufferResource.writeRef()));
        m_structuredBuffer = static_cast<BufferResource*>(bufferResource.get());

        // Create read-only (shader-resource) and mutable (unordered access) views.
        ComPtr<IResourceView> resourceView;
        IResourceView::Desc viewDesc = {};
        viewDesc.format = Format::Unknown;
        viewDesc.type = IResourceView::Type::ShaderResource;
        SLANG_RETURN_NULL_ON_FAIL(device->createBufferView(
            bufferResource.get(),
            nullptr,
            viewDesc,
            resourceView.writeRef()));
        m_structuredBufferView = static_cast<ResourceViewBase*>(resourceView.get());
        viewDesc.type = IResourceView::Type::UnorderedAccess;
        SLANG_RETURN_NULL_ON_FAIL(device->createBufferView(
            bufferResource.get(),
            nullptr,
            viewDesc,
            resourceView.writeRef()));
        m_rwStructuredBufferView = static_cast<ResourceViewBase*>(resourceView.get());
    }

    switch (bindingType)
    {
    case slang::BindingType::RawBuffer:
        return m_structuredBufferView.Ptr();
    case slang::BindingType::MutableRawBuffer:
        return m_rwStructuredBufferView.Ptr();
    default:
        SLANG_ASSERT(false && "Invalid binding type.");
        return nullptr;
    }
}

void ShaderProgramBase::init(const IShaderProgram::Desc& inDesc)
{
    desc = inDesc;

    slangGlobalScope = desc.slangGlobalScope;
    for (GfxIndex i = 0; i < desc.entryPointCount; i++)
    {
        slangEntryPoints.add(ComPtr<slang::IComponentType>(desc.slangEntryPoints[i]));
    }

    auto session = desc.slangGlobalScope ? desc.slangGlobalScope->getSession() : nullptr;
    if (desc.linkingStyle == IShaderProgram::LinkingStyle::SingleProgram)
    {
        List<slang::IComponentType*> components;
        if (desc.slangGlobalScope)
        {
            components.add(desc.slangGlobalScope);
        }
        for (GfxIndex i = 0; i < desc.entryPointCount; i++)
        {
            if (!session)
            {
                session = desc.slangEntryPoints[i]->getSession();
            }
            components.add(desc.slangEntryPoints[i]);
        }
        session->createCompositeComponentType(
            components.getBuffer(),
            components.getCount(),
            linkedProgram.writeRef());
    }
    else
    {
        for (GfxIndex i = 0; i < desc.entryPointCount; i++)
        {
            if (desc.slangGlobalScope)
            {
                slang::IComponentType* entryPointComponents[2] = {
                    desc.slangGlobalScope,
                    desc.slangEntryPoints[i]};
                ComPtr<slang::IComponentType> linkedEntryPoint;
                session->createCompositeComponentType(
                    entryPointComponents,
                    2,
                    linkedEntryPoint.writeRef());
                linkedEntryPoints.add(linkedEntryPoint);
            }
            else
            {
                linkedEntryPoints.add(ComPtr<slang::IComponentType>(desc.slangEntryPoints[i]));
            }
        }
        linkedProgram = desc.slangGlobalScope;
    }
}

Result ShaderProgramBase::compileShaders(RendererBase* device)
{
    auto compileTarget = device->slangContext.compileTarget;
    // For a fully specialized program, read and store its kernel code in `shaderProgram`.
    auto compileShader = [&](slang::EntryPointReflection* entryPointInfo,
                             slang::IComponentType* entryPointComponent,
                             SlangInt entryPointIndex)
    {
        auto stage = entryPointInfo->getStage();
        List<ComPtr<ISlangBlob>> kernelCodes;
        {
            ComPtr<ISlangBlob> downstreamIR;
            ComPtr<ISlangBlob> diagnostics;
            auto compileResult = device->getEntryPointCodeFromShaderCache(
                entryPointComponent,
                entryPointIndex,
                0,
                downstreamIR.writeRef(),
                diagnostics.writeRef());
            if (diagnostics)
            {
                DebugMessageType msgType = DebugMessageType::Warning;
                if (compileResult != SLANG_OK)
                    msgType = DebugMessageType::Error;
                getDebugCallback()->handleMessage(
                    msgType,
                    DebugMessageSource::Slang,
                    (char*)diagnostics->getBufferPointer());
            }
            SLANG_RETURN_ON_FAIL(compileResult);

            kernelCodes.add(downstreamIR);
        }

        // If target precompilation with deferred downstream linking is enabled,
        // kernelCode may only represent the glue code holding together the
        // bits of precompiled target IR. It's the application's job to pull it
        // together. Collect those dependency target IRs too.
        ComPtr<slang::IModulePrecompileService_Experimental> componentPrecompileService;
        if (this->desc.downstreamLinkMode == DownstreamLinkMode::Deferred &&
            entryPointComponent->queryInterface(
                slang::IModulePrecompileService_Experimental::getTypeGuid(),
                (void**)componentPrecompileService.writeRef()) == SLANG_OK)
        {
            SlangInt dependencyCount = componentPrecompileService->getModuleDependencyCount();
            if (dependencyCount > 0)
            {
                for (int dependencyIndex = 0; dependencyIndex < dependencyCount; dependencyIndex++)
                {
                    ComPtr<slang::IModule> dependencyModule;
                    {
                        ComPtr<slang::IBlob> diagnosticsBlob;
                        auto result = componentPrecompileService->getModuleDependency(
                            dependencyIndex,
                            dependencyModule.writeRef(),
                            diagnosticsBlob.writeRef());
                        if (diagnosticsBlob)
                        {
                            DebugMessageType msgType = DebugMessageType::Warning;
                            if (result != SLANG_OK)
                                msgType = DebugMessageType::Error;
                            getDebugCallback()->handleMessage(
                                msgType,
                                DebugMessageSource::Slang,
                                (char*)diagnosticsBlob->getBufferPointer());
                        }
                        SLANG_RETURN_ON_FAIL(result);
                    }

                    ComPtr<slang::IBlob> downstreamIR;
                    {
                        ComPtr<slang::IBlob> diagnosticsBlob;
                        SlangResult result = SLANG_OK;
                        ComPtr<slang::IModulePrecompileService_Experimental> precompileService;
                        result = dependencyModule->queryInterface(
                            slang::IModulePrecompileService_Experimental::getTypeGuid(),
                            (void**)precompileService.writeRef());
                        if (result == SLANG_OK)
                        {
                            ComPtr<slang::IBlob> diagnosticsBlob;
                            auto result = precompileService->getPrecompiledTargetCode(
                                compileTarget,
                                downstreamIR.writeRef(),
                                diagnosticsBlob.writeRef());
                            if (result == SLANG_OK)
                            {
                                kernelCodes.add(downstreamIR);
                            }
                            if (diagnosticsBlob)
                            {
                                DebugMessageType msgType = DebugMessageType::Warning;
                                if (result != SLANG_OK)
                                    msgType = DebugMessageType::Error;
                                getDebugCallback()->handleMessage(
                                    msgType,
                                    DebugMessageSource::Slang,
                                    (char*)diagnosticsBlob->getBufferPointer());
                            }
                        }
                        SLANG_RETURN_ON_FAIL(result);
                    }
                }
            }
        }

        SLANG_RETURN_ON_FAIL(createShaderModule(entryPointInfo, kernelCodes));
        return SLANG_OK;
    };

    if (linkedEntryPoints.getCount() == 0)
    {
        // If the user does not explicitly specify entry point components, find them from
        // `linkedEntryPoints`.
        auto programReflection = linkedProgram->getLayout();
        for (SlangUInt i = 0; i < programReflection->getEntryPointCount(); i++)
        {
            SLANG_RETURN_ON_FAIL(compileShader(
                programReflection->getEntryPointByIndex(i),
                linkedProgram,
                (SlangInt)i));
        }
    }
    else
    {
        // If the user specifies entry point components via the separated entry point array,
        // compile code from there.
        for (auto& entryPoint : linkedEntryPoints)
        {
            SLANG_RETURN_ON_FAIL(
                compileShader(entryPoint->getLayout()->getEntryPointByIndex(0), entryPoint, 0));
        }
    }
    return SLANG_OK;
}

Result ShaderProgramBase::createShaderModule(
    slang::EntryPointReflection* entryPointInfo,
    List<ComPtr<ISlangBlob>>& kernelCodes)
{
    SLANG_UNUSED(entryPointInfo);
    SLANG_UNUSED(kernelCodes);
    return SLANG_OK;
}

bool ShaderProgramBase::isMeshShaderProgram() const
{
    // Similar to above, interrogate either explicity specified entry point
    // componenets or the ones in the linked program entry point array
    if (linkedEntryPoints.getCount())
    {
        for (const auto& e : linkedEntryPoints)
            if (e->getLayout()->getEntryPointByIndex(0)->getStage() == SLANG_STAGE_MESH)
                return true;
    }
    else
    {
        const auto programReflection = linkedProgram->getLayout();
        for (SlangUInt i = 0; i < programReflection->getEntryPointCount(); ++i)
            if (programReflection->getEntryPointByIndex(i)->getStage() == SLANG_STAGE_MESH)
                return true;
    }
    return false;
}

Result RendererBase::maybeSpecializePipeline(
    PipelineStateBase* currentPipeline,
    ShaderObjectBase* rootObject,
    RefPtr<PipelineStateBase>& outNewPipeline)
{
    outNewPipeline = static_cast<PipelineStateBase*>(currentPipeline);

    auto pipelineType = currentPipeline->desc.type;
    if (currentPipeline->unspecializedPipelineState)
        currentPipeline = currentPipeline->unspecializedPipelineState;
    // If the currently bound pipeline is specializable, we need to specialize it based on bound
    // shader objects.
    if (currentPipeline->isSpecializable)
    {
        specializationArgs.clear();
        SLANG_RETURN_ON_FAIL(rootObject->collectSpecializationArgs(specializationArgs));

        // Construct a shader cache key that represents the specialized shader kernels.
        PipelineKey pipelineKey;
        pipelineKey.pipeline = currentPipeline;
        pipelineKey.specializationArgs.addRange(specializationArgs.componentIDs);
        pipelineKey.updateHash();

        RefPtr<PipelineStateBase> specializedPipelineState =
            shaderCache.getSpecializedPipelineState(pipelineKey);
        // Try to find specialized pipeline from shader cache.
        if (!specializedPipelineState)
        {
            auto unspecializedProgram = static_cast<ShaderProgramBase*>(
                pipelineType == PipelineType::Compute ? currentPipeline->desc.compute.program
                                                      : currentPipeline->desc.graphics.program);
            auto unspecializedProgramLayout = unspecializedProgram->linkedProgram->getLayout();

            ComPtr<slang::IComponentType> specializedComponentType;
            ComPtr<slang::IBlob> diagnosticBlob;
            auto compileRs = unspecializedProgram->linkedProgram->specialize(
                specializationArgs.components.getArrayView().getBuffer(),
                specializationArgs.getCount(),
                specializedComponentType.writeRef(),
                diagnosticBlob.writeRef());
            if (diagnosticBlob)
            {
                getDebugCallback()->handleMessage(
                    compileRs == SLANG_OK ? DebugMessageType::Warning : DebugMessageType::Error,
                    DebugMessageSource::Slang,
                    (char*)diagnosticBlob->getBufferPointer());
            }
            SLANG_RETURN_ON_FAIL(compileRs);

            // Now create the specialized shader program using compiled binaries.
            ComPtr<IShaderProgram> specializedProgram;
            IShaderProgram::Desc specializedProgramDesc = unspecializedProgram->desc;
            specializedProgramDesc.slangGlobalScope = specializedComponentType;

            if (specializedProgramDesc.linkingStyle == IShaderProgram::LinkingStyle::SingleProgram)
            {
                // When linking style is GraphicsCompute, the specialized global scope already
                // contains entry-points, so we do not need to supply them again when creating the
                // specialized pipeline.
                specializedProgramDesc.entryPointCount = 0;
            }
            SLANG_RETURN_ON_FAIL(
                createProgram(specializedProgramDesc, specializedProgram.writeRef()));

            // Create specialized pipeline state.
            ComPtr<IPipelineState> specializedPipelineComPtr;
            switch (pipelineType)
            {
            case PipelineType::Compute:
                {
                    auto pipelineDesc = currentPipeline->desc.compute;
                    pipelineDesc.program = specializedProgram;
                    SLANG_RETURN_ON_FAIL(createComputePipelineState(
                        pipelineDesc,
                        specializedPipelineComPtr.writeRef()));
                    break;
                }
            case PipelineType::Graphics:
                {
                    auto pipelineDesc = currentPipeline->desc.graphics;
                    pipelineDesc.program =
                        static_cast<ShaderProgramBase*>(specializedProgram.get());
                    SLANG_RETURN_ON_FAIL(createGraphicsPipelineState(
                        pipelineDesc,
                        specializedPipelineComPtr.writeRef()));
                    break;
                }
            case PipelineType::RayTracing:
                {
                    auto pipelineDesc = currentPipeline->desc.rayTracing;
                    pipelineDesc.program =
                        static_cast<ShaderProgramBase*>(specializedProgram.get());
                    SLANG_RETURN_ON_FAIL(createRayTracingPipelineState(
                        pipelineDesc.get(),
                        specializedPipelineComPtr.writeRef()));
                    break;
                }
            default:
                break;
            }
            specializedPipelineState =
                static_cast<PipelineStateBase*>(specializedPipelineComPtr.get());
            specializedPipelineState->unspecializedPipelineState = currentPipeline;
            shaderCache.addSpecializedPipeline(pipelineKey, specializedPipelineState);
        }
        auto specializedPipelineStateBase =
            static_cast<PipelineStateBase*>(specializedPipelineState.Ptr());
        outNewPipeline = specializedPipelineStateBase;
    }
    return SLANG_OK;
}

IDebugCallback*& _getDebugCallback()
{
    static IDebugCallback* callback = nullptr;
    return callback;
}

class NullDebugCallback : public IDebugCallback
{
public:
    virtual SLANG_NO_THROW void SLANG_MCALL
    handleMessage(DebugMessageType type, DebugMessageSource source, const char* message) override
    {
        SLANG_UNUSED(type);
        SLANG_UNUSED(source);
        SLANG_UNUSED(message);
    }
};
IDebugCallback* _getNullDebugCallback()
{
    static NullDebugCallback result = {};
    return &result;
}

Result ShaderObjectBase::copyFrom(IShaderObject* object, ITransientResourceHeap* transientHeap)
{
    if (auto srcObj = dynamic_cast<MutableRootShaderObject*>(object))
    {
        setData(
            gfx::ShaderOffset(),
            srcObj->m_data.begin(),
            (size_t)srcObj->m_data.getCount()); // TODO: Change size_t to Count?
        for (auto& kv : srcObj->m_objects)
        {
            ComPtr<IShaderObject> subObject;
            SLANG_RETURN_ON_FAIL(kv.value->getCurrentVersion(transientHeap, subObject.writeRef()));
            setObject(kv.key, subObject);
        }
        for (auto& kv : srcObj->m_resources)
        {
            setResource(kv.key, kv.value.Ptr());
        }
        for (auto& kv : srcObj->m_samplers)
        {
            setSampler(kv.key, kv.value.Ptr());
        }
        for (auto& kv : srcObj->m_specializationArgs)
        {
            setSpecializationArgs(kv.key, kv.value.begin(), (uint32_t)kv.value.getCount());
        }
        return SLANG_OK;
    }
    return SLANG_FAIL;
}

Result ShaderTableBase::init(const IShaderTable::Desc& desc)
{
    m_rayGenShaderCount = desc.rayGenShaderCount;
    m_missShaderCount = desc.missShaderCount;
    m_hitGroupCount = desc.hitGroupCount;
    m_callableShaderCount = desc.callableShaderCount;
    m_shaderGroupNames.reserve(
        desc.hitGroupCount + desc.missShaderCount + desc.rayGenShaderCount +
        desc.callableShaderCount);
    m_recordOverwrites.reserve(
        desc.hitGroupCount + desc.missShaderCount + desc.rayGenShaderCount +
        desc.callableShaderCount);
    for (GfxIndex i = 0; i < desc.rayGenShaderCount; i++)
    {
        m_shaderGroupNames.add(desc.rayGenShaderEntryPointNames[i]);
        if (desc.rayGenShaderRecordOverwrites)
        {
            m_recordOverwrites.add(desc.rayGenShaderRecordOverwrites[i]);
        }
        else
        {
            m_recordOverwrites.add(ShaderRecordOverwrite{});
        }
    }
    for (GfxIndex i = 0; i < desc.missShaderCount; i++)
    {
        m_shaderGroupNames.add(desc.missShaderEntryPointNames[i]);
        if (desc.missShaderRecordOverwrites)
        {
            m_recordOverwrites.add(desc.missShaderRecordOverwrites[i]);
        }
        else
        {
            m_recordOverwrites.add(ShaderRecordOverwrite{});
        }
    }
    for (GfxIndex i = 0; i < desc.hitGroupCount; i++)
    {
        m_shaderGroupNames.add(desc.hitGroupNames[i]);
        if (desc.hitGroupRecordOverwrites)
        {
            m_recordOverwrites.add(desc.hitGroupRecordOverwrites[i]);
        }
        else
        {
            m_recordOverwrites.add(ShaderRecordOverwrite{});
        }
    }
    for (GfxIndex i = 0; i < desc.callableShaderCount; i++)
    {
        m_shaderGroupNames.add(desc.callableShaderEntryPointNames[i]);
        if (desc.callableShaderRecordOverwrites)
        {
            m_recordOverwrites.add(desc.callableShaderRecordOverwrites[i]);
        }
        else
        {
            m_recordOverwrites.add(ShaderRecordOverwrite{});
        }
    }
    return SLANG_OK;
}

bool isDepthFormat(Format format)
{
    switch (format)
    {
    case Format::D16_UNORM:
    case Format::D32_FLOAT:
    case Format::D32_FLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

bool isStencilFormat(Format format)
{
    switch (format)
    {
    case Format::D32_FLOAT_S8_UINT:
        return true;
    default:
        return false;
    }
}

} // namespace gfx
