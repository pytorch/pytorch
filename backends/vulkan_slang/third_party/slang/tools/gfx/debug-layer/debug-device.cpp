// debug-device.cpp
#include "debug-device.h"

#include "debug-buffer.h"
#include "debug-command-queue.h"
#include "debug-fence.h"
#include "debug-framebuffer.h"
#include "debug-helper-functions.h"
#include "debug-pipeline-state.h"
#include "debug-query.h"
#include "debug-render-pass.h"
#include "debug-resource-views.h"
#include "debug-sampler-state.h"
#include "debug-shader-object.h"
#include "debug-shader-program.h"
#include "debug-shader-table.h"
#include "debug-swap-chain.h"
#include "debug-texture.h"
#include "debug-transient-heap.h"
#include "debug-vertex-layout.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

SlangResult DebugDevice::queryInterface(SlangUUID const& uuid, void** outObject) noexcept
{
    void* intf = getInterface(uuid);
    if (intf)
    {
        addRef();
        *outObject = intf;
        return SLANG_OK;
    }

    // Fallback to trying to get the interface from the debugged object
    return baseObject->queryInterface(uuid, outObject);
}

Result DebugDevice::getNativeDeviceHandles(InteropHandles* outHandles)
{
    return baseObject->getNativeDeviceHandles(outHandles);
}

Result DebugDevice::getFeatures(
    const char** outFeatures,
    Size bufferSize,
    GfxCount* outFeatureCount)
{
    SLANG_GFX_API_FUNC;

    return baseObject->getFeatures(outFeatures, bufferSize, outFeatureCount);
}

Result DebugDevice::getFormatSupportedResourceStates(Format format, ResourceStateSet* outStates)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getFormatSupportedResourceStates(format, outStates);
}

DebugDevice::DebugDevice()
{
    SLANG_GFX_API_FUNC_NAME("CreateDevice");
    GFX_DIAGNOSE_INFO("Debug layer is enabled.");
}

SLANG_NO_THROW bool SLANG_MCALL DebugDevice::hasFeature(const char* feature)
{
    SLANG_GFX_API_FUNC;

    return baseObject->hasFeature(feature);
}

Result DebugDevice::getSlangSession(slang::ISession** outSlangSession)
{
    SLANG_GFX_API_FUNC;

    return baseObject->getSlangSession(outSlangSession);
}

Result DebugDevice::createTransientResourceHeap(
    const ITransientResourceHeap::Desc& desc,
    ITransientResourceHeap** outHeap)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugTransientResourceHeap> outObject = new DebugTransientResourceHeap();
    auto result = baseObject->createTransientResourceHeap(desc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outHeap, outObject);
    return result;
}

Result DebugDevice::createTextureResource(
    const ITextureResource::Desc& desc,
    const ITextureResource::SubresourceData* initData,
    ITextureResource** outResource)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugTextureResource> outObject = new DebugTextureResource();
    auto result =
        baseObject->createTextureResource(desc, initData, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outResource, outObject);
    return result;
}

Result DebugDevice::createTextureFromNativeHandle(
    InteropHandle handle,
    const ITextureResource::Desc& srcDesc,
    ITextureResource** outResource)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugTextureResource> outObject = new DebugTextureResource();
    auto result = baseObject->createTextureFromNativeHandle(
        handle,
        srcDesc,
        outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outResource, outObject);
    return result;
}

Result DebugDevice::createTextureFromSharedHandle(
    InteropHandle handle,
    const ITextureResource::Desc& srcDesc,
    const size_t size,
    ITextureResource** outResource)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugTextureResource> outObject = new DebugTextureResource();
    auto result = baseObject->createTextureFromSharedHandle(
        handle,
        srcDesc,
        size,
        outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outResource, outObject);
    return result;
}

Result DebugDevice::createBufferResource(
    const IBufferResource::Desc& desc,
    const void* initData,
    IBufferResource** outResource)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugBufferResource> outObject = new DebugBufferResource();
    auto result =
        baseObject->createBufferResource(desc, initData, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outResource, outObject);
    return result;
}

Result DebugDevice::createBufferFromNativeHandle(
    InteropHandle handle,
    const IBufferResource::Desc& srcDesc,
    IBufferResource** outResource)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugBufferResource> outObject = new DebugBufferResource();
    auto result =
        baseObject->createBufferFromNativeHandle(handle, srcDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outResource, outObject);
    return result;
}

Result DebugDevice::createBufferFromSharedHandle(
    InteropHandle handle,
    const IBufferResource::Desc& srcDesc,
    IBufferResource** outResource)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugBufferResource> outObject = new DebugBufferResource();
    auto result =
        baseObject->createBufferFromSharedHandle(handle, srcDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outResource, outObject);
    return result;
}

Result DebugDevice::createSamplerState(ISamplerState::Desc const& desc, ISamplerState** outSampler)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugSamplerState> outObject = new DebugSamplerState();
    auto result = baseObject->createSamplerState(desc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outSampler, outObject);
    return result;
}

Result DebugDevice::createTextureView(
    ITextureResource* texture,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugResourceView> outObject = new DebugResourceView();
    auto result =
        baseObject->createTextureView(getInnerObj(texture), desc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outView, outObject);
    return result;
}

Result DebugDevice::createBufferView(
    IBufferResource* buffer,
    IBufferResource* counterBuffer,
    IResourceView::Desc const& desc,
    IResourceView** outView)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugResourceView> outObject = new DebugResourceView();
    auto result = baseObject->createBufferView(
        getInnerObj(buffer),
        getInnerObj(counterBuffer),
        desc,
        outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outView, outObject);
    return result;
}

Result DebugDevice::getAccelerationStructurePrebuildInfo(
    const IAccelerationStructure::BuildInputs& buildInputs,
    IAccelerationStructure::PrebuildInfo* outPrebuildInfo)
{
    SLANG_GFX_API_FUNC;
    validateAccelerationStructureBuildInputs(buildInputs);
    return baseObject->getAccelerationStructurePrebuildInfo(buildInputs, outPrebuildInfo);
}

Result DebugDevice::createAccelerationStructure(
    const IAccelerationStructure::CreateDesc& desc,
    IAccelerationStructure** outAS)
{
    SLANG_GFX_API_FUNC;
    auto innerDesc = desc;
    innerDesc.buffer = getInnerObj(innerDesc.buffer);
    RefPtr<DebugAccelerationStructure> outObject = new DebugAccelerationStructure();
    auto result =
        baseObject->createAccelerationStructure(innerDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outAS, outObject);
    return SLANG_OK;
}

Result DebugDevice::createFramebufferLayout(
    IFramebufferLayout::Desc const& desc,
    IFramebufferLayout** outFrameBuffer)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugFramebufferLayout> outObject = new DebugFramebufferLayout();
    auto result = baseObject->createFramebufferLayout(desc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outFrameBuffer, outObject);
    return result;
}

Result DebugDevice::createFramebuffer(IFramebuffer::Desc const& desc, IFramebuffer** outFrameBuffer)
{
    SLANG_GFX_API_FUNC;

    auto innerDesc = desc;
    innerDesc.layout = getInnerObj(desc.layout);
    innerDesc.depthStencilView = getInnerObj(desc.depthStencilView);
    List<IResourceView*> innerRenderTargets;
    for (GfxIndex i = 0; i < desc.renderTargetCount; i++)
    {
        auto innerRenderTarget = getInnerObj(desc.renderTargetViews[i]);
        innerRenderTargets.add(innerRenderTarget);
    }
    innerDesc.renderTargetViews = innerRenderTargets.getBuffer();

    RefPtr<DebugFramebuffer> outObject = new DebugFramebuffer();
    auto result = baseObject->createFramebuffer(innerDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outFrameBuffer, outObject);
    return result;
}

Result DebugDevice::createRenderPassLayout(
    const IRenderPassLayout::Desc& desc,
    IRenderPassLayout** outRenderPassLayout)
{
    SLANG_GFX_API_FUNC;

    auto innerDesc = desc;
    innerDesc.framebufferLayout = getInnerObj(desc.framebufferLayout);
    RefPtr<DebugRenderPassLayout> outObject = new DebugRenderPassLayout();
    auto result = baseObject->createRenderPassLayout(innerDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outRenderPassLayout, outObject);
    return result;
}

Result DebugDevice::createSwapchain(
    ISwapchain::Desc const& desc,
    WindowHandle window,
    ISwapchain** outSwapchain)
{
    SLANG_GFX_API_FUNC;

    auto innerDesc = desc;
    innerDesc.queue = getInnerObj(desc.queue);
    RefPtr<DebugSwapchain> outObject = new DebugSwapchain();
    outObject->queue = getDebugObj(desc.queue);
    auto result = baseObject->createSwapchain(innerDesc, window, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outSwapchain, outObject);
    return Result();
}

Result DebugDevice::createInputLayout(IInputLayout::Desc const& desc, IInputLayout** outLayout)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugInputLayout> outObject = new DebugInputLayout();
    auto result = baseObject->createInputLayout(desc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outLayout, outObject);
    return result;
}

Result DebugDevice::createCommandQueue(const ICommandQueue::Desc& desc, ICommandQueue** outQueue)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugCommandQueue> outObject = new DebugCommandQueue();
    auto result = baseObject->createCommandQueue(desc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outQueue, outObject);
    return result;
}

Result DebugDevice::createShaderObject(
    slang::TypeReflection* type,
    ShaderObjectContainerType containerType,
    IShaderObject** outShaderObject)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugShaderObject> outObject = new DebugShaderObject();
    auto typeName = type->getName();
    auto result =
        baseObject->createShaderObject(type, containerType, outObject->baseObject.writeRef());
    outObject->m_typeName = typeName;
    outObject->m_device = this;
    outObject->m_slangType = type;
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outShaderObject, outObject);
    return result;
}

Result DebugDevice::createShaderObject2(
    slang::ISession* session,
    slang::TypeReflection* type,
    ShaderObjectContainerType containerType,
    IShaderObject** outShaderObject)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugShaderObject> outObject = new DebugShaderObject();
    auto typeName = type->getName();
    auto result = baseObject->createShaderObject2(
        session,
        type,
        containerType,
        outObject->baseObject.writeRef());
    outObject->m_typeName = typeName;
    outObject->m_device = this;
    outObject->m_slangType = type;
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outShaderObject, outObject);
    return result;
}

Result DebugDevice::createMutableShaderObject(
    slang::TypeReflection* type,
    ShaderObjectContainerType containerType,
    IShaderObject** outShaderObject)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugShaderObject> outObject = new DebugShaderObject();
    auto typeName = type->getName();
    auto result = baseObject->createMutableShaderObject(
        type,
        containerType,
        outObject->baseObject.writeRef());
    outObject->m_typeName = typeName;
    outObject->m_device = this;
    outObject->m_slangType = type;
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outShaderObject, outObject);
    return result;
}

Result DebugDevice::createMutableShaderObject2(
    slang::ISession* session,
    slang::TypeReflection* type,
    ShaderObjectContainerType containerType,
    IShaderObject** outShaderObject)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugShaderObject> outObject = new DebugShaderObject();
    auto typeName = type->getName();
    auto result = baseObject->createMutableShaderObject2(
        session,
        type,
        containerType,
        outObject->baseObject.writeRef());
    outObject->m_typeName = typeName;
    outObject->m_device = this;
    outObject->m_slangType = type;
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outShaderObject, outObject);
    return result;
}

Result DebugDevice::createMutableRootShaderObject(
    IShaderProgram* program,
    IShaderObject** outRootObject)
{
    SLANG_GFX_API_FUNC;
    RefPtr<DebugShaderObject> outObject = new DebugShaderObject();
    auto result = baseObject->createMutableRootShaderObject(
        getInnerObj(program),
        outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    outObject->m_device = this;
    outObject->m_slangType = nullptr;
    outObject->m_rootComponentType = getDebugObj(program)->m_slangProgram;
    returnComPtr(outRootObject, outObject);
    return result;
}

Result DebugDevice::createShaderObjectFromTypeLayout(
    slang::TypeLayoutReflection* typeLayout,
    IShaderObject** outShaderObject)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugShaderObject> outObject = new DebugShaderObject();
    auto result =
        baseObject->createShaderObjectFromTypeLayout(typeLayout, outObject->baseObject.writeRef());
    auto type = typeLayout->getType();
    auto typeName = type->getName();
    outObject->m_typeName = typeName;
    outObject->m_device = this;
    outObject->m_slangType = type;
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outShaderObject, outObject);
    return result;
}

Result DebugDevice::createMutableShaderObjectFromTypeLayout(
    slang::TypeLayoutReflection* typeLayout,
    IShaderObject** outShaderObject)
{
    SLANG_GFX_API_FUNC;
    RefPtr<DebugShaderObject> outObject = new DebugShaderObject();
    auto result = baseObject->createMutableShaderObjectFromTypeLayout(
        typeLayout,
        outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    auto type = typeLayout->getType();
    auto typeName = type->getName();
    outObject->m_typeName = typeName;
    outObject->m_device = this;
    outObject->m_slangType = type;
    returnComPtr(outShaderObject, outObject);
    return result;
}

Result DebugDevice::createProgram(
    const IShaderProgram::Desc& desc,
    IShaderProgram** outProgram,
    ISlangBlob** outDiagnostics)
{
    SLANG_GFX_API_FUNC;

    RefPtr<DebugShaderProgram> outObject = new DebugShaderProgram();
    auto result = baseObject->createProgram(desc, outObject->baseObject.writeRef(), outDiagnostics);
    if (SLANG_FAILED(result))
        return result;
    outObject->m_slangProgram = desc.slangGlobalScope;
    returnComPtr(outProgram, outObject);
    return result;
}

Result DebugDevice::createProgram2(
    const IShaderProgram::CreateDesc2& desc,
    IShaderProgram** outProgram,
    ISlangBlob** outDiagnostics)
{
    SLANG_GFX_API_FUNC;
    IShaderProgram::Desc desc1 = {};
    RefPtr<DebugShaderProgram> outObject = new DebugShaderProgram();
    auto result =
        baseObject->createProgram2(desc, outObject->baseObject.writeRef(), outDiagnostics);
    if (SLANG_FAILED(result))
        return result;
    auto base = static_cast<ShaderProgramBase*>(outObject->baseObject.get());
    outObject->m_slangProgram = base->desc.slangGlobalScope;
    returnComPtr(outProgram, outObject);
    return result;
}

Result DebugDevice::createGraphicsPipelineState(
    const GraphicsPipelineStateDesc& desc,
    IPipelineState** outState)
{
    SLANG_GFX_API_FUNC;

    GraphicsPipelineStateDesc innerDesc = desc;
    innerDesc.program = getInnerObj(desc.program);
    innerDesc.inputLayout = getInnerObj(desc.inputLayout);
    innerDesc.framebufferLayout = getInnerObj(desc.framebufferLayout);
    RefPtr<DebugPipelineState> outObject = new DebugPipelineState();
    auto result =
        baseObject->createGraphicsPipelineState(innerDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outState, outObject);
    return result;
}

Result DebugDevice::createComputePipelineState(
    const ComputePipelineStateDesc& desc,
    IPipelineState** outState)
{
    SLANG_GFX_API_FUNC;

    ComputePipelineStateDesc innerDesc = desc;
    innerDesc.program = getInnerObj(desc.program);

    RefPtr<DebugPipelineState> outObject = new DebugPipelineState();
    auto result =
        baseObject->createComputePipelineState(innerDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outState, outObject);
    return result;
}

Result DebugDevice::createRayTracingPipelineState(
    const RayTracingPipelineStateDesc& desc,
    IPipelineState** outState)
{
    SLANG_GFX_API_FUNC;

    RayTracingPipelineStateDesc innerDesc = desc;
    innerDesc.program = getInnerObj(desc.program);

    RefPtr<DebugPipelineState> outObject = new DebugPipelineState();
    auto result =
        baseObject->createRayTracingPipelineState(innerDesc, outObject->baseObject.writeRef());
    if (SLANG_FAILED(result))
        return result;
    returnComPtr(outState, outObject);
    return result;
}

SlangResult DebugDevice::readTextureResource(
    ITextureResource* resource,
    ResourceState state,
    ISlangBlob** outBlob,
    size_t* outRowPitch,
    size_t* outPixelSize)
{
    SLANG_GFX_API_FUNC;
    return baseObject
        ->readTextureResource(getInnerObj(resource), state, outBlob, outRowPitch, outPixelSize);
}

SlangResult DebugDevice::readBufferResource(
    IBufferResource* buffer,
    size_t offset,
    size_t size,
    ISlangBlob** outBlob)
{
    SLANG_GFX_API_FUNC;
    return baseObject->readBufferResource(getInnerObj(buffer), offset, size, outBlob);
}

const DeviceInfo& DebugDevice::getDeviceInfo() const
{
    SLANG_GFX_API_FUNC;
    return baseObject->getDeviceInfo();
}

Result DebugDevice::createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outPool)
{
    SLANG_GFX_API_FUNC;
    RefPtr<DebugQueryPool> result = new DebugQueryPool();
    result->desc = desc;
    SLANG_RETURN_ON_FAIL(baseObject->createQueryPool(desc, result->baseObject.writeRef()));
    returnComPtr(outPool, result);
    return SLANG_OK;
}

Result DebugDevice::createFence(const IFence::Desc& desc, IFence** outFence)
{
    SLANG_GFX_API_FUNC;
    RefPtr<DebugFence> result = new DebugFence();
    SLANG_RETURN_ON_FAIL(baseObject->createFence(desc, result->baseObject.writeRef()));
    returnComPtr(outFence, result);
    return SLANG_OK;
}

Result DebugDevice::waitForFences(
    GfxCount fenceCount,
    IFence** fences,
    uint64_t* values,
    bool waitForAll,
    uint64_t timeout)
{
    SLANG_GFX_API_FUNC;
    ShortList<IFence*> innerFences;
    for (GfxCount i = 0; i < fenceCount; i++)
    {
        innerFences.add(getInnerObj(fences[i]));
    }
    return baseObject->waitForFences(
        fenceCount,
        innerFences.getArrayView().getBuffer(),
        values,
        waitForAll,
        timeout);
}

Result DebugDevice::getTextureAllocationInfo(
    const ITextureResource::Desc& desc,
    size_t* outSize,
    size_t* outAlignment)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getTextureAllocationInfo(desc, outSize, outAlignment);
}

Result DebugDevice::getTextureRowAlignment(size_t* outAlignment)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getTextureRowAlignment(outAlignment);
}

Result DebugDevice::getCooperativeVectorProperties(
    CooperativeVectorProperties* properties,
    uint32_t* propertyCount)
{
    SLANG_GFX_API_FUNC;
    return baseObject->getCooperativeVectorProperties(properties, propertyCount);
}

Result DebugDevice::createShaderTable(const IShaderTable::Desc& desc, IShaderTable** outTable)
{
    SLANG_GFX_API_FUNC;
    RefPtr<DebugShaderTable> result = new DebugShaderTable();
    SLANG_RETURN_ON_FAIL(baseObject->createShaderTable(desc, result->baseObject.writeRef()));
    returnComPtr(outTable, result);
    return SLANG_OK;
}

} // namespace debug
} // namespace gfx
