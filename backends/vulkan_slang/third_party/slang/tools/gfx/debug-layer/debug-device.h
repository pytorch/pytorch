// debug-device.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugDevice : public DebugObject<IDevice>
{
public:
    SlangResult SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) noexcept override;
    SLANG_COM_OBJECT_IUNKNOWN_ADD_REF;
    SLANG_COM_OBJECT_IUNKNOWN_RELEASE;

public:
    DebugDevice();
    IDevice* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeDeviceHandles(InteropHandles* outHandles) override;
    virtual SLANG_NO_THROW bool SLANG_MCALL hasFeature(const char* feature) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getFeatures(const char** outFeatures, Size bufferSize, GfxCount* outFeatureCount) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getFormatSupportedResourceStates(Format format, ResourceStateSet* outStates) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getSlangSession(slang::ISession** outSlangSession) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTransientResourceHeap(
        const ITransientResourceHeap::Desc& desc,
        ITransientResourceHeap** outHeap) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureResource(
        const ITextureResource::Desc& desc,
        const ITextureResource::SubresourceData* initData,
        ITextureResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureFromNativeHandle(
        InteropHandle handle,
        const ITextureResource::Desc& srcDesc,
        ITextureResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureFromSharedHandle(
        InteropHandle handle,
        const ITextureResource::Desc& srcDesc,
        const Size size,
        ITextureResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferResource(
        const IBufferResource::Desc& desc,
        const void* initData,
        IBufferResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferFromNativeHandle(
        InteropHandle handle,
        const IBufferResource::Desc& srcDesc,
        IBufferResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferFromSharedHandle(
        InteropHandle handle,
        const IBufferResource::Desc& srcDesc,
        IBufferResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createSamplerState(ISamplerState::Desc const& desc, ISamplerState** outSampler) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureView(
        ITextureResource* texture,
        IResourceView::Desc const& desc,
        IResourceView** outView) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferView(
        IBufferResource* buffer,
        IBufferResource* counterBuffer,
        IResourceView::Desc const& desc,
        IResourceView** outView) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getAccelerationStructurePrebuildInfo(
        const IAccelerationStructure::BuildInputs& buildInputs,
        IAccelerationStructure::PrebuildInfo* outPrebuildInfo) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createAccelerationStructure(
        const IAccelerationStructure::CreateDesc& desc,
        IAccelerationStructure** outView) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createFramebufferLayout(
        IFramebufferLayout::Desc const& desc,
        IFramebufferLayout** outFrameBuffer) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createFramebuffer(IFramebuffer::Desc const& desc, IFramebuffer** outFrameBuffer) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createRenderPassLayout(
        const IRenderPassLayout::Desc& desc,
        IRenderPassLayout** outRenderPassLayout) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createSwapchain(
        ISwapchain::Desc const& desc,
        WindowHandle window,
        ISwapchain** outSwapchain) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createInputLayout(IInputLayout::Desc const& desc, IInputLayout** outLayout) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandQueue(const ICommandQueue::Desc& desc, ICommandQueue** outQueue) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createShaderObject(
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createShaderObject2(
        slang::ISession* session,
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createMutableShaderObject(
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createMutableShaderObject2(
        slang::ISession* session,
        slang::TypeReflection* type,
        ShaderObjectContainerType container,
        IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createShaderObjectFromTypeLayout(
        slang::TypeLayoutReflection* typeLayout,
        IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createMutableShaderObjectFromTypeLayout(
        slang::TypeLayoutReflection* typeLayout,
        IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createMutableRootShaderObject(IShaderProgram* program, IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createProgram(
        const IShaderProgram::Desc& desc,
        IShaderProgram** outProgram,
        ISlangBlob** outDiagnostics) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createProgram2(
        const IShaderProgram::CreateDesc2& desc,
        IShaderProgram** outProgram,
        ISlangBlob** outDiagnostics) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createGraphicsPipelineState(
        const GraphicsPipelineStateDesc& desc,
        IPipelineState** outState) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createComputePipelineState(
        const ComputePipelineStateDesc& desc,
        IPipelineState** outState) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createRayTracingPipelineState(
        const RayTracingPipelineStateDesc& desc,
        IPipelineState** outState) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL readTextureResource(
        ITextureResource* resource,
        ResourceState state,
        ISlangBlob** outBlob,
        Size* outRowPitch,
        Size* outPixelSize) override;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL readBufferResource(
        IBufferResource* buffer,
        Offset offset,
        Size size,
        ISlangBlob** outBlob) override;
    virtual SLANG_NO_THROW const DeviceInfo& SLANG_MCALL getDeviceInfo() const override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outPool) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createFence(const IFence::Desc& desc, IFence** outFence) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL waitForFences(
        GfxCount fenceCount,
        IFence** fences,
        uint64_t* values,
        bool waitForAll,
        uint64_t timeout) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getTextureAllocationInfo(
        const ITextureResource::Desc& desc,
        size_t* outSize,
        size_t* outAlignment) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getTextureRowAlignment(size_t* outAlignment) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getCooperativeVectorProperties(
        CooperativeVectorProperties* properties,
        uint32_t* propertyCount) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createShaderTable(const IShaderTable::Desc& desc, IShaderTable** outTable) override;
};

} // namespace debug
} // namespace gfx
