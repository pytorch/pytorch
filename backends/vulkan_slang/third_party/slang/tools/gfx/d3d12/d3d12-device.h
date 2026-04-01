// d3d12-device.h
#pragma once
#include "d3d12-command-buffer.h"
#include "d3d12-command-queue.h"
#include "d3d12-texture.h"
#include "d3d12-transient-heap.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>

namespace gfx
{
namespace d3d12
{

using namespace Slang;

// Define function pointer types for PIX library.
typedef HRESULT(WINAPI* PFN_BeginEventOnCommandList)(
    ID3D12GraphicsCommandList* commandList,
    UINT64 color,
    PCSTR formatString);
typedef HRESULT(WINAPI* PFN_EndEventOnCommandList)(ID3D12GraphicsCommandList* commandList);

struct D3D12DeviceInfo
{
    void clear()
    {
        m_dxgiFactory.setNull();
        m_device.setNull();
        m_adapter.setNull();
        m_desc = {};
        m_desc1 = {};
        m_isWarp = false;
        m_isSoftware = false;
    }

    bool m_isWarp;
    bool m_isSoftware;
    ComPtr<IDXGIFactory> m_dxgiFactory;
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12Device5> m_device5;
    ComPtr<IDXGIAdapter> m_adapter;
    DXGI_ADAPTER_DESC m_desc;
    DXGI_ADAPTER_DESC1 m_desc1;
};

class DeviceImpl : public RendererBase
{
public:
    Desc m_desc;
    D3D12DeviceExtendedDesc m_extendedDesc;

    DeviceInfo m_info;
    String m_adapterName;

    bool m_isInitialized = false;

    ComPtr<ID3D12Debug> m_dxDebug;

    static const bool g_isAftermathEnabled;

    D3D12DeviceInfo m_deviceInfo;
    ID3D12Device* m_device = nullptr;
    ID3D12Device5* m_device5 = nullptr;

    VirtualObjectPool m_queueIndexAllocator;

    RefPtr<CommandQueueImpl> m_resourceCommandQueue;
    RefPtr<TransientResourceHeapImpl> m_resourceCommandTransientHeap;

    RefPtr<D3D12GeneralExpandingDescriptorHeap> m_rtvAllocator;
    RefPtr<D3D12GeneralExpandingDescriptorHeap> m_dsvAllocator;

    // Space in the GPU-visible heaps is precious, so we will also keep
    // around CPU-visible heaps for storing shader-objects' descriptors in a format
    // that is ready for copying into the GPU-visible heaps as needed.
    //
    RefPtr<D3D12GeneralExpandingDescriptorHeap> m_cpuViewHeap;    ///< Cbv, Srv, Uav
    RefPtr<D3D12GeneralExpandingDescriptorHeap> m_cpuSamplerHeap; ///< Heap for samplers

    // Dll entry points
    PFN_D3D12_GET_DEBUG_INTERFACE m_D3D12GetDebugInterface = nullptr;
    PFN_D3D12_CREATE_DEVICE m_D3D12CreateDevice = nullptr;
    PFN_D3D12_SERIALIZE_ROOT_SIGNATURE m_D3D12SerializeRootSignature = nullptr;
    PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE m_D3D12SerializeVersionedRootSignature = nullptr;
    PFN_BeginEventOnCommandList m_BeginEventOnCommandList = nullptr;
    PFN_EndEventOnCommandList m_EndEventOnCommandList = nullptr;

    bool m_nvapi = false;

    // Command signatures required for indirect draws. These indicate the format of the indirect
    // as well as the command type to be used (DrawInstanced and DrawIndexedInstanced, in this
    // case).
    ComPtr<ID3D12CommandSignature> drawIndirectCmdSignature;
    ComPtr<ID3D12CommandSignature> drawIndexedIndirectCmdSignature;
    ComPtr<ID3D12CommandSignature> dispatchIndirectCmdSignature;

public:
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL initialize(const Desc& desc) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getFormatSupportedResourceStates(Format format, ResourceStateSet* outStates) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createCommandQueue(const ICommandQueue::Desc& desc, ICommandQueue** outQueue) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTransientResourceHeap(
        const ITransientResourceHeap::Desc& desc,
        ITransientResourceHeap** outHeap) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createSwapchain(
        const ISwapchain::Desc& desc,
        WindowHandle window,
        ISwapchain** outSwapchain) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL getTextureAllocationInfo(
        const ITextureResource::Desc& desc,
        Size* outSize,
        Size* outAlignment) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getTextureRowAlignment(Size* outAlignment) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureResource(
        const ITextureResource::Desc& desc,
        const ITextureResource::SubresourceData* initData,
        ITextureResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createTextureFromNativeHandle(
        InteropHandle handle,
        const ITextureResource::Desc& srcDesc,
        ITextureResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferResource(
        const IBufferResource::Desc& desc,
        const void* initData,
        IBufferResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createBufferFromNativeHandle(
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

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createFramebuffer(IFramebuffer::Desc const& desc, IFramebuffer** outFrameBuffer) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createFramebufferLayout(
        IFramebufferLayout::Desc const& desc,
        IFramebufferLayout** outLayout) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL createRenderPassLayout(
        const IRenderPassLayout::Desc& desc,
        IRenderPassLayout** outRenderPassLayout) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createInputLayout(IInputLayout::Desc const& desc, IInputLayout** outLayout) override;

    virtual Result createShaderObjectLayout(
        slang::ISession* session,
        slang::TypeLayoutReflection* typeLayout,
        ShaderObjectLayoutBase** outLayout) override;
    virtual Result createShaderObject(ShaderObjectLayoutBase* layout, IShaderObject** outObject)
        override;
    virtual Result createMutableShaderObject(
        ShaderObjectLayoutBase* layout,
        IShaderObject** outObject) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    createMutableRootShaderObject(IShaderProgram* program, IShaderObject** outObject) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createShaderTable(const IShaderTable::Desc& desc, IShaderTable** outShaderTable) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createProgram(
        const IShaderProgram::Desc& desc,
        IShaderProgram** outProgram,
        ISlangBlob** outDiagnostics) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createGraphicsPipelineState(
        const GraphicsPipelineStateDesc& desc,
        IPipelineState** outState) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createComputePipelineState(
        const ComputePipelineStateDesc& desc,
        IPipelineState** outState) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createQueryPool(const IQueryPool::Desc& desc, IQueryPool** outState) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    createFence(const IFence::Desc& desc, IFence** outFence) override;

    virtual SLANG_NO_THROW Result SLANG_MCALL waitForFences(
        GfxCount fenceCount,
        IFence** fences,
        uint64_t* fenceValues,
        bool waitForAll,
        uint64_t timeout) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL readTextureResource(
        ITextureResource* resource,
        ResourceState state,
        ISlangBlob** outBlob,
        Size* outRowPitch,
        Size* outPixelSize) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL readBufferResource(
        IBufferResource* resource,
        Offset offset,
        Size size,
        ISlangBlob** outBlob) override;

    virtual SLANG_NO_THROW const gfx::DeviceInfo& SLANG_MCALL getDeviceInfo() const override;

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeDeviceHandles(InteropHandles* outHandles) override;

    ~DeviceImpl();

    virtual SLANG_NO_THROW Result SLANG_MCALL getAccelerationStructurePrebuildInfo(
        const IAccelerationStructure::BuildInputs& buildInputs,
        IAccelerationStructure::PrebuildInfo* outPrebuildInfo) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createAccelerationStructure(
        const IAccelerationStructure::CreateDesc& desc,
        IAccelerationStructure** outView) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL createRayTracingPipelineState(
        const RayTracingPipelineStateDesc& desc,
        IPipelineState** outState) override;

public:
    static void* loadProc(SharedLibrary::Handle module, char const* name);

    Result createCommandQueueImpl(CommandQueueImpl** outQueue);

    Result createTransientResourceHeapImpl(
        ITransientResourceHeap::Flags::Enum flags,
        Size constantBufferSize,
        uint32_t viewDescriptors,
        uint32_t samplerDescriptors,
        TransientResourceHeapImpl** outHeap);

    Result createBuffer(
        const D3D12_RESOURCE_DESC& resourceDesc,
        const void* srcData,
        Size srcDataSize,
        D3D12_RESOURCE_STATES finalState,
        D3D12Resource& resourceOut,
        bool isShared,
        MemoryType access = MemoryType::DeviceLocal);

    Result captureTextureToSurface(
        TextureResourceImpl* resource,
        ResourceState state,
        ISlangBlob** blob,
        Size* outRowPitch,
        Size* outPixelSize);

    Result _createDevice(
        DeviceCheckFlags deviceCheckFlags,
        const AdapterLUID* adapterLUID,
        D3D_FEATURE_LEVEL featureLevel,
        D3D12DeviceInfo& outDeviceInfo);

    struct ResourceCommandRecordInfo
    {
        ComPtr<ICommandBuffer> commandBuffer;
        ID3D12GraphicsCommandList* d3dCommandList;
    };
    ResourceCommandRecordInfo encodeResourceCommands();
    void submitResourceCommandsAndWait(const ResourceCommandRecordInfo& info);

private:
    void processExperimentalFeaturesDesc(SharedLibrary::Handle d3dModule, void* desc);
};

} // namespace d3d12
} // namespace gfx
