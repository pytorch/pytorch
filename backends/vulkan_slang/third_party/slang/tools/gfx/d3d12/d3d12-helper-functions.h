// d3d12-helper-functions.h
#pragma once

#include "../../../source/core/slang-list.h"
#include "../../../source/core/slang-short-list.h"
#include "d3d12-base.h"
#include "d3d12-shader-object-layout.h"
#include "d3d12-submitter.h"
#include "slang-gfx.h"

#ifndef __ID3D12GraphicsCommandList1_FWD_DEFINED__
// If can't find a definition of CommandList1, just use an empty definition
struct ID3D12GraphicsCommandList1
{
};
#endif

namespace gfx
{

using namespace Slang;

namespace d3d12
{
struct PendingDescriptorTableBinding
{
    uint32_t rootIndex;
    D3D12_GPU_DESCRIPTOR_HANDLE handle;
};

/// Contextual data and operations required when binding shader objects to the pipeline state
struct BindingContext
{
    PipelineCommandEncoder* encoder;
    Submitter* submitter;
    TransientResourceHeapImpl* transientHeap;
    DeviceImpl* device;
    D3D12_DESCRIPTOR_HEAP_TYPE
    outOfMemoryHeap; // The type of descriptor heap that is OOM during binding.
    ShortList<PendingDescriptorTableBinding>* pendingTableBindings;
};

bool isSupportedNVAPIOp(ID3D12Device* dev, uint32_t op);

D3D12_RESOURCE_FLAGS calcResourceFlag(ResourceState state);
D3D12_RESOURCE_FLAGS calcResourceFlags(ResourceStateSet states);
D3D12_RESOURCE_DIMENSION calcResourceDimension(IResource::Type type);

DXGI_FORMAT getTypelessFormatFromDepthFormat(Format format);
bool isTypelessDepthFormat(DXGI_FORMAT format);

D3D12_FILTER_TYPE translateFilterMode(TextureFilteringMode mode);
D3D12_FILTER_REDUCTION_TYPE translateFilterReduction(TextureReductionOp op);
D3D12_TEXTURE_ADDRESS_MODE translateAddressingMode(TextureAddressingMode mode);
D3D12_COMPARISON_FUNC translateComparisonFunc(ComparisonFunc func);

uint32_t getViewDescriptorCount(const ITransientResourceHeap::Desc& desc);
void initSrvDesc(
    IResource::Type resourceType,
    const ITextureResource::Desc& textureDesc,
    const D3D12_RESOURCE_DESC& desc,
    DXGI_FORMAT pixelFormat,
    SubresourceRange subresourceRange,
    D3D12_SHADER_RESOURCE_VIEW_DESC& descOut);
Result initTextureResourceDesc(
    D3D12_RESOURCE_DESC& resourceDesc,
    const ITextureResource::Desc& srcDesc);
void initBufferResourceDesc(Size bufferSize, D3D12_RESOURCE_DESC& out);
Result uploadBufferDataImpl(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* cmdList,
    TransientResourceHeapImpl* transientHeap,
    BufferResourceImpl* buffer,
    Offset offset,
    Size size,
    void* data);

Result createNullDescriptor(
    ID3D12Device* d3dDevice,
    D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor,
    const ShaderObjectLayoutImpl::BindingRangeInfo& bindingRange);

void translatePostBuildInfoDescs(
    int propertyQueryCount,
    AccelerationStructureQueryDesc* queryDescs,
    List<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC>& postBuildInfoDescs);

} // namespace d3d12

Result SLANG_MCALL getD3D12Adapters(List<AdapterInfo>& outAdapters);

Result SLANG_MCALL createD3D12Device(const IDevice::Desc* desc, IDevice** outDevice);

} // namespace gfx
