// d3d12-helper-functions.cpp
#include "d3d12-helper-functions.h"

#ifdef GFX_NVAPI
#include "../nvapi/nvapi-include.h"
#endif

#include "../nvapi/nvapi-util.h"
#include "d3d12-buffer.h"
#include "d3d12-query.h"
#include "d3d12-transient-heap.h"

#ifdef _DEBUG
#define ENABLE_DEBUG_LAYER 1
#else
#define ENABLE_DEBUG_LAYER 0
#endif

namespace gfx
{

using namespace Slang;

namespace d3d12
{

bool isSupportedNVAPIOp(ID3D12Device* dev, uint32_t op)
{
#ifdef GFX_NVAPI
    {
        bool isSupported;
        NvAPI_Status status =
            NvAPI_D3D12_IsNvShaderExtnOpCodeSupported(dev, NvU32(op), &isSupported);
        return status == NVAPI_OK && isSupported;
    }
#else
    return false;
#endif
}

D3D12_RESOURCE_FLAGS calcResourceFlag(ResourceState state)
{
    switch (state)
    {
    case ResourceState::RenderTarget:
        return D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    case ResourceState::DepthRead:
    case ResourceState::DepthWrite:
        return D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    case ResourceState::UnorderedAccess:
    case ResourceState::AccelerationStructure:
        return D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    default:
        return D3D12_RESOURCE_FLAG_NONE;
    }
}

D3D12_RESOURCE_FLAGS calcResourceFlags(ResourceStateSet states)
{
    int dstFlags = 0;
    for (uint32_t i = 0; i < (uint32_t)ResourceState::_Count; i++)
    {
        auto state = (ResourceState)i;
        if (states.contains(state))
            dstFlags |= calcResourceFlag(state);
    }
    return (D3D12_RESOURCE_FLAGS)dstFlags;
}

D3D12_RESOURCE_DIMENSION calcResourceDimension(IResource::Type type)
{
    switch (type)
    {
    case IResource::Type::Buffer:
        return D3D12_RESOURCE_DIMENSION_BUFFER;
    case IResource::Type::Texture1D:
        return D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    case IResource::Type::TextureCube:
    case IResource::Type::Texture2D:
        {
            return D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        }
    case IResource::Type::Texture3D:
        return D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    default:
        return D3D12_RESOURCE_DIMENSION_UNKNOWN;
    }
}

DXGI_FORMAT getTypelessFormatFromDepthFormat(Format format)
{
    switch (format)
    {
    case Format::D16_UNORM:
        return DXGI_FORMAT_R16_TYPELESS;
    case Format::D32_FLOAT:
        return DXGI_FORMAT_R32_TYPELESS;
    case Format::D32_FLOAT_S8_UINT:
        return DXGI_FORMAT_R32G8X24_TYPELESS;
    // case Format::D24_UNORM_S8_UINT:
    //     return DXGI_FORMAT_R24G8_TYPELESS;
    default:
        return D3DUtil::getMapFormat(format);
    }
}

bool isTypelessDepthFormat(DXGI_FORMAT format)
{
    switch (format)
    {
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_R24G8_TYPELESS:
        return true;
    default:
        return false;
    }
}

D3D12_FILTER_TYPE translateFilterMode(TextureFilteringMode mode)
{
    switch (mode)
    {
    default:
        return D3D12_FILTER_TYPE(0);

#define CASE(SRC, DST)              \
    case TextureFilteringMode::SRC: \
        return D3D12_FILTER_TYPE_##DST

        CASE(Point, POINT);
        CASE(Linear, LINEAR);

#undef CASE
    }
}

D3D12_FILTER_REDUCTION_TYPE translateFilterReduction(TextureReductionOp op)
{
    switch (op)
    {
    default:
        return D3D12_FILTER_REDUCTION_TYPE(0);

#define CASE(SRC, DST)            \
    case TextureReductionOp::SRC: \
        return D3D12_FILTER_REDUCTION_TYPE_##DST

        CASE(Average, STANDARD);
        CASE(Comparison, COMPARISON);
        CASE(Minimum, MINIMUM);
        CASE(Maximum, MAXIMUM);

#undef CASE
    }
}

D3D12_TEXTURE_ADDRESS_MODE translateAddressingMode(TextureAddressingMode mode)
{
    switch (mode)
    {
    default:
        return D3D12_TEXTURE_ADDRESS_MODE(0);

#define CASE(SRC, DST)               \
    case TextureAddressingMode::SRC: \
        return D3D12_TEXTURE_ADDRESS_MODE_##DST

        CASE(Wrap, WRAP);
        CASE(ClampToEdge, CLAMP);
        CASE(ClampToBorder, BORDER);
        CASE(MirrorRepeat, MIRROR);
        CASE(MirrorOnce, MIRROR_ONCE);

#undef CASE
    }
}

D3D12_COMPARISON_FUNC translateComparisonFunc(ComparisonFunc func)
{
    switch (func)
    {
    default:
        // TODO: need to report failures
        return D3D12_COMPARISON_FUNC_ALWAYS;

#define CASE(FROM, TO)         \
    case ComparisonFunc::FROM: \
        return D3D12_COMPARISON_FUNC_##TO

        CASE(Never, NEVER);
        CASE(Less, LESS);
        CASE(Equal, EQUAL);
        CASE(LessEqual, LESS_EQUAL);
        CASE(Greater, GREATER);
        CASE(NotEqual, NOT_EQUAL);
        CASE(GreaterEqual, GREATER_EQUAL);
        CASE(Always, ALWAYS);
#undef CASE
    }
}

uint32_t getViewDescriptorCount(const ITransientResourceHeap::Desc& desc)
{
    return Math::Max(
        Math::Max(
            desc.srvDescriptorCount,
            desc.uavDescriptorCount,
            desc.accelerationStructureDescriptorCount),
        desc.constantBufferDescriptorCount,
        2048);
}

void initSrvDesc(
    IResource::Type resourceType,
    const ITextureResource::Desc& textureDesc,
    const D3D12_RESOURCE_DESC& desc,
    DXGI_FORMAT pixelFormat,
    SubresourceRange subresourceRange,
    D3D12_SHADER_RESOURCE_VIEW_DESC& descOut)
{
    // create SRV
    descOut = D3D12_SHADER_RESOURCE_VIEW_DESC();

    descOut.Format = (pixelFormat == DXGI_FORMAT_UNKNOWN)
                         ? D3DUtil::calcFormat(D3DUtil::USAGE_SRV, desc.Format)
                         : pixelFormat;
    descOut.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (desc.DepthOrArraySize == 1)
    {
        switch (desc.Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
            descOut.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            descOut.Texture1D.MipLevels = subresourceRange.mipLevelCount == 0
                                              ? desc.MipLevels - subresourceRange.mipLevel
                                              : subresourceRange.mipLevelCount;
            descOut.Texture1D.MostDetailedMip = subresourceRange.mipLevel;
            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
            descOut.ViewDimension = textureDesc.sampleDesc.numSamples > 1
                                        ? D3D12_SRV_DIMENSION_TEXTURE2DMS
                                        : D3D12_SRV_DIMENSION_TEXTURE2D;
            descOut.Texture2D.PlaneSlice =
                D3DUtil::getPlaneSlice(descOut.Format, subresourceRange.aspectMask);
            descOut.Texture2D.ResourceMinLODClamp = 0.0f;
            descOut.Texture2D.MipLevels = subresourceRange.mipLevelCount == 0
                                              ? desc.MipLevels - subresourceRange.mipLevel
                                              : subresourceRange.mipLevelCount;
            descOut.Texture2D.MostDetailedMip = subresourceRange.mipLevel;
            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
            descOut.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            descOut.Texture3D.MipLevels = subresourceRange.mipLevelCount == 0
                                              ? desc.MipLevels - subresourceRange.mipLevel
                                              : subresourceRange.mipLevelCount;
            descOut.Texture3D.MostDetailedMip = subresourceRange.mipLevel;
            break;
        default:
            assert(!"Unknown dimension");
        }
    }
    else if (resourceType == IResource::Type::TextureCube)
    {
        if (textureDesc.arraySize > 1)
        {
            descOut.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;

            descOut.TextureCubeArray.NumCubes = subresourceRange.layerCount == 0
                                                    ? textureDesc.arraySize
                                                    : subresourceRange.layerCount / 6;
            descOut.TextureCubeArray.First2DArrayFace = subresourceRange.baseArrayLayer;
            descOut.TextureCubeArray.MipLevels = subresourceRange.mipLevelCount == 0
                                                     ? desc.MipLevels - subresourceRange.mipLevel
                                                     : subresourceRange.mipLevelCount;
            descOut.TextureCubeArray.MostDetailedMip = subresourceRange.mipLevel;
            descOut.TextureCubeArray.ResourceMinLODClamp = 0;
        }
        else
        {
            descOut.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;

            descOut.TextureCube.MipLevels = subresourceRange.mipLevelCount == 0
                                                ? desc.MipLevels - subresourceRange.mipLevel
                                                : subresourceRange.mipLevelCount;
            descOut.TextureCube.MostDetailedMip = subresourceRange.mipLevel;
            descOut.TextureCube.ResourceMinLODClamp = 0;
        }
    }
    else
    {
        assert(desc.DepthOrArraySize > 1);

        switch (desc.Dimension)
        {
        case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
            descOut.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            descOut.Texture1D.MostDetailedMip = subresourceRange.mipLevel;
            descOut.Texture1D.MipLevels = subresourceRange.mipLevelCount == 0
                                              ? desc.MipLevels
                                              : subresourceRange.mipLevelCount;
            descOut.Texture1DArray.ArraySize = subresourceRange.layerCount == 0
                                                   ? desc.DepthOrArraySize
                                                   : subresourceRange.layerCount;
            descOut.Texture1DArray.FirstArraySlice = subresourceRange.baseArrayLayer;
            descOut.Texture1DArray.ResourceMinLODClamp = 0;
            descOut.Texture1DArray.MostDetailedMip = subresourceRange.mipLevel;
            descOut.Texture1DArray.MipLevels = subresourceRange.mipLevelCount == 0
                                                   ? desc.MipLevels - subresourceRange.mipLevel
                                                   : subresourceRange.mipLevelCount;
            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
            descOut.ViewDimension = textureDesc.sampleDesc.numSamples > 1
                                        ? D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY
                                        : D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            if (descOut.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY)
            {
                descOut.Texture2DArray.ArraySize = subresourceRange.layerCount == 0
                                                       ? desc.DepthOrArraySize
                                                       : subresourceRange.layerCount;
                descOut.Texture2DArray.FirstArraySlice = subresourceRange.baseArrayLayer;
                descOut.Texture2DArray.PlaneSlice =
                    D3DUtil::getPlaneSlice(descOut.Format, subresourceRange.aspectMask);
                descOut.Texture2DArray.ResourceMinLODClamp = 0;
                descOut.Texture2DArray.MostDetailedMip = subresourceRange.mipLevel;
                descOut.Texture2DArray.MipLevels = subresourceRange.mipLevelCount == 0
                                                       ? desc.MipLevels - subresourceRange.mipLevel
                                                       : subresourceRange.mipLevelCount;
            }
            else
            {
                assert(descOut.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY);
                descOut.Texture2DMSArray.FirstArraySlice = subresourceRange.baseArrayLayer;
                descOut.Texture2DMSArray.ArraySize = subresourceRange.layerCount == 0
                                                         ? desc.DepthOrArraySize
                                                         : subresourceRange.layerCount;
            }

            break;
        case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
            descOut.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            descOut.Texture3D.MostDetailedMip = subresourceRange.mipLevel;
            descOut.Texture3D.MipLevels = subresourceRange.mipLevelCount == 0
                                              ? desc.MipLevels
                                              : subresourceRange.mipLevelCount;
            break;

        default:
            assert(!"Unknown dimension");
        }
    }
}

Result initTextureResourceDesc(
    D3D12_RESOURCE_DESC& resourceDesc,
    const ITextureResource::Desc& srcDesc)
{
    const DXGI_FORMAT pixelFormat = D3DUtil::getMapFormat(srcDesc.format);
    if (pixelFormat == DXGI_FORMAT_UNKNOWN)
    {
        return SLANG_FAIL;
    }

    const int arraySize = calcEffectiveArraySize(srcDesc);

    const D3D12_RESOURCE_DIMENSION dimension = calcResourceDimension(srcDesc.type);
    if (dimension == D3D12_RESOURCE_DIMENSION_UNKNOWN)
    {
        return SLANG_FAIL;
    }

    const int numMipMaps = srcDesc.numMipLevels;
    resourceDesc.Dimension = dimension;
    resourceDesc.Format = pixelFormat;
    resourceDesc.Width = srcDesc.size.width;
    resourceDesc.Height = srcDesc.size.height;
    resourceDesc.DepthOrArraySize = (srcDesc.size.depth > 1) ? srcDesc.size.depth : arraySize;

    resourceDesc.MipLevels = numMipMaps;
    resourceDesc.SampleDesc.Count = srcDesc.sampleDesc.numSamples;
    resourceDesc.SampleDesc.Quality = srcDesc.sampleDesc.quality;

    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;

    resourceDesc.Flags |= calcResourceFlags(srcDesc.allowedStates);

    resourceDesc.Alignment = 0;

    if (isDepthFormat(srcDesc.format) &&
        (srcDesc.allowedStates.contains(ResourceState::ShaderResource) ||
         srcDesc.allowedStates.contains(ResourceState::UnorderedAccess)))
    {
        resourceDesc.Format = getTypelessFormatFromDepthFormat(srcDesc.format);
    }

    return SLANG_OK;
}

void initBufferResourceDesc(Size bufferSize, D3D12_RESOURCE_DESC& out)
{
    out = {};

    out.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    out.Alignment = 0;
    out.Width = bufferSize;
    out.Height = 1;
    out.DepthOrArraySize = 1;
    out.MipLevels = 1;
    out.Format = DXGI_FORMAT_UNKNOWN;
    out.SampleDesc.Count = 1;
    out.SampleDesc.Quality = 0;
    out.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    out.Flags = D3D12_RESOURCE_FLAG_NONE;
}

Result uploadBufferDataImpl(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* cmdList,
    TransientResourceHeapImpl* transientHeap,
    BufferResourceImpl* buffer,
    Offset offset,
    Size size,
    void* data)
{
    IBufferResource* uploadResource;
    Offset uploadResourceOffset = 0;
    if (buffer->getDesc()->memoryType != MemoryType::Upload)
    {
        SLANG_RETURN_ON_FAIL(transientHeap->allocateStagingBuffer(
            size,
            uploadResource,
            uploadResourceOffset,
            MemoryType::Upload));
    }
    else
    {
        uploadResourceOffset = offset;
    }
    D3D12Resource& uploadResourceRef =
        (buffer->getDesc()->memoryType == MemoryType::Upload)
            ? buffer->m_resource
            : static_cast<BufferResourceImpl*>(uploadResource)->m_resource;

    D3D12_RANGE readRange = {};
    readRange.Begin = 0;
    readRange.End = 0;
    void* uploadData;
    SLANG_RETURN_ON_FAIL(
        uploadResourceRef.getResource()->Map(0, &readRange, reinterpret_cast<void**>(&uploadData)));
    memcpy((uint8_t*)uploadData + uploadResourceOffset, data, size);
    D3D12_RANGE writtenRange = {};
    writtenRange.Begin = uploadResourceOffset;
    writtenRange.End = uploadResourceOffset + size;
    uploadResourceRef.getResource()->Unmap(0, &writtenRange);

    if (buffer->getDesc()->memoryType != MemoryType::Upload)
    {
        cmdList->CopyBufferRegion(
            buffer->m_resource.getResource(),
            offset,
            uploadResourceRef.getResource(),
            uploadResourceOffset,
            size);
    }

    return SLANG_OK;
}

Result createNullDescriptor(
    ID3D12Device* d3dDevice,
    D3D12_CPU_DESCRIPTOR_HANDLE destDescriptor,
    const ShaderObjectLayoutImpl::BindingRangeInfo& bindingRange)
{
    switch (bindingRange.bindingType)
    {
    case slang::BindingType::ConstantBuffer:
        {
            D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc = {};
            cbvDesc.BufferLocation = 0;
            cbvDesc.SizeInBytes = 0;
            d3dDevice->CreateConstantBufferView(&cbvDesc, destDescriptor);
        }
        break;
    case slang::BindingType::MutableRawBuffer:
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            d3dDevice->CreateUnorderedAccessView(nullptr, nullptr, &uavDesc, destDescriptor);
        }
        break;
    case slang::BindingType::MutableTypedBuffer:
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            d3dDevice->CreateUnorderedAccessView(nullptr, nullptr, &uavDesc, destDescriptor);
        }
        break;
    case slang::BindingType::RawBuffer:
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d3dDevice->CreateShaderResourceView(nullptr, &srvDesc, destDescriptor);
        }
        break;
    case slang::BindingType::TypedBuffer:
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d3dDevice->CreateShaderResourceView(nullptr, &srvDesc, destDescriptor);
        }
        break;
    case slang::BindingType::Texture:
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            switch (bindingRange.resourceShape)
            {
            case SLANG_TEXTURE_1D:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
                break;
            case SLANG_TEXTURE_1D_ARRAY:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                break;
            case SLANG_TEXTURE_2D:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                break;
            case SLANG_TEXTURE_2D_ARRAY:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                break;
            case SLANG_TEXTURE_3D:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                break;
            case SLANG_TEXTURE_CUBE:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                break;
            case SLANG_TEXTURE_CUBE_ARRAY:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                break;
            case SLANG_TEXTURE_2D_MULTISAMPLE:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
                break;
            case SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY:
                srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                break;
            default:
                return SLANG_OK;
            }
            d3dDevice->CreateShaderResourceView(nullptr, &srvDesc, destDescriptor);
        }
        break;
    case slang::BindingType::MutableTexture:
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            switch (bindingRange.resourceShape)
            {
            case SLANG_TEXTURE_1D:
                uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
                break;
            case SLANG_TEXTURE_1D_ARRAY:
                uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                break;
            case SLANG_TEXTURE_2D:
                uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                break;
            case SLANG_TEXTURE_2D_ARRAY:
                uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                break;
            case SLANG_TEXTURE_3D:
                uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                break;
            case SLANG_TEXTURE_CUBE:
            case SLANG_TEXTURE_CUBE_ARRAY:
            case SLANG_TEXTURE_2D_MULTISAMPLE:
            case SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY:
            default:
                return SLANG_OK;
            }
            d3dDevice->CreateUnorderedAccessView(nullptr, nullptr, &uavDesc, destDescriptor);
        }
        break;
    default:
        break;
    }
    return SLANG_OK;
}

void translatePostBuildInfoDescs(
    int propertyQueryCount,
    AccelerationStructureQueryDesc* queryDescs,
    List<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC>& postBuildInfoDescs)
{
    postBuildInfoDescs.setCount(propertyQueryCount);
    for (int i = 0; i < propertyQueryCount; i++)
    {
        switch (queryDescs[i].queryType)
        {
        case QueryType::AccelerationStructureCompactedSize:
            postBuildInfoDescs[i].InfoType =
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
            postBuildInfoDescs[i].DestBuffer =
                static_cast<PlainBufferProxyQueryPoolImpl*>(queryDescs[i].queryPool)
                    ->m_bufferResource->getDeviceAddress() +
                sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC) *
                    queryDescs[i].firstQueryIndex;
            break;
        case QueryType::AccelerationStructureCurrentSize:
            postBuildInfoDescs[i].InfoType =
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE;
            postBuildInfoDescs[i].DestBuffer =
                static_cast<PlainBufferProxyQueryPoolImpl*>(queryDescs[i].queryPool)
                    ->m_bufferResource->getDeviceAddress() +
                sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE_DESC) *
                    queryDescs[i].firstQueryIndex;
            break;
        case QueryType::AccelerationStructureSerializedSize:
            postBuildInfoDescs[i].InfoType =
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION;
            postBuildInfoDescs[i].DestBuffer =
                static_cast<PlainBufferProxyQueryPoolImpl*>(queryDescs[i].queryPool)
                    ->m_bufferResource->getDeviceAddress() +
                sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION_DESC) *
                    queryDescs[i].firstQueryIndex;
            break;
        }
    }
}

} // namespace d3d12

Result SLANG_MCALL getD3D12Adapters(List<AdapterInfo>& outAdapters)
{
    List<ComPtr<IDXGIAdapter>> dxgiAdapters;
    SLANG_RETURN_ON_FAIL(
        D3DUtil::findAdapters(DeviceCheckFlag::UseHardwareDevice, nullptr, dxgiAdapters));

    outAdapters.clear();
    for (const auto& dxgiAdapter : dxgiAdapters)
    {
        DXGI_ADAPTER_DESC desc;
        dxgiAdapter->GetDesc(&desc);
        AdapterInfo info = {};
        auto name = String::fromWString(desc.Description);
        memcpy(
            info.name,
            name.getBuffer(),
            Math::Min(name.getLength(), (Index)sizeof(AdapterInfo::name) - 1));
        info.vendorID = desc.VendorId;
        info.deviceID = desc.DeviceId;
        info.luid = D3DUtil::getAdapterLUID(dxgiAdapter);
        outAdapters.add(info);
    }
    return SLANG_OK;
}

Result SLANG_MCALL createD3D12Device(const IDevice::Desc* desc, IDevice** outDevice)
{
    RefPtr<d3d12::DeviceImpl> result = new d3d12::DeviceImpl();
    SLANG_RETURN_ON_FAIL(result->initialize(*desc));
    returnComPtr(outDevice, result);
    return SLANG_OK;
}

} // namespace gfx
