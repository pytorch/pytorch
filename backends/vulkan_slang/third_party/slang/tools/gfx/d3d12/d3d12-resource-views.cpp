// d3d12-resource-views.cpp
#include "d3d12-resource-views.h"

#include "d3d12-device.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

ResourceViewInternalImpl::~ResourceViewInternalImpl()
{
    if (m_descriptor.cpuHandle.ptr)
        m_allocator->free(m_descriptor);
    for (auto desc : m_mapBufferStrideToDescriptor)
    {
        m_allocator->free(desc.second);
    }
}

SlangResult createD3D12BufferDescriptor(
    BufferResourceImpl* buffer,
    BufferResourceImpl* counterBuffer,
    IResourceView::Desc const& desc,
    uint32_t bufferStride,
    DeviceImpl* device,
    D3D12GeneralExpandingDescriptorHeap* descriptorHeap,
    D3D12Descriptor* outDescriptor)
{

    auto resourceImpl = (BufferResourceImpl*)buffer;
    auto resourceDesc = *resourceImpl->getDesc();
    const auto counterResourceImpl = static_cast<BufferResourceImpl*>(counterBuffer);

    uint64_t offset = desc.bufferRange.offset;
    uint64_t size = desc.bufferRange.size == 0 ? buffer->getDesc()->sizeInBytes - offset
                                               : desc.bufferRange.size;

    switch (desc.type)
    {
    default:
        return SLANG_FAIL;

    case IResourceView::Type::UnorderedAccess:
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
            uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uavDesc.Format = D3DUtil::getMapFormat(desc.format);
            if (bufferStride)
            {
                uavDesc.Buffer.FirstElement = offset / bufferStride;
                uavDesc.Buffer.NumElements = UINT(size / bufferStride);
                uavDesc.Buffer.StructureByteStride = bufferStride;
            }
            else if (desc.format == Format::Unknown)
            {
                uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
                uavDesc.Buffer.FirstElement = offset / 4;
                uavDesc.Buffer.NumElements = UINT(size / 4);
                uavDesc.Buffer.Flags |= D3D12_BUFFER_UAV_FLAG_RAW;
            }
            else
            {
                FormatInfo sizeInfo;
                gfxGetFormatInfo(desc.format, &sizeInfo);
                assert(sizeInfo.pixelsPerBlock == 1);
                uavDesc.Buffer.FirstElement = offset / sizeInfo.blockSizeInBytes;
                uavDesc.Buffer.NumElements = UINT(size / sizeInfo.blockSizeInBytes);
            }

            if (size >= (1ull << 32) - 8)
            {
                // D3D12 does not support view descriptors that has size near 4GB.
                // We will not create actual SRV/UAVs for such large buffers.
                // However, a buffer this large can still be bound as root parameter.
                // So instead of failing, we quietly ignore descriptor creation.
                outDescriptor->cpuHandle.ptr = 0;
            }
            else
            {
                SLANG_RETURN_ON_FAIL(descriptorHeap->allocate(outDescriptor));
                device->m_device->CreateUnorderedAccessView(
                    resourceImpl->m_resource,
                    counterResourceImpl ? counterResourceImpl->m_resource.getResource() : nullptr,
                    &uavDesc,
                    outDescriptor->cpuHandle);
            }
        }
        break;

    case IResourceView::Type::ShaderResource:
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srvDesc.Format = D3DUtil::getMapFormat(desc.format);
            srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (bufferStride)
            {
                srvDesc.Buffer.FirstElement = offset / bufferStride;
                srvDesc.Buffer.NumElements = UINT(size / bufferStride);
                srvDesc.Buffer.StructureByteStride = bufferStride;
            }
            else if (desc.format == Format::Unknown)
            {
                srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
                srvDesc.Buffer.FirstElement = offset / 4;
                srvDesc.Buffer.NumElements = UINT(size / 4);
                srvDesc.Buffer.Flags |= D3D12_BUFFER_SRV_FLAG_RAW;
            }
            else
            {
                FormatInfo sizeInfo;
                gfxGetFormatInfo(desc.format, &sizeInfo);
                assert(sizeInfo.pixelsPerBlock == 1);
                srvDesc.Buffer.FirstElement = offset / sizeInfo.blockSizeInBytes;
                srvDesc.Buffer.NumElements = UINT(size / sizeInfo.blockSizeInBytes);
            }

            if (size >= (1ull << 32) - 8)
            {
                // D3D12 does not support view descriptors that has size near 4GB.
                // We will not create actual SRV/UAVs for such large buffers.
                // However, a buffer this large can still be bound as root parameter.
                // So instead of failing, we quietly ignore descriptor creation.
                outDescriptor->cpuHandle.ptr = 0;
            }
            else
            {
                SLANG_RETURN_ON_FAIL(descriptorHeap->allocate(outDescriptor));
                device->m_device->CreateShaderResourceView(
                    resourceImpl->m_resource,
                    &srvDesc,
                    outDescriptor->cpuHandle);
            }
        }
        break;
    }
    return SLANG_OK;
}

SlangResult ResourceViewInternalImpl::getBufferDescriptorForBinding(
    DeviceImpl* device,
    ResourceViewImpl* view,
    uint32_t bufferStride,
    D3D12Descriptor& outDescriptor)
{
    // Look for an existing descriptor from the cache if it exists.
    if (auto descriptor = m_mapBufferStrideToDescriptor.tryGetValue(bufferStride))
    {
        outDescriptor = *descriptor;
        return SLANG_OK;
    }

    // We need to create and cache a d3d12 descriptor for the resource view that encodes
    // the given buffer stride.
    auto bufferResImpl = static_cast<BufferResourceImpl*>(view->m_resource.get());
    auto desc = view->m_desc;
    SLANG_RETURN_ON_FAIL(createD3D12BufferDescriptor(
        bufferResImpl,
        static_cast<BufferResourceImpl*>(view->m_counterResource.get()),
        desc,
        bufferStride,
        device,
        m_allocator,
        &outDescriptor));
    m_mapBufferStrideToDescriptor[bufferStride] = outDescriptor;

    return SLANG_OK;
}

Result ResourceViewImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::D3D12CpuDescriptorHandle;
    outHandle->handleValue = m_descriptor.cpuHandle.ptr;
    return SLANG_OK;
}

#if SLANG_GFX_HAS_DXR_SUPPORT

DeviceAddress AccelerationStructureImpl::getDeviceAddress()
{
    return m_buffer->getDeviceAddress() + m_offset;
}

Result AccelerationStructureImpl::getNativeHandle(InteropHandle* outHandle)
{
    outHandle->api = InteropHandleAPI::DeviceAddress;
    outHandle->handleValue = getDeviceAddress();
    return SLANG_OK;
}

#endif // SLANG_GFX_HAS_DXR_SUPPORT

} // namespace d3d12
} // namespace gfx
