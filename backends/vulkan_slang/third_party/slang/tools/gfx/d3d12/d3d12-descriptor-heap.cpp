
#include "d3d12-descriptor-heap.h"

namespace gfx
{
using namespace Slang;

D3D12DescriptorHeap::D3D12DescriptorHeap()
    : m_totalSize(0), m_currentIndex(0), m_descriptorSize(0)
{
}

Result D3D12DescriptorHeap::init(
    ID3D12Device* device,
    int size,
    D3D12_DESCRIPTOR_HEAP_TYPE type,
    D3D12_DESCRIPTOR_HEAP_FLAGS flags)
{
    m_device = device;

    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = size;
    srvHeapDesc.Flags = flags;
    srvHeapDesc.Type = type;
    SLANG_RETURN_ON_FAIL(
        device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(m_heap.writeRef())));

    m_descriptorSize = device->GetDescriptorHandleIncrementSize(type);
    m_totalSize = size;
    m_heapFlags = flags;

    return SLANG_OK;
}

Result D3D12DescriptorHeap::init(
    ID3D12Device* device,
    const D3D12_CPU_DESCRIPTOR_HANDLE* handles,
    int numHandles,
    D3D12_DESCRIPTOR_HEAP_TYPE type,
    D3D12_DESCRIPTOR_HEAP_FLAGS flags)
{
    SLANG_RETURN_ON_FAIL(init(device, numHandles, type, flags));
    D3D12_CPU_DESCRIPTOR_HANDLE dst = m_heap->GetCPUDescriptorHandleForHeapStart();

    // Copy them all
    for (int i = 0; i < numHandles; i++, dst.ptr += m_descriptorSize)
    {
        D3D12_CPU_DESCRIPTOR_HANDLE src = handles[i];
        if (src.ptr != 0)
        {
            device->CopyDescriptorsSimple(1, dst, src, type);
        }
    }

    return SLANG_OK;
}

} // namespace gfx
