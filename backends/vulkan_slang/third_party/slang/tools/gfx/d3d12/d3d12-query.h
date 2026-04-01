// d3d12-query.h
#pragma once

#include "d3d12-base.h"
#include "d3d12-buffer.h"
#include "d3d12-device.h"

namespace gfx
{
namespace d3d12
{

using namespace Slang;

class QueryPoolImpl : public QueryPoolBase
{
public:
    Result init(const IQueryPool::Desc& desc, DeviceImpl* device);

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex queryIndex, GfxCount count, uint64_t* data) override;

    void writeTimestamp(ID3D12GraphicsCommandList* cmdList, GfxIndex index);

public:
    D3D12_QUERY_TYPE m_queryType;
    ComPtr<ID3D12QueryHeap> m_queryHeap;
    D3D12Resource m_readBackBuffer;
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12Fence> m_fence;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    HANDLE m_waitEvent;
    UINT64 m_eventValue = 0;
};

/// Implements the IQueryPool interface with a plain buffer.
/// Used for query types that does not correspond to a D3D query,
/// such as ray-tracing acceleration structure post-build info.
class PlainBufferProxyQueryPoolImpl : public QueryPoolBase
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    IQueryPool* getInterface(const Guid& guid);

public:
    Result init(const IQueryPool::Desc& desc, DeviceImpl* device, uint32_t stride);

    virtual SLANG_NO_THROW Result SLANG_MCALL reset() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex queryIndex, GfxCount count, uint64_t* data) override;

public:
    QueryType m_queryType;
    RefPtr<BufferResourceImpl> m_bufferResource;
    RefPtr<DeviceImpl> m_device;
    List<uint8_t> m_result;
    bool m_resultDirty = true;
    uint32_t m_stride = 0;
    uint32_t m_count = 0;
};

} // namespace d3d12
} // namespace gfx
