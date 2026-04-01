// d3d11-query.h
#pragma once
#include "d3d11-base.h"
#include "d3d11-device.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

class QueryPoolImpl : public QueryPoolBase
{
public:
    List<ComPtr<ID3D11Query>> m_queries;
    RefPtr<DeviceImpl> m_device;
    D3D11_QUERY_DESC m_queryDesc;

    Result init(const IQueryPool::Desc& desc, DeviceImpl* device);
    ID3D11Query* getQuery(SlangInt index);
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex queryIndex, GfxCount count, uint64_t* data) override;
};

} // namespace d3d11
} // namespace gfx
