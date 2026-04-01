// d3d11-query.cpp
#include "d3d11-query.h"

#include "core/slang-process.h"

namespace gfx
{

using namespace Slang;

namespace d3d11
{

Result QueryPoolImpl::init(const IQueryPool::Desc& desc, DeviceImpl* device)
{
    m_device = device;
    m_queryDesc.MiscFlags = 0;
    switch (desc.type)
    {
    case QueryType::Timestamp:
        m_queryDesc.Query = D3D11_QUERY_TIMESTAMP;
        break;
    default:
        return SLANG_E_INVALID_ARG;
    }
    m_queries.setCount(desc.count);
    return SLANG_OK;
}

ID3D11Query* QueryPoolImpl::getQuery(SlangInt index)
{
    if (!m_queries[index])
        m_device->m_device->CreateQuery(&m_queryDesc, m_queries[index].writeRef());
    return m_queries[index].get();
}

SLANG_NO_THROW Result SLANG_MCALL
QueryPoolImpl::getResult(GfxIndex queryIndex, GfxCount count, uint64_t* data)
{
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT disjointData;
    while (S_OK != m_device->m_immediateContext->GetData(
                       m_device->m_disjointQuery,
                       &disjointData,
                       sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT),
                       0))
    {
        Process::sleepCurrentThread(1);
    }
    m_device->m_info.timestampFrequency = disjointData.Frequency;

    for (SlangInt i = 0; i < count; i++)
    {
        SLANG_RETURN_ON_FAIL(m_device->m_immediateContext->GetData(
            m_queries[queryIndex + i],
            data + i,
            sizeof(uint64_t),
            0));
    }
    return SLANG_OK;
}

} // namespace d3d11
} // namespace gfx
