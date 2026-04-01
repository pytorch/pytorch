// cpu-query.cpp
#include "cpu-query.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

Result QueryPoolImpl::init(const IQueryPool::Desc& desc)
{
    m_queries.setCount(desc.count);
    return SLANG_OK;
}

SLANG_NO_THROW Result SLANG_MCALL
QueryPoolImpl::getResult(GfxIndex queryIndex, GfxCount count, uint64_t* data)
{
    for (GfxCount i = 0; i < count; i++)
    {
        data[i] = m_queries[queryIndex + i];
    }
    return SLANG_OK;
}

} // namespace cpu
} // namespace gfx
