// cpu-query.h
#pragma once
#include "cpu-base.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

class QueryPoolImpl : public QueryPoolBase
{
public:
    List<uint64_t> m_queries;
    Result init(const IQueryPool::Desc& desc);
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex queryIndex, GfxCount count, uint64_t* data) override;
};

} // namespace cpu
} // namespace gfx
