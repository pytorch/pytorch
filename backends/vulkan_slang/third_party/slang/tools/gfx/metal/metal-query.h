// metal-query.h
#pragma once

#include "metal-base.h"
#include "metal-device.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

class QueryPoolImpl : public QueryPoolBase
{
public:
    RefPtr<DeviceImpl> m_device;
    NS::SharedPtr<MTL::CounterSampleBuffer> m_counterSampleBuffer;

    ~QueryPoolImpl();

    Result init(DeviceImpl* device, const IQueryPool::Desc& desc);

    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex index, GfxCount count, uint64_t* data) override;
};

} // namespace metal
} // namespace gfx
