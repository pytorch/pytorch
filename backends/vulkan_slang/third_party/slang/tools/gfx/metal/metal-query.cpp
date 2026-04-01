// metal-query.cpp
#include "metal-query.h"

// #include "metal-util.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

QueryPoolImpl::~QueryPoolImpl() {}

static MTL::CounterSet* findCounterSet(MTL::Device* device, QueryType queryType)
{
    if (queryType != QueryType::Timestamp)
    {
        return nullptr;
    }

    static NS::String* timestampStr = MTLSTR("timestamp");

    for (int i = 0; i < device->counterSets()->count(); ++i)
    {
        MTL::CounterSet* counterSet =
            static_cast<MTL::CounterSet*>(device->counterSets()->object(i));
        for (int j = 0; j < counterSet->counters()->count(); ++j)
        {
            MTL::Counter* counter = static_cast<MTL::Counter*>(counterSet->counters()->object(j));
            if (counter->name()->isEqualToString(MTL::CommonCounterTimestamp))
            {
                return counterSet;
            }
        }
    }
    return nullptr;
}

Result QueryPoolImpl::init(DeviceImpl* device, const IQueryPool::Desc& desc)
{
    m_device = device;
    m_desc = desc;

    MTL::CounterSet* counterSet = findCounterSet(m_device->m_device.get(), m_desc.type);
    if (!counterSet)
    {
        return SLANG_E_NOT_AVAILABLE;
    }

    NS::SharedPtr<MTL::CounterSampleBufferDescriptor> counterSampleBufferDesc =
        NS::TransferPtr(MTL::CounterSampleBufferDescriptor::alloc()->init());
    counterSampleBufferDesc->setStorageMode(MTL::StorageModeShared);
    counterSampleBufferDesc->setSampleCount(m_desc.count);
    counterSampleBufferDesc->setCounterSet(counterSet);

    m_device->m_device->counterSets();

    NS::Error* error;
    m_counterSampleBuffer = NS::TransferPtr(
        m_device->m_device->newCounterSampleBuffer(counterSampleBufferDesc.get(), &error));

    return m_counterSampleBuffer ? SLANG_OK : SLANG_FAIL;
}

Result QueryPoolImpl::getResult(GfxIndex index, GfxCount count, uint64_t* data)
{
    return SLANG_E_NOT_IMPLEMENTED;
}

} // namespace metal
} // namespace gfx
