// metal-fence.cpp
#include "metal-fence.h"

#include "metal-device.h"

namespace gfx
{

using namespace Slang;

namespace metal
{

FenceImpl::~FenceImpl() {}

Result FenceImpl::init(DeviceImpl* device, const IFence::Desc& desc)
{
    m_device = device;
    m_event = NS::TransferPtr(m_device->m_device->newSharedEvent());
    if (!m_event)
    {
        return SLANG_FAIL;
    }
    m_event->setSignaledValue(desc.initialValue);

    m_eventListener = NS::TransferPtr(MTL::SharedEventListener::alloc()->init());
    if (!m_eventListener)
    {
        return SLANG_FAIL;
    }

    return SLANG_OK;
}

Result FenceImpl::getCurrentValue(uint64_t* outValue)
{
    *outValue = m_event->signaledValue();
    return SLANG_OK;
}

Result FenceImpl::setCurrentValue(uint64_t value)
{
    m_event->setSignaledValue(value);
    return SLANG_OK;
}

Result FenceImpl::getSharedHandle(InteropHandle* outHandle)
{
    return SLANG_E_NOT_AVAILABLE;
}

Result FenceImpl::getNativeHandle(InteropHandle* outNativeHandle)
{
    outNativeHandle->api = InteropHandleAPI::Metal;
    outNativeHandle->handleValue = reinterpret_cast<intptr_t>(m_event.get());
    return SLANG_OK;
}

bool FenceImpl::waitForFence(uint64_t value, uint64_t timeout)
{
    // Create a semaphore to synchronize the notification
    dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

    // Create and store the notification block
    MTL::SharedEventNotificationBlock block = ^(MTL::SharedEvent* event, uint64_t eventValue) {
      dispatch_semaphore_signal(semaphore);
    };

    // Set up notification handler
    m_event->notifyListener(m_eventListener.get(), value, block);

    // Wait for the notification or timeout
    if (timeout & (1LLU << 63))
    {
        timeout = DISPATCH_TIME_FOREVER;
    }
    else
    {
        timeout = dispatch_time(DISPATCH_TIME_NOW, timeout);
    }

    long result = dispatch_semaphore_wait(semaphore, timeout);
    dispatch_release(semaphore);

    return result == 0;
}

} // namespace metal
} // namespace gfx
