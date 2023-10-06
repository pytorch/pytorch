//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSGuardImpl.h>

namespace at {
namespace mps {

void MPSGuardImpl::createEvent(mpsEvent_t* event, const EventFlag flag) const {}

void MPSGuardImpl::destroyEvent(void* event, const DeviceIndex device_index) const noexcept {
  if (!event)
    return;
  auto mps_event = static_cast<mpsEvent_t>(event);
  mps_event->~MPSEvent();
}

void MPSGuardImpl::record(void** event,
                          const Stream& stream,
                          const DeviceIndex device_index,
                          const EventFlag flag) const {
  TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
              "Event device index ",
              device_index,
              " does not match recording stream's device index ",
              stream.device_index(),
              ".");

  auto mps_event = static_cast<mpsEvent_t>(*event);
  MPSStream mps_stream{stream};
  mps_event->record(true);
}

void MPSGuardImpl::block(void* event, const Stream& stream) const {
  auto mps_event = static_cast<mpsEvent_t>(event);
  MPSStream mps_stream{stream};

  mps_event->wait(true, false);
}

bool MPSGuardImpl::queryEvent(void* event) const {
  auto mps_event = static_cast<mpsEvent_t>(event);
  return mps_event->query();
}

}
}
