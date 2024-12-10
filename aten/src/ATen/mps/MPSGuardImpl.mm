//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSGuardImpl.h>

namespace at::mps {

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

  // Check if the MPS event ID is valid. If not, acquire a new event from the
  // MPS event pool and assign it to the event pointer. Then record the event in
  // the MPS event pool.
  auto mps_event_id = (__bridge id_t)(*event);
  if (!mps_event_id) {
    mps_event_id = at::mps::getMPSEventPool()->acquireEvent(EventFlag);
    *event = (__birdge void*)(mps_event_id);
  }
  MPSStream mps_stream{stream};
  at::mps::getMPSEventPool()->recordEvent(mps_event_id, true);
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

void MPSGuardImpl::synchronizeDevice(const DeviceIndex device_index) const {
  at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

} // namespace at::mps
