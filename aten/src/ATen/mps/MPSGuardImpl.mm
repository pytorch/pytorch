//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSGuardImpl.h>

namespace at::mps {

void MPSGuardImpl::createEvent(mpsEvent_t* event, const EventFlag flag) const {}

void MPSGuardImpl::destroyEvent(void* event, const DeviceIndex device_index) const noexcept {
  if (!event)
    return;

  auto mps_event_id = (__bridge id_t)(intptr_t)(event);
  at::mps::getMPSEventPool()->releaseEvent(mps_event_id);
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
  auto mps_event_id = (__bridge id_t)(intptr_t)(*event);
  if (!mps_event_id) {
    mps_event_id = at::mps::getMPSEventPool()->acquireEvent(flag == EventFlag::BACKEND_DEFAULT);
    *event = (__bridge void*)(intptr_t)(mps_event_id);
  }
  MPSStream mps_stream{stream};
  at::mps::getMPSEventPool()->recordEvent(mps_event_id, true);
}

void MPSGuardImpl::block(void* event, const Stream& stream) const {
  auto mps_event_id = (__bridge id_t)(intptr_t)(event);
  MPSStream mps_stream{stream};

  at::mps::getMPSEventPool()->waitForEvent(mps_event_id, false);
}

bool MPSGuardImpl::queryEvent(void* event) const {
  auto mps_event_id = (__bridge id_t)(intptr_t)(event);
  return at::mps::getMPSEventPool()->queryEvent(mps_event_id);
}

void MPSGuardImpl::synchronizeEvent(void* event) const {
  auto mps_event_id = (__bridge id_t)(intptr_t)(event);
  return at::mps::getMPSEventPool()->synchronizeEvent(mps_event_id);
}

double MPSGuardImpl::elapsedTime(void* event1, void* event2, const DeviceIndex device_index) const {
  TORCH_CHECK(event1 && event2, "Both events must be recorded before calculating elapsed time.");
  auto start_event_id = (__bridge id_t)(intptr_t)(event1);
  auto end_event_id = (__bridge id_t)(intptr_t)(event2);
  return at::mps::getMPSEventPool()->elapsedTime(start_event_id, end_event_id);
}

void MPSGuardImpl::synchronizeDevice(const DeviceIndex device_index) const {
  at::mps::getDefaultMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);
}

} // namespace at::mps
