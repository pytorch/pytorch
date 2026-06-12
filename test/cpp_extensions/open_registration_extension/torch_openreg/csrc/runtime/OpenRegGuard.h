#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceCapability.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <include/openreg.h>

#include "OpenRegEvent.h"
#include "OpenRegFunctions.h"
#include "OpenRegStream.h"

namespace c10::openreg {

struct OpenRegGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = c10::DeviceType::PrivateUse1;

  OpenRegGuardImpl() = default;

  explicit OpenRegGuardImpl(DeviceType t) {
    TORCH_CHECK(t == static_type, "OpenRegGuardImpl initialized with non-PrivateUse1 DeviceType: ", t);
  }

  // LITERALINCLUDE START: OPENREG ALL DEVICE GUARD IMPL
  /**
   * Return the type of device managed by this guard implementation.
   */
  DeviceType type() const override {
    return static_type;
  }
  /**
   * Set the current device to device d, and return the previous Device.
   */
  // LITERALINCLUDE START: OPENREG GUARD DEVICE MANAGEMENT
  Device exchangeDevice(Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Expected a PrivateUse1 device, but got ", d);

    auto old_device_index = ExchangeDevice(d.index());
    return Device(static_type, old_device_index);
  }
  // LITERALINCLUDE END: OPENREG GUARD DEVICE MANAGEMENT

  /**
   * Get the current device.
   */
  Device getDevice() const override {
    int device_index = current_device();
    return c10::Device(static_type, device_index);
  }

  /**
   * Get the device capability for a given device.
   * By default, OpenReg has 2 same devices with the same capability.
   */
  DeviceCapability getDeviceCapability(Device /*unused*/) const override {
    return DeviceCapability();
  }

  /**
   * Set the current device to c10::Device.
   */
  void setDevice(Device d) const override {
    TORCH_CHECK(d.is_privateuseone(), "Expected a PrivateUse1 device, but got ", d);

    set_device(d.index());
  }

  /**
   * Set the current device to device d, without checking for errors
   * (so, e.g., this can be called from a destructor).
   */
  void uncheckedSetDevice(Device d) const noexcept override {
    set_device(d.index());
  }

  /**
   * Get the number of devices.
   *
   * WARNING: This is REQUIRED to not raise an exception.
   * If there is some sort of problem, e.g., driver error,
   * you should report that there are zero available devices.
   */
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  /**
   * Wait (by blocking the calling thread) until all the work has
   * completed running on the device.
   */
  void synchronizeDevice(const DeviceIndex device_index) const override {
    OPENREG_CHECK(orDeviceSynchronize());
  }
  // LITERALINCLUDE END: OPENREG ALL DEVICE GUARD IMPL

  // LITERALINCLUDE START: OPENREG ALL STREAM GUARD IMPL
  /**
   * Get the current stream for a given device.
   */
  Stream getStream(Device d) const noexcept override {
    return getCurrentOpenRegStream(d.index()).unwrap();
  }

  /**
   * Get the default stream for a given device.
   */
  Stream getDefaultStream(Device d) const override {
    return getDefaultOpenRegStream(d.index());
  }

  /**
   * Return a new stream for a given device and priority. The stream will be
   * copied and shared around, device backend should be able to correctly handle
   * the lifetime of the stream.
   */
  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }

  /**
   * Get a stream from the global pool for a given device.
   */
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false) const override {
    return getStreamFromPool(isHighPriority, d.index());
  }

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  Stream exchangeStream(Stream s) const noexcept override {
    const OpenRegStream stream(s);
    const auto old_stream = getCurrentOpenRegStream(s.device().index());
    setCurrentOpenRegStream(stream);
    return old_stream.unwrap();
  }

  /**
   * Return true if all the work previously enqueued on the stream for
   * asynchronous execution has completed running on the device.
   */
  bool queryStream(const Stream& stream) const override {
    OpenRegStream or_stream{stream};
    return or_stream.query();
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * enqueued on the stream has completed running on the device.
   */
  void synchronizeStream(const Stream& stream) const override {
    OpenRegStream or_stream{stream};
    or_stream.synchronize();
  }
  // LITERALINCLUDE END: OPENREG ALL STREAM GUARD IMPL

  // LITERALINCLUDE START: OPENREG ALL EVENT GUARD IMPL
  /**
   * Destroys the given event.
   */
  void destroyEvent(void* event, const DeviceIndex device_index) const noexcept override {
    if (!event)
      return;

    auto or_event = static_cast<orEvent_t>(event);
    auto orig_device = current_device();
    set_device(device_index);
    OPENREG_CHECK(orEventDestroy(or_event));
    set_device(orig_device);
  }

  /**
   * Increments the event's version and enqueues a job with this version
   * in the stream's work queue. When the stream process that job
   * it notifies all streams waiting on / blocked by that version of the
   * event to continue and marks that version as recorded.
   * */
  // LITERALINCLUDE START: OPENREG GUARD EVENT RECORD
  void record(void** event, const Stream& stream, const DeviceIndex device_index, const EventFlag flag) const override {
    TORCH_CHECK(device_index == -1 || device_index == stream.device_index(),
                "Event device index ",
                device_index,
                " does not match recording stream's device index ",
                stream.device_index(),
                ".");

    orEvent_t or_event = static_cast<orEvent_t>(*event);
    OpenRegStream or_stream{stream};

    const auto orig_device = current_device();
    set_device(stream.device().index());

    if (!or_event) {
      auto or_flag = orEventDisableTiming;
      switch (flag) {
        case EventFlag::PYTORCH_DEFAULT:
          or_flag = orEventDisableTiming;
          break;
        case EventFlag::BACKEND_DEFAULT:
          or_flag = orEventEnableTiming;
          break;
        default:
          TORCH_CHECK(false, "Received unknown flag");
      }

      OPENREG_CHECK(orEventCreateWithFlags(&or_event, or_flag));
    }

    OPENREG_CHECK(orEventRecord(or_event, or_stream));
    *event = or_event;

    set_device(orig_device);
  }
  // LITERALINCLUDE END: OPENREG GUARD EVENT RECORD

  /**
   * Does nothing if the event has not been scheduled to be recorded.
   * If the event was previously enqueued to be recorded, a command
   * to wait for the version of the event that exists at the time of this call
   * is inserted in the stream's work queue.
   * When the stream reaches this command it will stop processing
   * additional commands until that version of the event is marked as recorded.
   */
  void block(void* event, const Stream& stream) const override {
    if (!event)
      return;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    OpenRegStream or_stream{stream};
    const auto orig_device = current_device();
    set_device(stream.device().index());
    OPENREG_CHECK(orStreamWaitEvent(or_stream, or_event, 0));
    set_device(orig_device);
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  bool queryEvent(void* event) const override {
    if (!event)
      return true;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    const orError_t err = orEventQuery(or_event);

    return err == orSuccess;
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * recorded on the event has completed running on the device.
   */
  void synchronizeEvent(void* event) const override {
    if (!event)
      return;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    OPENREG_CHECK(orEventSynchronize(or_event));
  }

  /**
   * Fetch the elapsed time between two recorded events.
   */
  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index) const override {
    TORCH_CHECK(event1 && event2, "Both events must be recorded before calculating elapsed time.");
    auto orig_device = current_device();
    set_device(device_index);

    orEvent_t or_event1 = static_cast<orEvent_t>(event1);
    orEvent_t or_event2 = static_cast<orEvent_t>(event2);
    float time_ms = 0;
    OPENREG_CHECK(orEventElapsedTime(&time_ms, or_event1, or_event2));

    set_device(orig_device);

    return static_cast<double>(time_ms);
  }
  // LITERALINCLUDE END: OPENREG ALL EVENT GUARD IMPL
};

} // namespace c10::openreg
