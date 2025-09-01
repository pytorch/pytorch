#pragma once

#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <include/openreg.h>

#include "OpenRegEvent.h"
#include "OpenRegFunctions.h"
#include "OpenRegStream.h"

namespace c10::openreg {

struct OpenRegGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = c10::DeviceType::PrivateUse1;

  OpenRegGuardImpl() = default;

  explicit OpenRegGuardImpl(c10::DeviceType t) {
    TORCH_CHECK(
        t == static_type,
        "OpenRegGuardImpl initialized with non-PrivateUse1 DeviceType: ",
        t);
  }

  c10::DeviceType type() const override {
    return static_type;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_CHECK(
        d.is_privateuseone(), "Excepted a PrivateUse1 device, but got ", d);

    auto old_device_index = ExchangeDevice(d.index());
    return c10::Device(static_type, old_device_index);
  }

  c10::Device getDevice() const override {
    int device_index = current_device();
    return c10::Device(static_type, device_index);
  }

  void setDevice(c10::Device d) const override {
    TORCH_CHECK(
        d.is_privateuseone(), "Excepted a PrivateUse1 device, but got ", d);

    set_device(d.index());
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {
    set_device(d.index());
  }

  c10::Stream getStream(c10::Device d) const noexcept override {
    return getCurrentOpenRegStream(d.index()).unwrap();
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return getDefaultOpenRegStream(d.index());
  }

  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }

  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    const OpenRegStream stream(s);
    const auto old_stream = getCurrentOpenRegStream(s.device().index());
    setCurrentOpenRegStream(stream);
    return old_stream.unwrap();
  }

  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;

    auto or_event = static_cast<orEvent_t>(event);
    auto orig_device = current_device();
    set_device(device_index);
    orEventDestroy(or_event);
    set_device(orig_device);
  }

  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
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

      orEventCreateWithFlags(&or_event, or_flag);
    }

    orEventRecord(or_event, or_stream);
    *event = or_event;

    set_device(orig_device);
  }

  void block(void* event, const c10::Stream& stream) const override {
    if (!event)
      return;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    OpenRegStream or_stream{stream};
    const auto orig_device = current_device();
    set_device(stream.device().index());
    orStreamWaitEvent(or_stream, or_event, 0);
    set_device(orig_device);
  }

  bool queryEvent(void* event) const override {
    if (!event)
      return true;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    const orError_t err = orEventQuery(or_event);

    return err == orSuccess ? true : false;
  }

  bool queryStream(const c10::Stream& stream) const override {
    OpenRegStream or_stream{stream};
    return or_stream.query();
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    OpenRegStream or_stream{stream};
    or_stream.synchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    orEventSynchronize(or_event);
  }

  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    DeviceIndex orig_device{-1};
    auto orig_devicec = current_device();
    set_device(device_index);
    orDeviceSynchronize();
    set_device(orig_device);
  }

  double elapsedTime(
      void* event1,
      void* event2,
      const c10::DeviceIndex device_index) const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    auto orig_device = current_device();
    set_device(device_index);

    orEvent_t or_event1 = static_cast<orEvent_t>(event1);
    orEvent_t or_event2 = static_cast<orEvent_t>(event2);
    float time_ms = 0;
    orEventElapsedTime(&time_ms, or_event1, or_event2);

    set_device(orig_device);

    return static_cast<double>(time_ms);
  }
};

} // namespace c10::openreg
