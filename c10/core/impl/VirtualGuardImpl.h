#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10::impl {

/**
 * An implementation of DeviceGuardImplInterface which delegates
 * to virtual dispatch on the DeviceGuardImpl registry.
 */
class VirtualGuardImpl final : public DeviceGuardImplInterface {
 public:
  VirtualGuardImpl(DeviceType device_type)
      : impl_(getDeviceGuardImpl(device_type)) {}
  // This constructor exists purely for testing
  VirtualGuardImpl(const DeviceGuardImplInterface* impl) : impl_(impl) {}

  // Copying and moving is OK!
  VirtualGuardImpl(const VirtualGuardImpl&) = default;
  VirtualGuardImpl& operator=(const VirtualGuardImpl&) = default;
  VirtualGuardImpl(VirtualGuardImpl&&) noexcept = default;
  VirtualGuardImpl& operator=(VirtualGuardImpl&&) noexcept = default;
  ~VirtualGuardImpl() override = default;

  DeviceType type() const override {
    return impl_->type();
  }
  Device exchangeDevice(Device d) const override {
    return impl_->exchangeDevice(d);
  }
  Device getDevice() const override {
    return impl_->getDevice();
  }
  void setDevice(Device d) const override {
    impl_->setDevice(d);
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    impl_->uncheckedSetDevice(d);
  }
  Stream getStream(Device d) const override {
    return impl_->getStream(d);
  }
  Stream getNewStream(Device d, int priority = 0) const override {
    return impl_->getNewStream(d, priority);
  }
  Stream getDefaultStream(Device d) const override {
    return impl_->getDefaultStream(d);
  }
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return impl_->getStreamFromGlobalPool(d, isHighPriority);
  }
  Stream exchangeStream(Stream s) const override {
    return impl_->exchangeStream(s);
  }
  DeviceIndex deviceCount() const noexcept override {
    return impl_->deviceCount();
  }

  // Event functions
  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    impl_->record(event, stream, device_index, flag);
  }
  void block(void* event, const Stream& stream) const override {
    impl_->block(event, stream);
  }
  bool queryEvent(void* event) const override {
    return impl_->queryEvent(event);
  }
  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    impl_->destroyEvent(event, device_index);
  }

  bool queryStream(const Stream& stream) const override {
    return impl_->queryStream(stream);
  }
  void synchronizeStream(const Stream& stream) const override {
    impl_->synchronizeStream(stream);
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    impl_->recordDataPtrOnStream(data_ptr, stream);
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    return impl_->elapsedTime(event1, event2, device_index);
  }

  void synchronizeEvent(void* event) const override {
    impl_->synchronizeEvent(event);
  }

  void synchronizeDevice(const DeviceIndex device_index) const override {
    impl_->synchronizeDevice(device_index);
  }

 private:
  const DeviceGuardImplInterface* impl_ = nullptr;
};

} // namespace c10::impl
