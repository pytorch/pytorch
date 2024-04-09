#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Logging.h>
#include <ATen/detail/MTIAHooksInterface.h>
namespace torch::mtia {

constexpr c10::DeviceType kMTIADeviceType = c10::DeviceType::MTIA;

struct MTIAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  MTIAGuardImpl() = default;
  explicit MTIAGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == kMTIADeviceType);
  }
  c10::DeviceType type() const override {
    return kMTIADeviceType;
  }
  c10::Device exchangeDevice(c10::Device d) const override {
    c10::Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      setDevice(d);
    }
    return old_device;
  }
  c10::Device getDevice() const override {
    int64_t device_ordinal = 0;
    return c10::Device(
        kMTIADeviceType, static_cast<c10::DeviceIndex>(device_ordinal));
  }

  void setDevice(c10::Device d) const override {
    c10::Device current_device = getDevice();
  }
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    (void) d;
  }
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream::unpack3(
        static_cast<c10::StreamId>(0), d.index(), d.type());
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream::unpack3(
        static_cast<c10::StreamId>(0), d.index(), d.type());
  }
  c10::Stream getStreamFromGlobalPool(
      c10::Device d,
      bool isHighPriority = false) const override {
    return c10::Stream::unpack3(
        static_cast<c10::StreamId>(0), d.index(), d.type());
  }
  // NB: These do NOT set the current device
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    c10::Stream old_stream = getStream(s.device());
    return old_stream;
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    // Avoid logging or throwing exception here, since PyTorch use this function
    // to check the device availability.
    uint32_t count = 2;
    return static_cast<c10::DeviceIndex>(count);
  }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    (void)device_index;
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

    const auto orig_device = getDevice();

    setDevice(stream.device());

    if (*event == nullptr) {
      int64_t mtia_event{1};
      *event = reinterpret_cast<void*>(mtia_event);
    }
    setDevice(orig_device);
  }

  void block(void* event, const c10::Stream& stream) const override {
    (void)event;
    (void)stream;
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    (void)event;
    return true;
  }

  // Stream-related functions
  bool queryStream(const c10::Stream& stream) const override {
    (void)stream;
    return true;
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    (void)stream;
  }

  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
    (void)data_ptr;
    (void)stream;
  }
};

struct MTIAHooks : public at::MTIAHooksInterface {
  explicit MTIAHooks(at::MTIAHooksArgs) {}
  void initMTIA() const override {}

  bool hasMTIA() const override {
    return true;
  }

  c10::DeviceIndex deviceCount() const override {
    return c10::DeviceIndex(2);
  }

  void deviceSynchronize(c10::DeviceIndex device_index) const override {
    (void)device_index;
  }

  std::string showConfig() const override{return "None config";}

  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
    return device;
  }

  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
    return device;
  }

  c10::Stream getDefaultStream(c10::DeviceIndex device) const override {
    return c10::Stream::unpack3(
        static_cast<c10::StreamId>(1), device, c10::DeviceType::MTIA);
  }

  c10::Stream getCurrentStream(c10::DeviceIndex device) const override {
    return c10::Stream::unpack3(
        static_cast<c10::StreamId>(1), device, c10::DeviceType::MTIA);
  }

  void setCurrentStream(const c10::Stream& stream) const override {
    (void)stream;
  }

  c10::DeviceIndex getCurrentDevice() const override {
    return c10::DeviceIndex(0);
  }

  void setCurrentDevice(c10::DeviceIndex device) const override {
    (void)device;
  }
};

using at::MTIAHooksRegistry;
using at::RegistererMTIAHooksRegistry;

REGISTER_MTIA_HOOKS(MTIAHooks);
C10_REGISTER_GUARD_IMPL(MTIA, MTIAGuardImpl);

} // namespace torch::mtia
