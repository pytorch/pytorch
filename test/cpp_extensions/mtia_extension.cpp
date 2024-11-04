#include <ATen/detail/MTIAHooksInterface.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Logging.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <thread>
namespace torch::mtia {

constexpr c10::DeviceType kMTIADeviceType = c10::DeviceType::MTIA;
constexpr c10::DeviceIndex kMTIADeviceCount = 2;
static thread_local c10::DeviceIndex current_device = 0;
static thread_local std::array<c10::Stream, kMTIADeviceCount> current_streams =
    {c10::Stream::unpack3(0, 0, c10::DeviceType::MTIA),
     c10::Stream::unpack3(0, 1, c10::DeviceType::MTIA)};
static int64_t stream_id_gen = 1;
static int64_t event_id_gen = 1;
static std::array<c10::Stream, kMTIADeviceCount> default_streams = {
    c10::Stream::unpack3(0, 0, c10::DeviceType::MTIA),
    c10::Stream::unpack3(0, 1, c10::DeviceType::MTIA)};
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
    return c10::Device(kMTIADeviceType, current_device);
  }

  void setDevice(c10::Device d) const override {
    c10::Device current_device = getDevice();
    if (current_device.index() != d.index()) {
      current_device = d;
    }
  }
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    (void)d;
  }
  c10::Stream getStream(c10::Device d) const noexcept override {
    return current_streams[d.index()];
  }
  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    (void)priority;
    return c10::Stream::unpack3(stream_id_gen++, d.index(), d.type());
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return default_streams[d.index()];
  }
  c10::Stream getStreamFromGlobalPool(
      c10::Device d,
      bool isHighPriority = false) const override {
    return c10::Stream::unpack3(stream_id_gen++, d.index(), d.type());
  }
  // NB: These do NOT set the current device
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    c10::Stream old_stream = getStream(s.device());
    return old_stream;
  }
  c10::DeviceIndex deviceCount() const noexcept override {
    return kMTIADeviceCount;
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
      *event = reinterpret_cast<void*>(event_id_gen++);
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

  double elapsedTime(void* event1, void* event2, const c10::DeviceIndex device_index) const override {
    (void)device_index;
    uint64_t elapsed_time = 1e6;
    return (double)(elapsed_time / 1e6);
  }

  void synchronizeEvent(void* event) const override {
    (void)event;
  }
};

struct MTIAHooks : public at::MTIAHooksInterface {
  explicit MTIAHooks(at::MTIAHooksArgs) {}
  void initMTIA() const override {}

  bool hasMTIA() const override {
    return true;
  }

  c10::DeviceIndex deviceCount() const override {
    torch::utils::device_lazy_init(at::kMTIA);
    return c10::DeviceIndex(2);
  }

  void deviceSynchronize(c10::DeviceIndex device_index) const override {
    torch::utils::device_lazy_init(at::kMTIA);
    (void)device_index;
  }

  std::string showConfig() const override {
    return "None config";
  }

  c10::DeviceIndex exchangeDevice(c10::DeviceIndex device) const override {
    torch::utils::device_lazy_init(at::kMTIA);
    auto orig_device = current_device;
    if (current_device != device) {
      current_device = device;
    }
    return orig_device;
  }

  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device) const override {
    torch::utils::device_lazy_init(at::kMTIA);

    auto orig_device = current_device;
    if (current_device != device) {
      current_device = device;
    }
    return orig_device;
  }

  c10::Stream getDefaultStream(c10::DeviceIndex device) const override {
    torch::utils::device_lazy_init(at::kMTIA);

    return default_streams[device];
  }

  c10::Stream getCurrentStream(c10::DeviceIndex device) const override {
    torch::utils::device_lazy_init(at::kMTIA);

    return current_streams[device];
  }

  void setCurrentStream(const c10::Stream& stream) const override {
    torch::utils::device_lazy_init(at::kMTIA);

    current_streams[stream.device_index()] = stream;
  }

  c10::DeviceIndex getCurrentDevice() const override {
    torch::utils::device_lazy_init(at::kMTIA);

    return current_device;
  }

  void setCurrentDevice(c10::DeviceIndex device) const override {
    torch::utils::device_lazy_init(at::kMTIA);

    if (current_device != device) {
      current_device = device;
    }
  }
};

using at::MTIAHooksRegistry;
using at::RegistererMTIAHooksRegistry;

REGISTER_MTIA_HOOKS(MTIAHooks);
C10_REGISTER_GUARD_IMPL(MTIA, MTIAGuardImpl);

} // namespace torch::mtia
