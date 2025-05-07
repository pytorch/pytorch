#include "OpenReg.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace openreg {
namespace {

// Python factory function where real implementations can be found
PyObject* py_factory;

struct HostAllocator final : at::Allocator {
  HostAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    void* data = nullptr;
    if (nbytes > 0) {
      data = reinterpret_cast<void*>(
          get_method("hostMalloc")(nbytes).cast<openreg_ptr_t>());
      TORCH_CHECK(data, "Failed to allocator ", nbytes, " bytes on host.");
    }
    return {data, data, &ReportAndDelete<kHostFreeMethod>, at::Device(at::kCPU)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete<kHostFreeMethod>;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("hostCopyData")(
        reinterpret_cast<openreg_ptr_t>(dest),
        reinterpret_cast<openreg_ptr_t>(src),
        count);
  }
};

static HostAllocator global_host_alloc;

static c10::DeviceIndex device_count() {
  py::gil_scoped_acquire acquire;
  return get_method("deviceCount")().cast<c10::DeviceIndex>();
}

static c10::DeviceIndex current_device_idx() {
  py::gil_scoped_acquire acquire;
  return get_method("getDevice")().cast<c10::DeviceIndex>();
}

class OpenRegGeneratorImpl : public at::CPUGeneratorImpl {
 public:
  OpenRegGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~OpenRegGeneratorImpl() override = default;
};

static at::Generator make_openreg_generator(c10::DeviceIndex device_index) {
  return at::make_generator<OpenRegGeneratorImpl>(device_index);
}

// Default, global generators, one per device.
static std::vector<at::Generator> default_generators;

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  OpenRegHooksInterface() {};
  ~OpenRegHooksInterface() override = default;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    py::gil_scoped_acquire acquire;
    return get_method("hasPrimaryContext")(device_index).cast<bool>();
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return &global_host_alloc;
  }

  bool isPinnedPtr(const void* data) const override {
    py::gil_scoped_acquire acquire;
    return get_method("isPinnedPtr")(reinterpret_cast<openreg_ptr_t>(data))
        .cast<bool>();
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    static bool flag [[maybe_unused]] = []() {
      auto deivce_nums = device_count();
      default_generators.resize(deivce_nums);
      for (auto i = 0; i < deivce_nums; i++) {
        default_generators[i] = make_openreg_generator(i);
        default_generators[i].seed();
      }
      return true;
    }();

    c10::DeviceIndex idx = device_index;
    if (idx == -1) {
      idx = current_device_idx();
    } else {
      TORCH_CHECK(idx >= 0 && idx < device_count());
    }
    return default_generators[idx];
  }

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return make_openreg_generator(device_index);
  }
};

static bool register_hook_flag [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new OpenRegHooksInterface());

  return true;
}();

// Device guard registration
struct OpenRegGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  OpenRegGuardImpl() = default;
  explicit OpenRegGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == static_type);
  }

  /**
   * Return the type of device managed by this guard implementation.
   */
  c10::DeviceType type() const override {
    return static_type;
  }

  /**
   * Set the current device to Device, and return the previous c10::Device.
   */
  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto old_device_index =
        get_method("exchangeDevice")(d.index()).cast<c10::DeviceIndex>();
    return c10::Device(static_type, old_device_index);
  }

  /**
   * Get the current device.
   */
  c10::Device getDevice() const override {
    return c10::Device(static_type, current_device_idx());
  }

  /**
   * Set the current device to c10::Device.
   */
  void setDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    py::gil_scoped_acquire acquire;
    auto device = get_method("setDevice")(d.index());
  }

  /**
   * Set the current device to c10::Device, without checking for errors
   * (so, e.g., this can be called from a destructor).
   */
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto device = get_method("uncheckedSetDevice")(d.index());
  }

  /**
   * Get the current stream for a given device.
   */
  c10::Stream getStream(c10::Device d) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto stream_id = get_method("getStream")(d.index()).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
  }

  /**
   * Get the default stream for a given device.
   */
  c10::Stream getDefaultStream(c10::Device d) const override {
    py::gil_scoped_acquire acquire;
    return get_method("getDefaultStream")(d.index()).cast<c10::Stream>();
  }

  /**
   * Get a stream from the global pool for a given device.
   */
  c10::Stream getStreamFromGlobalPool(
      c10::Device d,
      bool isHighPriority = false) const override {
    py::gil_scoped_acquire acquire;
    return get_method("getStreamFromGlobalPool")(d.index(), isHighPriority)
        .cast<c10::Stream>();
  }

  /**
   * Return a new stream for a given device and priority. The stream will be
   * copied and shared around, device backend should be able to correctly handle
   * the lifetime of the stream.
   */
  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    py::gil_scoped_acquire acquire;
    auto stream_id =
        get_method("getNewStream")(d.index(), priority).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, d, stream_id);
  }

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    py::gil_scoped_acquire acquire;
    auto stream_id = get_method("exchangeStream")(s).cast<c10::StreamId>();
    return c10::Stream(c10::Stream::UNSAFE, s.device(), stream_id);
  }

  /**
   * Destroys the given event.
   */
  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    py::gil_scoped_acquire acquire;
    get_method("destroyEvent")((int64_t)event, device_index);
  }

  /**
   * Increments the event's version and enqueues a job with this version
   * in the stream's work queue. When the stream process that job
   * it notifies all streams waiting on / blocked by that version of the
   * event to continue and marks that version as recorded.
   * */
  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    py::gil_scoped_acquire acquire;
    get_method("record")((int64_t)event, stream, device_index, (int64_t)flag);
  }

  /**
   * Does nothing if the event has not been scheduled to be recorded.
   * If the event was previously enqueued to be recorded, a command
   * to wait for the version of the event that exists at the time of this call
   * is inserted in the stream's work queue.
   * When the stream reaches this command it will stop processing
   * additional commands until that version of the event is marked as recorded.
   */
  void block(void* event, const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("block")((int64_t)event, stream);
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  bool queryEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    return get_method("queryEvent")((int64_t)event).cast<bool>();
  }

  /**
   * Get the number of devices.  WARNING: This is REQUIRED to not raise
   * an exception.  If there is some sort of problem, e.g., driver error,
   * you should report that there are zero available devices.
   */
  c10::DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }
  /**
   * Return true if all the work previously enqueued on the stream for
   * asynchronous execution has completed running on the device.
   */
  bool queryStream(const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    return get_method("queryStream")(stream).cast<bool>();
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * enqueued on the stream has completed running on the device.
   */
  virtual void synchronizeStream(const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("synchronizeStream")(stream);
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * recorded on the event has completed running on the device.
   */
  void synchronizeEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    get_method("synchronizeEvent")((int64_t)event);
  }

  /**
   * Ensure the caching allocator (if any) is aware that the given DataPtr is
   * being used on the given stream, and that it should thus avoid recycling the
   * DataPtr until all work on that stream is done.
   */
  void recordDataPtrOnStream(
      const c10::DataPtr& data_ptr,
      const c10::Stream& stream) const override {
    py::gil_scoped_acquire acquire;
    get_method("recordDataPtrOnStream")(data_ptr, stream);
  }

  /**
   * Fetch the elapsed time between two recorded events.
   */
  double elapsedTime(
      void* event1,
      void* event2,
      const c10::DeviceIndex device_index) const override {
    py::gil_scoped_acquire acquire;
    return get_method("elapsedTime")(
               (int64_t)event1, (int64_t)event2, device_index)
        .cast<double>();
  }
};

// Register our device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, OpenRegGuardImpl);

} // namespace

// Setter for the python dictionary with implementations
void set_impl_factory(PyObject* factory) {
  py_factory = factory;
}

py::function get_method(const char* name) {
  auto factory = py::cast<py::function>(py_factory);
  return factory(name);
}

} // namespace openreg
