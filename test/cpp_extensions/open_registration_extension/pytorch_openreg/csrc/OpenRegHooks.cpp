#include <OpenReg.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <iostream>

namespace openreg {

namespace {
// Python dictionary where real implementations can be found
PyObject* py_registry;

py::function get_method(const char* name) {
    return  py::cast<py::dict>(py_registry)[name];
}

// C++ hooks implementation
struct OpenRegHooksArgs : public at::PrivateUse1HooksArgs {};

struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
    OpenRegHooksInterface(OpenRegHooksArgs) {};
    ~OpenRegHooksInterface() override = default;

      bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
          return get_method("hasPrimaryContext")(device_index).cast<bool>();
      }
};

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, OpenRegHooksInterface, OpenRegHooksArgs);
C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, OpenRegHooksInterface, OpenRegHooksArgs);
// Using Create function to get PrivateUse1HooksInterface point from PrivateUse1HooksRegistry class.
C10_REGISTER_TYPED_CLASS(PrivateUse1HooksRegistry, "OpenRegHooks", OpenRegHooksInterface);

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
    auto old_device_index = get_method("exchangeDevice")(d.index()).cast<c10::DeviceIndex>();
    return c10::Device(static_type, old_device_index);
  }

  /**
   * Get the current device.
   */
  c10::Device getDevice() const override {
    py::gil_scoped_acquire acquire;
    auto device = get_method("getDevice")().cast<c10::DeviceIndex>();
    return c10::Device(static_type, device);
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
    return get_method("getStream")(d.index()).cast<c10::Stream>();
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
  c10::Stream getStreamFromGlobalPool(c10::Device d, bool isHighPriority = false) const override {
    py::gil_scoped_acquire acquire;
    return get_method("getStreamFromGlobalPool")(d.index(), isHighPriority).cast<c10::Stream>();
  }

  /**
   * Return a new stream for a given device and priority. The stream will be
   * copied and shared around, device backend should be able to correctly handle
   * the lifetime of the stream.
   */
  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    py::gil_scoped_acquire acquire;
    return get_method("getNewStream")(d.index(), priority).cast<c10::Stream>();
  }

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    py::gil_scoped_acquire acquire;
    return get_method("exchangeStream")(s).cast<c10::Stream>();
  }

  /**
   * Destroys the given event.
   */
  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    py::gil_scoped_acquire acquire;
    get_method("destroyEvent")(event, device_index);
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
    get_method("record")(event, stream, device_index, flag);
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
    get_method("block")(event, stream);
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  bool queryEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    return get_method("queryEvent")(event).cast<bool>();
  }

  /**
   * Get the number of devices.  WARNING: This is REQUIRED to not raise
   * an exception.  If there is some sort of problem, e.g., driver error,
   * you should report that there are zero available devices.
   */
  c10::DeviceIndex deviceCount() const noexcept override {
    py::gil_scoped_acquire acquire;
    return get_method("deviceCount")().cast<c10::DeviceIndex>();
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
  virtual void synchronizeStream(const c10::Stream& stream) const {
    py::gil_scoped_acquire acquire;
    get_method("synchronizeStream")(stream);
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * recorded on the event has completed running on the device.
   */
  void synchronizeEvent(void* event) const override {
    py::gil_scoped_acquire acquire;
    get_method("synchronizeEvent")(event);
  }

  /**
   * Ensure the caching allocator (if any) is aware that the given DataPtr is
   * being used on the given stream, and that it should thus avoid recycling the
   * DataPtr until all work on that stream is done.
   */
  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const c10::Stream& stream)
      const override {
    py::gil_scoped_acquire acquire;
    get_method("recordDataPtrOnStream")(data_ptr, stream);
  }

  /**
   * Fetch the elapsed time between two recorded events.
   */
  double elapsedTime(void* event1, void* event2, const c10::DeviceIndex device_index)
      const override {
    py::gil_scoped_acquire acquire;
    return get_method("elapsedTime")(event1, event2, device_index).cast<double>();
  }
};

// Register our device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, OpenRegGuardImpl);

} // anonymous namspaces

// Setter for the python dictionary with implementations
void set_impl_registry(PyObject* registry) {
    py_registry = registry;
}
} // openreg