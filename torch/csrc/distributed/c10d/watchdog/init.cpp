#include <torch/csrc/distributed/c10d/watchdog/init.hpp>

#ifdef TORCH_USE_LIBUV

#include <memory>
#include <utility>

#include <ATen/DeviceAccelerator.h>
#include <c10/util/Exception.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <torch/csrc/distributed/c10d/watchdog/Watchdog.hpp>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace c10d::watchdog {
namespace {

// Holder that releases the GIL before dropping its shared_ptr. Destroying the
// last reference to a Watchdog joins its loop thread, which may itself be
// acquiring the GIL to run or release a Python callback; releasing the GIL here
// avoids that deadlock. Mirrors IntrusivePtrNoGilDestructor in the main c10d
// bindings, but for std::shared_ptr (so it composes with makeWatchdog and
// enable_shared_from_this).
template <typename T>
class SharedPtrNoGilDestructor {
  std::shared_ptr<T> impl_{};

 public:
  SharedPtrNoGilDestructor() = default;
  SharedPtrNoGilDestructor(const SharedPtrNoGilDestructor&) = default;
  SharedPtrNoGilDestructor(SharedPtrNoGilDestructor&&) noexcept = default;
  SharedPtrNoGilDestructor& operator=(const SharedPtrNoGilDestructor&) = default;
  SharedPtrNoGilDestructor& operator=(SharedPtrNoGilDestructor&&) noexcept =
      default;
  /* implicit */ SharedPtrNoGilDestructor(std::shared_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~SharedPtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        py::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  [[nodiscard]] T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return static_cast<bool>(impl_);
  }
};

// Wraps a Python-provided callback so that an exception raised on the watchdog
// loop thread is reported via sys.unraisablehook instead of escaping into the
// loop. pybind already makes the underlying std::function GIL-safe to call and
// destroy.
Callback wrapCallback(std::function<void()> fn) {
  return [fn = std::move(fn)]() {
    try {
      fn();
    } catch (py::error_already_set& e) {
      py::gil_scoped_acquire gil;
      e.discard_as_unraisable("torch.distributed._watchdog callback");
    } catch (const std::exception& e) {
      TORCH_WARN("torch.distributed._watchdog callback raised: ", e.what());
    }
  };
}

c10::Stream currentStream() {
  return at::accelerator::getCurrentStream(at::accelerator::getDeviceIndex());
}

} // namespace
} // namespace c10d::watchdog

PYBIND11_DECLARE_HOLDER_TYPE(
    T,
    c10d::watchdog::SharedPtrNoGilDestructor<T>,
    /*always_construct_holder=*/false)

namespace c10d::watchdog {

void initWatchdogBindings(py::module& module) {
  auto m = module.def_submodule(
      "_distributed_c10d_watchdog", "c10d watchdog bindings");

  py::class_<Handle>(m, "_WatchdogHandle")
      .def("cancel", &Handle::cancel)
      .def("completed", &Handle::completed);

  py::class_<Watchdog, SharedPtrNoGilDestructor<Watchdog>>(m, "_Watchdog")
      .def(
          py::init([](std::chrono::milliseconds pollInterval) {
            return SharedPtrNoGilDestructor<Watchdog>(makeWatchdog(pollInterval));
          }),
          py::arg("poll_interval") = std::chrono::seconds(1))
      .def_static(
          "_singleton",
          []() { return SharedPtrNoGilDestructor<Watchdog>(Watchdog::singleton()); })
      .def(
          "cpu_timeout",
          [](Watchdog& self,
             std::chrono::milliseconds timeout,
             std::function<void()> callback) {
            return self.cpu_timeout(timeout, wrapCallback(std::move(callback)));
          },
          py::arg("timeout"),
          py::arg("callback"))
      .def(
          "stream_timeout",
          [](Watchdog& self,
             std::chrono::milliseconds timeout,
             std::function<void()> callback) {
            return self.stream_timeout(
                currentStream(), timeout, wrapCallback(std::move(callback)));
          },
          py::arg("timeout"),
          py::arg("callback"))
      .def(
          "stream_completed",
          [](Watchdog& self, std::function<void()> callback) {
            return self.stream_completed(
                currentStream(), wrapCallback(std::move(callback)));
          },
          py::arg("callback"))
      .def(
          "op_timeout",
          [](Watchdog& self,
             std::chrono::milliseconds timeout,
             std::function<void()> callback) {
            return self.op_timeout(
                currentStream(), timeout, wrapCallback(std::move(callback)));
          },
          py::arg("timeout"),
          py::arg("callback"))
      .def("num_active_stream_timeouts", &Watchdog::numActiveStreamTimeouts);
}

} // namespace c10d::watchdog

#else // TORCH_USE_LIBUV

namespace c10d::watchdog {

// Without the libuv backend the submodule is not registered; importing
// torch._C._distributed_c10d_watchdog then raises ImportError.
void initWatchdogBindings(pybind11::module& /*module*/) {}

} // namespace c10d::watchdog

#endif // TORCH_USE_LIBUV
