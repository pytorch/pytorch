#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/mtia/Module.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::mtia {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_mtia_init", []() {
    TORCH_INTERNAL_ASSERT(!torch::utils::is_device_in_bad_fork(at::kMTIA));
    torch::utils::register_fork_handler_for_device_init(at::kMTIA);
    at::globalContext().lazyInitDevice(c10::DeviceType::MTIA);
  });

  m.def("_mtia_isBuilt", []() {
    // Check if the MTIAHooks class has been registered with the registry.
    return at::detail::isMTIAHooksBuilt();
  });

  m.def("_mtia_isInBadFork", []() {
    return torch::utils::is_device_in_bad_fork(at::kMTIA);
  });

  m.def("_mtia_getCurrentStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getCurrentStream(device_index);
  });

  m.def("_mtia_getCurrentRawStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getCurrentRawStream(device_index);
  });

  m.def("_mtia_deviceSynchronize", []() {
    torch::utils::device_lazy_init(at::kMTIA);
    at::detail::getMTIAHooks().deviceSynchronize(
        at::detail::getMTIAHooks().getCurrentDevice());
  });

  m.def("_mtia_exchangeDevice", [](c10::DeviceIndex device_index) {
    if (device_index < 0) {
      return static_cast<c10::DeviceIndex>(-1);
    }
    return at::detail::getMTIAHooks().exchangeDevice(device_index);
  });

  m.def("_mtia_maybeExchangeDevice", [](c10::DeviceIndex device_index) {
    if (device_index < 0) {
      return static_cast<c10::DeviceIndex>(-1);
    }
    return at::detail::getMTIAHooks().maybeExchangeDevice(device_index);
  });

  m.def("_mtia_getDefaultStream", [](c10::DeviceIndex device_index) {
    torch::utils::device_lazy_init(at::kMTIA);
    return at::detail::getMTIAHooks().getDefaultStream(device_index);
  });

  m.def(
      "_mtia_setStream",
      [](int64_t stream_id,
         c10::DeviceIndex device_index,
         int64_t device_type) {
        torch::utils::device_lazy_init(at::kMTIA);
        at::detail::getMTIAHooks().setCurrentStream(c10::Stream::unpack3(
            stream_id,
            device_index,
            static_cast<c10::DeviceType>(device_type)));
      });

  m.def("_mtia_setCurrentStream", [](const c10::Stream& stream) {
    torch::utils::device_lazy_init(at::kMTIA);
    auto device = at::detail::getMTIAHooks().getCurrentDevice();
    if (device != stream.device_index()) {
      at::detail::getMTIAHooks().setCurrentDevice(stream.device_index());
    }
    at::detail::getMTIAHooks().setCurrentStream(stream);
  });

  m.def("_mtia_memoryStats", [](c10::DeviceIndex device_index) {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().memoryStats(device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_getDeviceCapability", [](c10::DeviceIndex device_index) {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().getDeviceCapability(device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_getDeviceProperties", [](c10::DeviceIndex device_index) {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().getDeviceProperties(device_index);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_emptyCache", []() { at::detail::getMTIAHooks().emptyCache(); });

  m.def(
      "_mtia_recordMemoryHistory",
      [](const std::optional<std::string>& enabled,
         const std::string& stacks,
         size_t max_entries) {
        at::detail::getMTIAHooks().recordMemoryHistory(
            enabled, stacks, max_entries);
      });

  m.def("_mtia_memorySnapshot", []() {
    PyObject* raw_pyobject =
        at::detail::getMTIAHooks().memorySnapshot(std::nullopt);
    return py::reinterpret_steal<py::object>(raw_pyobject);
  });

  m.def("_mtia_attachOutOfMemoryObserver", [](const py::function& observer) {
    at::detail::getMTIAHooks().attachOutOfMemoryObserver(observer.ptr());
    return;
  });

  m.def("_mtia_getDeviceCount", []() {
    return at::detail::getMTIAHooks().deviceCount();
  });

  m.def("_mtia_resetPeakMemoryStats", [](c10::DeviceIndex device_index) {
    at::detail::getMTIAHooks().resetPeakMemoryStats(device_index);
  });
}

} // namespace torch::mtia
