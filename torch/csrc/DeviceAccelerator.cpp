#include <torch/csrc/DeviceAccelerator.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch::accelerator {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_accelerator_getAccelerator", []() -> std::optional<c10::Device> {
    // If no accelerator was available at compile time, return None.
    auto acc = at::getAccelerator(false);
    if (acc.has_value()) {
      return acc.value();
    } else {
      return std::nullopt;
    }
  });

  m.def("_accelerator_deviceCount", []() {
    auto device_type = at::accelerator::getAccelerator(false);
    if (!device_type.has_value()) {
      return static_cast<c10::DeviceIndex>(0);
    }

    // Why not call at::accelerator::deviceCount() directly like other
    // accelerator python binding functions?
    // 1. Some accelerators, such as CUDA, have a Python implementation of
    // `device_count` that is non-poisoning.
    // 2. Features like DataLoader and Distributed Data Parallel (DDP) rely on
    // this behavior.
    // 3. To maintain consistency, we delegate the device count retrieval to the
    // Python implementation.

    std::string module_name =
        "torch." + at::DeviceTypeName(device_type.value(), true);
    auto module = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module) {
      throw python_error();
    }

    // Call the Python `device_count` method from the device-specific module
    auto res =
        THPObjectPtr(PyObject_CallMethod(module.get(), "device_count", ""));
    if (!res) {
      throw python_error();
    }
    return static_cast<c10::DeviceIndex>(PyLong_AsLongLong(res.get()));
  });

  m.def("_accelerator_setDeviceIndex", [](c10::DeviceIndex device_index) {
    // If device index is negative, no-op
    if (device_index < 0) {
      return;
    }
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    at::accelerator::setDeviceIndex(device_index);
  });

  m.def("_accelerator_getDeviceIndex", []() {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::getDeviceIndex();
  });

  m.def("_accelerator_setStream", [](c10::Stream stream) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    // Set the current device to the device of stream
    if (at::accelerator::getDeviceIndex() != stream.device_index()) {
      at::accelerator::setDeviceIndex(stream.device_index());
    }
    at::accelerator::setCurrentStream(stream);
  });

  m.def("_accelerator_getStream", [](c10::DeviceIndex device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    return at::accelerator::getCurrentStream(device_index);
  });

  m.def("_accelerator_synchronizeDevice", [](c10::DeviceIndex device_index) {
    const auto device_type = at::accelerator::getAccelerator(true).value();
    if (torch::utils::is_device_lazy_init_supported(device_type) &&
        !torch::utils::is_device_initialized(device_type)) {
      return;
    }
    torch::utils::maybe_initialize_device(device_type);
    {
      py::gil_scoped_release no_gil;
      at::accelerator::synchronizeDevice(device_index);
    }
  });
}

} // namespace torch::accelerator
