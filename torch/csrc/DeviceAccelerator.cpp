#include <c10/core/DeviceGuard.h>
#include <torch/csrc/DeviceAccelerator.h>
#include <torch/csrc/utils/device_lazy_init.h>

namespace torch::accelerator {

void initModule(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_accelerator_getAccelerator", []() {
    // If no accelerator is currently available, raise an exception.
    return c10::Device(at::getAccelerator(true).value());
  });

  m.def("_accelerator_deviceCount", []() {
    const auto device_type = at::getAccelerator(false);
    if (!device_type.has_value()) {
      return static_cast<c10::DeviceIndex>(0);
    }
    torch::utils::maybe_initialize_device(device_type.value());
    c10::impl::VirtualGuardImpl impl(device_type.value());
    return static_cast<c10::DeviceIndex>(impl.deviceCount());
  });

  m.def("_accelerator_setDeviceIndex", [](c10::DeviceIndex device_index) {
    const auto device_type = at::getAccelerator(true).value();
    // If device index is negative, no-op
    if (device_index < 0) {
      return;
    }
    torch::utils::maybe_initialize_device(device_type);
    c10::impl::VirtualGuardImpl impl(device_type);
    impl.setDevice({device_type, device_index});
  });

  m.def("_accelerator_getDeviceIndex", []() {
    const auto device_type = at::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    c10::impl::VirtualGuardImpl impl(device_type);
    return static_cast<c10::DeviceIndex>(impl.getDevice().index());
  });

  m.def("_accelerator_setStream", [](c10::Stream stream) {
    const auto device_type = at::getAccelerator(true).value();
    TORCH_CHECK(
        device_type == stream.device_type(),
        "stream's device type ",
        c10::DeviceTypeName(stream.device_type()),
        " doesn't match the current accelerator ",
        c10::DeviceTypeName(device_type));
    torch::utils::maybe_initialize_device(device_type);
    c10::impl::VirtualGuardImpl impl(device_type);
    // Set the current device to the device of stream
    if (impl.getDevice().index() != stream.device_index()) {
      impl.setDevice(stream.device());
    }
    impl.exchangeStream(stream);
  });

  m.def("_accelerator_getStream", [](c10::DeviceIndex device_index) {
    const auto device_type = at::getAccelerator(true).value();
    torch::utils::maybe_initialize_device(device_type);
    c10::impl::VirtualGuardImpl impl(device_type);
    return impl.getStream({device_type, device_index});
  });

  m.def("_accelerator_synchronizeDevice", [](c10::DeviceIndex device_index) {
    const auto device_type = at::getAccelerator(true).value();
    if (torch::utils::is_lazy_initialized_supported(device_type) &&
        !torch::utils::is_device_initialized(device_type)) {
      return;
    }
    torch::utils::maybe_initialize_device(device_type);
    c10::impl::VirtualGuardImpl impl(device_type);
    // impl.synchronizeDevice should can be safely called from any device
    {
      py::gil_scoped_release no_gil;
      impl.synchronizeDevice(device_index);
    }
  });
}

} // namespace torch::accelerator
