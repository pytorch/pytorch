#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/utils/device_lazy_init.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <iostream>
namespace torch::utils {
namespace {

std::array<bool, at::COMPILE_TIME_MAX_DEVICE_TYPES> is_initialized{};

} // anonymous namespace

bool is_device_initialized(at::DeviceType device_type) {
  pybind11::gil_scoped_acquire g;
  return is_initialized[static_cast<int>(device_type)];
}

void device_lazy_init(at::DeviceType device_type) {
  pybind11::gil_scoped_acquire g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (is_device_initialized(device_type)) {
    return;
  }

  auto maybe_mode = c10::impl::TorchDispatchModeTLS::get_mode(
      c10::impl::TorchDispatchModeKey::FAKE);
  if (maybe_mode) {
    return;
  }

  std::string module_name = "torch." + at::DeviceTypeName(device_type, true);
  auto module = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
  if (!module) {
    throw python_error();
  }

  auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
  if (!res) {
    throw python_error();
  }

  is_initialized[static_cast<int>(device_type)] = true;
}

void set_requires_device_init(at::DeviceType device_type, bool value) {
  is_initialized[static_cast<int>(device_type)] = !value;
}

} // namespace torch::utils
