#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/utils/device_lazy_init.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

#ifndef WIN32
#include <pthread.h>
#endif

namespace torch::utils {
namespace {

bool is_initialized_flag = false;
bool is_in_bad_fork_flag = false;

c10::once_flag at_fork_global_once_flag;
at::DeviceType at_fork_device_type_global;

} // anonymous namespace

bool is_device_initialized(at::DeviceType device_type) {
  pybind11::gil_scoped_acquire g;
  if (device_type != at_fork_device_type_global) {
    return false;
  }
  return is_initialized_flag;
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

  if (device_type == at::DeviceType::PrivateUse1) {
    auto has_lazy_init_method =
        PyObject_HasAttrString(module.get(), "_lazy_init") == 1;
    if (!has_lazy_init_method) {
      return;
    }
  }

  auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
  if (!res) {
    throw python_error();
  }

  is_initialized_flag = true;
}

void set_requires_device_init(at::DeviceType device_type, bool value) {
  if (device_type == at_fork_device_type_global) {
    is_initialized_flag = !value;
  }
}

bool is_device_in_bad_fork(at::DeviceType device_type) {
  if (device_type != at_fork_device_type_global) {
    return false;
  }
  return is_in_bad_fork_flag;
}

void set_device_in_bad_fork(at::DeviceType device_type, bool value) {
  if (device_type == at_fork_device_type_global) {
    is_in_bad_fork_flag = value;
  }
}

// Should be called before the first device runtime call.
void register_fork_handler_for_device_init(at::DeviceType device_type) {
#ifndef WIN32
  c10::call_once(at_fork_global_once_flag, [device_type]() {
    at_fork_device_type_global = device_type;
    pthread_atfork(nullptr, nullptr, []() {
      set_device_in_bad_fork(at_fork_device_type_global, true);
      if (is_device_lazy_init_supported(at_fork_device_type_global)) {
        set_requires_device_init(at_fork_device_type_global, true);
      }
    });
  });
  TORCH_CHECK(
      device_type == at_fork_device_type_global,
      "register_fork_handler_for_device_init only supports a single accelerator type per process. ",
      "First registered device type: ",
      at::DeviceTypeName(at_fork_device_type_global, /*lower_case=*/true),
      ", got: ",
      at::DeviceTypeName(device_type, /*lower_case=*/true));
#endif
}

} // namespace torch::utils
