#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/utils/device_lazy_init.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>

#ifndef WIN32
#include <pthread.h>
#endif

namespace torch::utils {
namespace {

std::array<bool, at::COMPILE_TIME_MAX_DEVICE_TYPES> is_initialized{};
std::array<bool, at::COMPILE_TIME_MAX_DEVICE_TYPES> is_in_bad_fork{};
std::array<bool, at::COMPILE_TIME_MAX_DEVICE_TYPES> at_fork_registered{};
c10::once_flag at_fork_register_once{};

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

  is_initialized[static_cast<int>(device_type)] = true;
}

void set_requires_device_init(at::DeviceType device_type, bool value) {
  is_initialized[static_cast<int>(device_type)] = !value;
}

bool is_device_in_bad_fork(at::DeviceType device_type) {
  return is_in_bad_fork[static_cast<int>(device_type)];
}

void set_device_in_bad_fork(at::DeviceType device_type, bool value) {
  is_in_bad_fork[static_cast<int>(device_type)] = value;
}

// Should be called before the first device runtime call.
void register_fork_handler_for_device_init(at::DeviceType device_type) {
#ifndef WIN32
  at_fork_registered[static_cast<int>(device_type)] = true;
  c10::call_once(at_fork_register_once, []() {
    pthread_atfork(nullptr, nullptr, []() {
      for (int i = 0; i < static_cast<int>(at::COMPILE_TIME_MAX_DEVICE_TYPES);
           ++i) {
        if (!at_fork_registered[i]) {
          continue;
        }
        auto dt = static_cast<at::DeviceType>(i);
        set_device_in_bad_fork(dt, true);
        if (is_device_lazy_init_supported(dt)) {
          set_requires_device_init(dt, true);
        }
      }
    });
  });
#endif
}

} // namespace torch::utils
