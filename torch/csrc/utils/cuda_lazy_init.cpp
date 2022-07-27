#include <torch/csrc/utils/cuda_lazy_init.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

namespace torch {
namespace utils {
namespace {

bool is_initialized = false;

}

void cuda_lazy_init() {
  pybind11::gil_scoped_acquire g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (is_initialized) {
    return;
  }

  auto module = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
  if (!module) {
    throw python_error();
  }

  auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
  if (!res) {
    throw python_error();
  }

  is_initialized = true;
}

void set_requires_cuda_init(bool value) {
  is_initialized = !value;
}

} // namespace utils
} // namespace torch
