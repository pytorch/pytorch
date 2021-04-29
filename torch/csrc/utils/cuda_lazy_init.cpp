#include <torch/csrc/utils/cuda_lazy_init.h>

#include <torch/csrc/python_headers.h>
#include <mutex>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
namespace torch {
namespace utils {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool run_yet = false;

void cuda_lazy_init() {
  pybind11::gil_scoped_acquire g;
  // Protected by the GIL.  We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (!run_yet) {
    auto module = THPObjectPtr(PyImport_ImportModule("torch.cuda"));
    if (!module) throw python_error();
    auto res = THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) throw python_error();
    run_yet = true;
  }
}

void set_run_yet_variable_to_false() {
  run_yet = false;
}

}
}
