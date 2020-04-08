#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>

#include <stdexcept>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#define SYSASSERT(rv, ...)                                                 \
  if ((rv) < 0) {                                                          \
    throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); \
  }

namespace torch {
namespace multiprocessing {

namespace {

PyObject* multiprocessing_init(PyObject* _unused, PyObject *noargs) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
  TORCH_WARN(
    "torch.multiprocessing: your pytorch binary has address sanitizer (asan) built in, "
    "asan is currently not compatible with spawn-based (start method is 'spawn') multiprocessing "
    "which is provided in this module, you might get unexpected behavior (eg. missing attribute, crash, etc.), "
    "please rebuild pytorch without asan if you need spawn-based multiprocessing");
#endif
#endif
  auto multiprocessing_module =
      THPObjectPtr(PyImport_ImportModule("torch.multiprocessing"));
  if (!multiprocessing_module) {
    throw python_error();
  }

  auto module = py::handle(multiprocessing_module).cast<py::module>();

  module.def("_prctl_pr_set_pdeathsig", [](int signal) {
#if defined(__linux__)
    auto rv = prctl(PR_SET_PDEATHSIG, signal);
    SYSASSERT(rv, "prctl");
#endif
  });

  Py_RETURN_TRUE;
}

} // namespace

// multiprocessing methods on torch._C
static PyMethodDef methods[] = {
    {
        "_multiprocessing_init",
        (PyCFunction)multiprocessing_init,
        METH_NOARGS,
        nullptr,
    },
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace multiprocessing
} // namespace torch
