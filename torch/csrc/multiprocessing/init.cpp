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
        multiprocessing_init,
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
