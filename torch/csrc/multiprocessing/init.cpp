#include <c10/util/thread_name.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include <stdexcept>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#define SYSASSERT(rv, ...)                                                 \
  if ((rv) < 0) {                                                          \
    throw std::system_error(errno, std::system_category(), ##__VA_ARGS__); \
  }

namespace torch::multiprocessing {

namespace {

PyObject* multiprocessing_init(PyObject* _unused, PyObject* noargs) {
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

PyObject* set_thread_name(PyObject* _unused, PyObject* arg) {
  TORCH_CHECK(THPUtils_checkString(arg), "invalid argument to setDevice");

  auto name = THPUtils_unpackString(arg);
  c10::setThreadName(name);

  Py_RETURN_TRUE;
}

PyObject* get_thread_name(PyObject* _unused, PyObject* noargs) {
  return THPUtils_packString(c10::getThreadName());
}

} // namespace

// multiprocessing methods on torch._C
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef methods[] = {
    {
        "_multiprocessing_init",
        multiprocessing_init,
        METH_NOARGS,
        nullptr,
    },
    {
        "_set_thread_name",
        set_thread_name,
        METH_O,
        nullptr,
    },
    {
        "_get_thread_name",
        get_thread_name,
        METH_NOARGS,
        nullptr,
    },
    {nullptr, nullptr, 0, nullptr},
};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace torch::multiprocessing
