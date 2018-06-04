#include "torch/csrc/python_headers.h"

#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/TCPStore.hpp>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/pybind.h"

namespace py = pybind11;

namespace torch {
namespace c10d {

namespace {

template <typename T>
py::class_<T, std::shared_ptr<T>> registerStore(
    py::module& m,
    const char* name) {
  return py::class_<T, std::shared_ptr<T>>(m, name)
      // Convert from std::string to std::vector<uint8>.
      .def(
          "set",
          [](T& store, const std::string& key, const std::string& value) {
            std::vector<unsigned char> value_(value.begin(), value.end());
            store.set(key, value_);
          },
          py::call_guard<py::gil_scoped_release>())
      // Convert from std::vector<uint8_t> to py::bytes.
      // The returned value is not guaranteed to be valid UTF-8.
      .def(
          "get",
          [](T& store, const std::string& key) -> py::bytes {
            auto value = store.get(key);
            return py::bytes(
                reinterpret_cast<char*>(value.data()), value.size());
          },
          py::call_guard<py::gil_scoped_release>())
      .def("add", &T::add, py::call_guard<py::gil_scoped_release>())
      .def("wait", &T::wait, py::call_guard<py::gil_scoped_release>());
}

PyObject* c10d_init(PyObject* _unused) {
  auto c10d_module = THPObjectPtr(PyImport_ImportModule("torch.c10d"));
  if (!c10d_module) {
    throw python_error();
  }

  auto m = py::handle(c10d_module).cast<py::module>();

  registerStore<::c10d::FileStore>(m, "FileStore")
      .def(py::init<const std::string&>());
  registerStore<::c10d::TCPStore>(m, "TCPStore")
      .def(py::init<const std::string&, int, bool>());

  Py_RETURN_TRUE;
}

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = {
    {"_c10d_init", (PyCFunction)c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace c10d
} // namespace torch
