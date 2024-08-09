#include <torch/csrc/utils/python_symnode.h>

namespace torch {

py::handle get_symint_class() {
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch").attr("SymInt");
      })
      .get_stored();
}

py::handle get_symfloat_class() {
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch").attr("SymFloat");
      })
      .get_stored();
}

py::handle get_symbool_class() {
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch").attr("SymBool");
      })
      .get_stored();
}

} // namespace torch
